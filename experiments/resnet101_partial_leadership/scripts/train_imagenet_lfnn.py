#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from lfnnexp.checkpoints import save_checkpoint, unwrap_model
from lfnnexp.config import apply_cli_overrides, load_config, save_config
from lfnnexp.data import build_imagenet_loaders
from lfnnexp.distributed import (
    barrier,
    cleanup,
    initialize_distributed,
    reduce_sum,
    seed_everything,
)
from lfnnexp.lfnn_loss import (
    compute_lfnn_loss,
    realized_leader_count,
    worker_gradient_norms,
)
from lfnnexp.metrics import ClassificationTotals, parameter_count
from lfnnexp.models import LFNNResNet101
from lfnnexp.provenance import (
    JsonlWriter,
    assert_clean_git,
    collect_metadata,
    write_json,
)
from lfnnexp.training import (
    build_optimizer,
    build_scheduler,
    clip_gradients_if_requested,
    ensure_new_or_resumed_run,
    evaluate_classifier,
    finite_or_raise,
    learning_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train four-worker LFNN-l ResNet-101 on ImageNet-1K."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Nonreported smoke tests only; the value is stored in run metadata.",
    )
    return parser.parse_args()


def critical_source_files(package_root: Path, config_path: str) -> list[Path]:
    files = [Path(config_path).resolve()]
    files.extend(sorted((package_root / "src").rglob("*.py")))
    files.append(Path(__file__).resolve())
    return files


def load_resume(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_top1", float("-inf"))),
        int(checkpoint.get("global_step", 0)),
    )


def main() -> None:
    args = parse_args()
    package_root = Path(__file__).resolve().parents[1]
    config = apply_cli_overrides(
        load_config(args.config),
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
    if config["model"]["architecture"] != "resnet101":
        raise ValueError("This trainer currently implements LFNN-l ResNet-101 only.")

    context = initialize_distributed()
    metrics_writer: JsonlWriter | None = None
    trace_writer: JsonlWriter | None = None
    try:
        git = assert_clean_git(package_root, args.allow_dirty)
        deterministic = bool(config["runtime"].get("deterministic", False))
        seed_everything(int(config["experiment"]["seed"]), context.rank, deterministic)
        torch.backends.cudnn.benchmark = bool(
            config["runtime"].get("cudnn_benchmark", not deterministic)
        ) and not deterministic

        output_dir = ensure_new_or_resumed_run(
            config["experiment"]["output_dir"], args.resume
        )
        if context.is_main and args.resume is None:
            save_config(config, output_dir / "resolved_config.yaml")
            metadata = collect_metadata(
                config=config,
                source_files=critical_source_files(
                    package_root, config["_config_path"]
                ),
                git=git,
            )
            metadata["max_steps_per_epoch"] = args.max_steps_per_epoch
            write_json(output_dir / "run_metadata.json", metadata)
        barrier(context)

        model_config = dict(config["model"])
        model_config["num_classes"] = int(config["data"]["num_classes"])
        model_config.pop("architecture", None)
        model = LFNNResNet101(**model_config).to(context.device)
        if bool(config["runtime"].get("sync_batch_norm", False)):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        counts = parameter_count(model)
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
            )

        worker_count = len(unwrap_model(model).workers)
        lfnn_config = dict(config["lfnn"])
        lfnn_config["label_smoothing"] = float(
            config["optimization"].get("label_smoothing", 0.0)
        )
        leader_count = realized_leader_count(
            worker_count,
            float(lfnn_config["leadership_fraction"]),
            str(lfnn_config.get("leader_rounding", "ceil")),
        )
        follower_count = worker_count - leader_count
        if lfnn_config.get("require_followers") and follower_count == 0:
            raise RuntimeError(
                f"Configured delta realizes {leader_count} leaders and no followers."
            )

        train_loader, val_loader, train_sampler = build_imagenet_loaders(
            config["data"],
            int(config["optimization"]["global_batch_size"]),
            context,
        )
        optimizer = build_optimizer(model, config["optimization"])
        scheduler = build_scheduler(optimizer, config["optimization"])
        amp = bool(config["optimization"].get("amp", True))
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

        start_epoch = 0
        best_top1 = float("-inf")
        global_step = 0
        if args.resume:
            start_epoch, best_top1, global_step = load_resume(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
        if context.is_main:
            metrics_writer = JsonlWriter(output_dir / "metrics.jsonl")
            trace_writer = JsonlWriter(output_dir / "selection_trace.jsonl")
            metrics_writer.write(
                {
                    "event": "run_start" if args.resume is None else "run_resume",
                    "worker_count": worker_count,
                    "realized_leader_count": leader_count,
                    "realized_follower_count": follower_count,
                    "parameter_count": counts,
                    "world_size": context.world_size,
                    "local_batch_size": train_loader.batch_size,
                    "train_examples": len(train_loader.dataset),
                    "validation_examples": len(val_loader.dataset),
                    "optimizer": config["optimization"],
                    "data_loader": {
                        key: config["data"].get(key)
                        for key in (
                            "num_workers",
                            "pin_memory",
                            "persistent_workers",
                            "prefetch_factor",
                        )
                    },
                }
            )

        epochs = int(config["optimization"]["epochs"])
        print_frequency = int(config["logging"].get("print_frequency", 50))
        trace_every = int(config["logging"].get("trace_every_steps", 1))
        checkpoint_every = int(
            config["logging"].get("checkpoint_every_epochs", 10)
        )

        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            totals = ClassificationTotals()
            loss_sums = torch.zeros(4, dtype=torch.float64, device=context.device)
            sample_count = 0
            leader_frequency: Counter[int] = Counter()
            best_frequency: Counter[int] = Counter()
            epoch_started = time.perf_counter()
            lr_start = learning_rate(optimizer)

            for batch_index, (images, targets) in enumerate(train_loader):
                if (
                    args.max_steps_per_epoch is not None
                    and batch_index >= args.max_steps_per_epoch
                ):
                    break
                images = images.to(context.device, non_blocking=True)
                targets = targets.to(context.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    output_logits, local_logits = model(images)
                    result = compute_lfnn_loss(
                        output_logits,
                        local_logits,
                        targets,
                        lfnn_config,
                        context,
                    )
                finite_or_raise(result.total, global_step)
                scaler.scale(result.total).backward()
                clipped_norm = clip_gradients_if_requested(
                    model,
                    optimizer,
                    scaler,
                    config["optimization"].get("gradient_clip_norm"),
                )
                grad_norms = worker_gradient_norms(unwrap_model(model))
                scaler.step(optimizer)
                scaler.update()

                batch_size = targets.numel()
                totals.update(output_logits, targets, result.output)
                sample_count += batch_size
                loss_sums += torch.tensor(
                    [
                        float(result.total.detach()) * batch_size,
                        float(result.output.detach()) * batch_size,
                        float(result.leader.detach()) * batch_size,
                        float(result.follower.detach()) * batch_size,
                    ],
                    dtype=torch.float64,
                    device=context.device,
                )
                leader_frequency.update(result.selection.leader_ids)
                best_frequency.update([result.selection.best_leader_id])

                if context.is_main and global_step % trace_every == 0:
                    assert trace_writer is not None
                    trace_writer.write(
                        {
                            "event": "selection",
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "global_step": global_step,
                            "leader_ids": result.selection.leader_ids,
                            "follower_ids": result.selection.follower_ids,
                            "best_leader_id": result.selection.best_leader_id,
                            "global_worker_cross_entropy": result.selection.global_worker_losses,
                            "follower_alignment_loss": float(result.follower.detach()),
                            "worker_gradient_norms": grad_norms,
                            "gradient_norm_before_clip": clipped_norm,
                            "learning_rate": learning_rate(optimizer),
                        }
                    )
                if context.is_main and batch_index % print_frequency == 0:
                    print(
                        f"epoch={epoch:03d} batch={batch_index:05d} "
                        f"step={global_step:07d} total={float(result.total):.6f} "
                        f"output={float(result.output):.6f} "
                        f"leader={float(result.leader):.6f} "
                        f"follower={float(result.follower):.8f} "
                        f"leaders={list(result.selection.leader_ids)} "
                        f"followers={list(result.selection.follower_ids)}",
                        flush=True,
                    )
                global_step += 1

            torch.cuda.synchronize(context.device)
            epoch_elapsed = time.perf_counter() - epoch_started
            reduced_losses = reduce_sum(loss_sums, context).cpu().tolist()
            reduced_count = float(
                reduce_sum(
                    torch.tensor(sample_count, dtype=torch.float64, device=context.device),
                    context,
                ).item()
            )
            train_metrics = totals.distributed_values(context, context.device)
            train_metrics.update(
                {
                    "total_loss": reduced_losses[0] / reduced_count,
                    "output_loss": reduced_losses[1] / reduced_count,
                    "leader_loss": reduced_losses[2] / reduced_count,
                    "follower_loss": reduced_losses[3] / reduced_count,
                }
            )
            scheduler.step()
            val_metrics = evaluate_classifier(
                model,
                val_loader,
                context,
                amp=amp,
                lfnn_model=True,
            )
            is_best = val_metrics["top1"] > best_top1
            best_top1 = max(best_top1, val_metrics["top1"])

            if context.is_main:
                assert metrics_writer is not None
                epoch_record = {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate_start": lr_start,
                    "learning_rate_next_epoch": learning_rate(optimizer),
                    "train": train_metrics,
                    "validation": val_metrics,
                    "epoch_elapsed_seconds_local_rank": epoch_elapsed,
                    "leader_selection_count": dict(sorted(leader_frequency.items())),
                    "best_leader_count": dict(sorted(best_frequency.items())),
                    "best_top1_so_far": best_top1,
                    "is_best": is_best,
                }
                metrics_writer.write(epoch_record)
                checkpoint = {
                    "format_version": 1,
                    "kind": "lfnn_resnet101",
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_top1": best_top1,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": {
                        k: v for k, v in config.items() if not k.startswith("_")
                    },
                    "epoch_metrics": epoch_record,
                    "git": git,
                }
                save_checkpoint(output_dir / "checkpoints" / "last.pt", checkpoint)
                if is_best and bool(config["logging"].get("save_best", True)):
                    save_checkpoint(output_dir / "checkpoints" / "best.pt", checkpoint)
                if (epoch + 1) % checkpoint_every == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt",
                        checkpoint,
                    )
            barrier(context)

        if context.is_main and metrics_writer is not None:
            metrics_writer.write(
                {
                    "event": "run_end",
                    "epochs_completed": epochs,
                    "global_step": global_step,
                    "best_top1": best_top1,
                }
            )
    finally:
        if metrics_writer is not None:
            metrics_writer.close()
        if trace_writer is not None:
            trace_writer.close()
        cleanup(context)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise

