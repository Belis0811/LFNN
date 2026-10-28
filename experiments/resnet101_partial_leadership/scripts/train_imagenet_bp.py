#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from lfnnexp.checkpoints import save_checkpoint, unwrap_model
from lfnnexp.config import apply_cli_overrides, load_config, save_config
from lfnnexp.data import build_imagenet_loaders
from lfnnexp.distributed import (
    barrier,
    cleanup,
    initialize_distributed,
    seed_everything,
)
from lfnnexp.metrics import ClassificationTotals, parameter_count
from lfnnexp.models import build_bp_model
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
    parser = argparse.ArgumentParser(description="Train an end-to-end BP ImageNet model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Nonreported smoke tests only; stored in run metadata.",
    )
    return parser.parse_args()


def source_files(root: Path, config_path: str) -> list[Path]:
    return [
        Path(config_path).resolve(),
        *sorted((root / "src").rglob("*.py")),
        Path(__file__).resolve(),
    ]


def restore(
    path: str,
    model: nn.Module,
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
    context = initialize_distributed()
    writer: JsonlWriter | None = None
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
                source_files=source_files(package_root, config["_config_path"]),
                git=git,
            )
            metadata["max_steps_per_epoch"] = args.max_steps_per_epoch
            write_json(output_dir / "run_metadata.json", metadata)
        barrier(context)

        model_config = dict(config["model"])
        model_config["num_classes"] = int(config["data"]["num_classes"])
        model = build_bp_model(model_config).to(context.device)
        if bool(config["runtime"].get("sync_batch_norm", False)):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        counts = parameter_count(model)
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
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
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(config["optimization"].get("label_smoothing", 0.0))
        )

        start_epoch, best_top1, global_step = 0, float("-inf"), 0
        if args.resume:
            start_epoch, best_top1, global_step = restore(
                args.resume, model, optimizer, scheduler, scaler
            )
        if context.is_main:
            writer = JsonlWriter(output_dir / "metrics.jsonl")
            writer.write(
                {
                    "event": "run_start" if args.resume is None else "run_resume",
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
        checkpoint_every = int(
            config["logging"].get("checkpoint_every_epochs", 10)
        )
        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            totals = ClassificationTotals()
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
                    logits = model(images)
                    loss = criterion(logits, targets)
                finite_or_raise(loss, global_step)
                scaler.scale(loss).backward()
                clipped_norm = clip_gradients_if_requested(
                    model,
                    optimizer,
                    scaler,
                    config["optimization"].get("gradient_clip_norm"),
                )
                scaler.step(optimizer)
                scaler.update()
                totals.update(logits, targets, loss)
                if context.is_main and batch_index % print_frequency == 0:
                    print(
                        f"epoch={epoch:03d} batch={batch_index:05d} "
                        f"step={global_step:07d} loss={float(loss):.6f} "
                        f"grad_norm_before_clip={clipped_norm}",
                        flush=True,
                    )
                global_step += 1
            torch.cuda.synchronize(context.device)
            elapsed = time.perf_counter() - epoch_started
            train_metrics = totals.distributed_values(context, context.device)
            scheduler.step()
            val_metrics = evaluate_classifier(
                model, val_loader, context, amp=amp, lfnn_model=False
            )
            is_best = val_metrics["top1"] > best_top1
            best_top1 = max(best_top1, val_metrics["top1"])
            if context.is_main:
                assert writer is not None
                record = {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate_start": lr_start,
                    "learning_rate_next_epoch": learning_rate(optimizer),
                    "train": train_metrics,
                    "validation": val_metrics,
                    "epoch_elapsed_seconds_local_rank": elapsed,
                    "best_top1_so_far": best_top1,
                    "is_best": is_best,
                }
                writer.write(record)
                checkpoint = {
                    "format_version": 1,
                    "kind": "bp",
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
                    "epoch_metrics": record,
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
        if context.is_main and writer is not None:
            writer.write(
                {
                    "event": "run_end",
                    "epochs_completed": epochs,
                    "global_step": global_step,
                    "best_top1": best_top1,
                }
            )
    finally:
        if writer is not None:
            writer.close()
        cleanup(context)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise

