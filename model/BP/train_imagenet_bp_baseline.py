#!/usr/bin/env python3
"""Paper-matched ImageNet BP baseline trainer for reviewer evidence.

The script intentionally writes a verbose text log rather than relying only on
stdout. It is designed to make the BP rows auditable: command/config, optimizer,
data pipeline, per-epoch metrics, checkpoint paths, and runtime are all recorded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet101_Weights, ResNet152_Weights, ViT_B_16_Weights


EXPERIMENT_PRESETS = {
    "resnet101": {
        "init": "scratch",
        "epochs": 90,
        "batch_size": 256,
        "num_workers": 8,
        "lr": 0.1,
        "eta_min": 0.0001,
        "weight_decay": 0.00005,
        "momentum": 0.9,
        "log_interval": 50,
        "checkpoint_every": 5,
        "scheduler_t_max": 89,
    },
    "resnet152": {
        "init": "scratch",
        "epochs": 90,
        "batch_size": 256,
        "num_workers": 8,
        "lr": 0.05,
        "eta_min": 0.00001,
        "weight_decay": 0.00005,
        "momentum": 0.9,
        "log_interval": 50,
        "checkpoint_every": 5,
        "scheduler_t_max": 89,
    },
    "vit_b_16": {
        "init": "swag_e2e_ignore_mismatch_224",
        "epochs": 90,
        "batch_size": 256,
        "num_workers": 8,
        "lr": 0.001,
        "eta_min": 0.0,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "log_interval": 50,
        "checkpoint_every": 5,
        "scheduler_t_max": 90,
    },
}


def resolve_setting(cli_value, env_name: str, preset_value, cast):
    """Resolve CLI > environment > experiment preset."""
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_name)
    return cast(env_value) if env_value is not None else preset_value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topk_correct(outputs: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...]) -> list[int]:
    _, pred = outputs.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
    return [int(correct[:k].reshape(-1).float().sum().item()) for k in topk]


def build_model(name: str, init: str) -> nn.Module:
    if name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V2 if init == "torchvision" else None
        return torchvision.models.resnet101(weights=weights)
    if name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V2 if init == "torchvision" else None
        return torchvision.models.resnet152(weights=weights)
    if name == "vit_b_16":
        if init in {"swag", "swag_e2e"}:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            model = torchvision.models.vit_b_16(weights=weights)
            model._lfnn_init_note = "torchvision_swag_e2e_native_384"  # type: ignore[attr-defined]
            return model
        if init == "swag_e2e_ignore_mismatch_224":
            model = torchvision.models.vit_b_16(weights=None, image_size=224)
            pretrained = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            model_state = model.state_dict()
            pretrained_state = pretrained.state_dict()
            matched = {
                key: value
                for key, value in pretrained_state.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            skipped = [
                {
                    "key": key,
                    "pretrained_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
                for key, value in pretrained_state.items()
                if key in model_state and value.shape != model_state[key].shape
            ]
            model_state.update(matched)
            model.load_state_dict(model_state)
            model._lfnn_init_note = "swag_e2e_loaded_into_224_model_ignore_mismatch"  # type: ignore[attr-defined]
            model._lfnn_init_matched_params = len(matched)  # type: ignore[attr-defined]
            model._lfnn_init_total_params = len(model_state)  # type: ignore[attr-defined]
            model._lfnn_init_skipped = skipped  # type: ignore[attr-defined]
            return model
        elif init == "swag_linear":
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        elif init == "torchvision":
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = torchvision.models.vit_b_16(weights=weights)
        model._lfnn_init_note = f"torchvision_{init}"  # type: ignore[attr-defined]
        return model
    raise ValueError(f"Unsupported model: {name}")


def make_loaders(data_root: Path, batch_size: int, num_workers: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = torchvision.datasets.ImageFolder(root=data_root / "train", transform=train_transform)
    val_set = torchvision.datasets.ImageFolder(root=data_root / "val", transform=val_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def log_line(handle, text: str) -> None:
    print(text, flush=True)
    handle.write(text + "\n")
    handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet101", "resnet152", "vit_b_16"], required=True)
    parser.add_argument("--data-root", default="model/LFNN-BPfree/ImageNet/imagenet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--init",
        choices=["scratch", "torchvision", "swag", "swag_e2e", "swag_e2e_ignore_mismatch_224", "swag_linear"],
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    args = parser.parse_args()

    preset = EXPERIMENT_PRESETS[args.model]
    args.init = resolve_setting(args.init, "LFNN_INIT", preset["init"], str)
    args.epochs = resolve_setting(args.epochs, "LFNN_EPOCHS", preset["epochs"], int)
    args.batch_size = resolve_setting(args.batch_size, "LFNN_BATCH_SIZE", preset["batch_size"], int)
    args.num_workers = resolve_setting(args.num_workers, "LFNN_NUM_WORKERS", preset["num_workers"], int)
    args.lr = resolve_setting(args.lr, "LFNN_LR", preset["lr"], float)
    args.eta_min = resolve_setting(args.eta_min, "LFNN_ETA_MIN", preset["eta_min"], float)
    args.weight_decay = resolve_setting(
        args.weight_decay, "LFNN_WEIGHT_DECAY", preset["weight_decay"], float
    )
    args.momentum = resolve_setting(args.momentum, "LFNN_MOMENTUM", preset["momentum"], float)
    args.log_interval = resolve_setting(
        args.log_interval, "LFNN_LOG_INTERVAL", preset["log_interval"], int
    )
    args.checkpoint_every = resolve_setting(
        args.checkpoint_every, "LFNN_CHECKPOINT_EVERY", preset["checkpoint_every"], int
    )
    args.scheduler_t_max = preset["scheduler_t_max"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{args.model}_bp_training_log.txt"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    start = time.time()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        config = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": args.model,
            "init": args.init,
            "data_root": str(Path(args.data_root).resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "optimizer": "SGD",
            "lr": args.lr,
            "eta_min": args.eta_min,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": args.scheduler_t_max,
            "preset": "released_experiment",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in device_ids],
        }
        log_line(log, "CONFIG " + json.dumps(config, sort_keys=True))

        train_loader, val_loader = make_loaders(Path(args.data_root), args.batch_size, args.num_workers)
        log_line(
            log,
            f"DATA train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
            f"classes={len(train_loader.dataset.classes)} train_batches={len(train_loader)} val_batches={len(val_loader)}",
        )

        model = build_model(args.model, args.init)
        log_line(
            log,
            "INIT "
            + json.dumps(
                {
                    "note": getattr(model, "_lfnn_init_note", ""),
                    "matched_params": getattr(model, "_lfnn_init_matched_params", None),
                    "total_params": getattr(model, "_lfnn_init_total_params", None),
                    "skipped": getattr(model, "_lfnn_init_skipped", None),
                },
                sort_keys=True,
            ),
        )
        if len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_t_max, eta_min=args.eta_min
        )

        best_top1 = 0.0
        best_path: Path | None = None
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            train_top1 = 0
            train_top5 = 0
            train_total = 0
            for step, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_size = targets.size(0)
                train_loss += float(loss.item())
                c1, c5 = topk_correct(outputs.detach(), targets, (1, 5))
                train_top1 += c1
                train_top5 += c5
                train_total += batch_size
                if step % args.log_interval == 0:
                    log_line(
                        log,
                        f"TRAIN epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.6f} "
                        f"top1={100.0 * train_top1 / train_total:.3f} top5={100.0 * train_top5 / train_total:.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f}",
                    )

            model.eval()
            val_loss = 0.0
            val_top1 = 0
            val_top5 = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += float(loss.item())
                    c1, c5 = topk_correct(outputs, targets, (1, 5))
                    val_top1 += c1
                    val_top5 += c5
                    val_total += targets.size(0)

            train_top1_pct = 100.0 * train_top1 / train_total
            train_top5_pct = 100.0 * train_top5 / train_total
            val_top1_pct = 100.0 * val_top1 / val_total
            val_top5_pct = 100.0 * val_top5 / val_total
            log_line(
                log,
                f"EPOCH epoch={epoch} train_loss={train_loss / len(train_loader):.6f} "
                f"train_top1={train_top1_pct:.3f} train_top5={train_top5_pct:.3f} "
                f"val_loss={val_loss / len(val_loader):.6f} val_top1={val_top1_pct:.3f} "
                f"val_top5={val_top5_pct:.3f} elapsed_sec={time.time() - epoch_start:.1f}",
            )

            state = model.state_dict()
            if val_top1_pct > best_top1:
                best_top1 = val_top1_pct
                best_path = checkpoint_dir / f"{args.model}_bp_best.pth"
                torch.save(state, best_path)
                log_line(log, f"CHECKPOINT best path={best_path} val_top1={best_top1:.3f}")
            if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
                path = checkpoint_dir / f"{args.model}_bp_epoch{epoch + 1}.pth"
                torch.save(state, path)
                log_line(log, f"CHECKPOINT periodic path={path}")
            scheduler.step()

        final_path = checkpoint_dir / f"{args.model}_bp_final.pth"
        torch.save(model.state_dict(), final_path)
        log_line(log, f"CHECKPOINT final path={final_path}")
        for path in sorted(checkpoint_dir.glob("*.pth")):
            log_line(log, f"ARTIFACT path={path} bytes={path.stat().st_size} sha256={sha256_file(path)}")
        log_line(
            log,
            f"DONE elapsed_sec={time.time() - start:.1f} best_top1={best_top1:.3f} "
            f"best_checkpoint={best_path if best_path else ''}",
        )
        log_line(
            log,
            f"FINAL_METRICS model={args.model} init={args.init} training=BP "
            f"loss=global_cross_entropy val_top1={val_top1_pct:.3f} val_top5={val_top5_pct:.3f} "
            f"top1_error={100.0 - val_top1_pct:.3f} top5_error={100.0 - val_top5_pct:.3f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
