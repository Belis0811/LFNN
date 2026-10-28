from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .distributed import DistributedContext
from .metrics import ClassificationTotals


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    name = str(config["optimizer"]).lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config["learning_rate"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            nesterov=bool(config.get("nesterov", False)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, config: dict[str, Any]) -> LRScheduler:
    name = str(config["scheduler"]).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["scheduler_t_max"]),
            eta_min=float(config.get("scheduler_eta_min", 0.0)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["scheduler_step_size"]),
            gamma=float(config.get("scheduler_gamma", 0.1)),
        )
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=1
        )
    raise ValueError(f"Unsupported scheduler: {name}")


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader[Any],
    context: DistributedContext,
    *,
    amp: bool,
    lfnn_model: bool,
) -> dict[str, float]:
    evaluation_model = model.module if hasattr(model, "module") else model
    evaluation_model.eval()
    totals = ClassificationTotals()
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    for images, targets in loader:
        images = images.to(context.device, non_blocking=True)
        targets = targets.to(context.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            result = evaluation_model(images)
            logits = result[0] if lfnn_model else result
            loss = criterion(logits, targets)
        totals.update(logits, targets, loss)
    torch.cuda.synchronize(context.device)
    metrics = totals.distributed_values(context, context.device)
    metrics["elapsed_seconds_local_rank"] = time.perf_counter() - started
    return metrics


def ensure_new_or_resumed_run(output_dir: str | Path, resume: str | None) -> Path:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    existing = [
        destination / "run_metadata.json",
        destination / "metrics.jsonl",
        destination / "selection_trace.jsonl",
    ]
    if resume is None and any(path.exists() for path in existing):
        names = ", ".join(path.name for path in existing if path.exists())
        raise FileExistsError(
            f"Refusing to append a new run to {destination}; found {names}. "
            "Choose a new output directory or pass --resume."
        )
    return destination


def learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def clip_gradients_if_requested(
    model: nn.Module, optimizer: Optimizer, scaler: torch.cuda.amp.GradScaler, value: Any
) -> float | None:
    scaler.unscale_(optimizer)
    if value is None:
        return None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(value))
    return float(norm.detach())


def safe_rate(count: int, elapsed: float) -> float:
    return float(count) / max(elapsed, 1e-12)


def finite_or_raise(loss: torch.Tensor, step: int) -> None:
    if not math.isfinite(float(loss.detach())):
        raise FloatingPointError(f"Nonfinite loss at global step {step}: {loss.item()}")
