from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .distributed import DistributedContext


@dataclass(frozen=True)
class Selection:
    leader_ids: tuple[int, ...]
    follower_ids: tuple[int, ...]
    best_leader_id: int
    global_worker_losses: tuple[float, ...]


@dataclass
class LFNNLossOutput:
    total: torch.Tensor
    output: torch.Tensor
    leader: torch.Tensor
    follower: torch.Tensor
    selection: Selection


def realized_leader_count(
    worker_count: int, fraction: float, rounding: str = "ceil"
) -> int:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("leadership fraction must be in (0, 1]")
    raw = fraction * worker_count
    if rounding == "ceil":
        count = math.ceil(raw)
    elif rounding == "floor":
        count = math.floor(raw)
    elif rounding == "round":
        count = round(raw)
    else:
        raise ValueError(f"Unsupported leader rounding: {rounding}")
    return min(worker_count, max(1, int(count)))


def select_global_minibatch_leaders(
    local_logits: list[torch.Tensor],
    targets: torch.Tensor,
    *,
    fraction: float,
    rounding: str,
    label_smoothing: float,
    context: DistributedContext,
) -> Selection:
    if not local_logits:
        raise ValueError("At least one worker output is required.")
    with torch.no_grad():
        loss_sums = torch.stack(
            [
                F.cross_entropy(
                    logits,
                    targets,
                    reduction="sum",
                    label_smoothing=label_smoothing,
                )
                for logits in local_logits
            ]
        )
        count = torch.tensor(
            float(targets.numel()), dtype=loss_sums.dtype, device=targets.device
        )
        if context.distributed:
            dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        mean_losses = loss_sums / count
        leader_count = realized_leader_count(len(local_logits), fraction, rounding)
        ordered = torch.argsort(mean_losses, stable=True).tolist()
        leaders = tuple(sorted(int(index) for index in ordered[:leader_count]))
        followers = tuple(index for index in range(len(local_logits)) if index not in leaders)
        best = int(ordered[0])
        return Selection(
            leader_ids=leaders,
            follower_ids=followers,
            best_leader_id=best,
            global_worker_losses=tuple(float(x) for x in mean_losses.cpu().tolist()),
        )


def compute_lfnn_loss(
    output_logits: torch.Tensor,
    local_logits: list[torch.Tensor],
    targets: torch.Tensor,
    config: dict[str, Any],
    context: DistributedContext,
) -> LFNNLossOutput:
    if config.get("selection_scope") != "global_minibatch":
        raise ValueError("This implementation supports selection_scope=global_minibatch.")
    smoothing = float(config.get("label_smoothing", 0.0))
    selection = select_global_minibatch_leaders(
        local_logits,
        targets,
        fraction=float(config["leadership_fraction"]),
        rounding=str(config.get("leader_rounding", "ceil")),
        label_smoothing=smoothing,
        context=context,
    )
    if bool(config.get("require_followers", False)) and not selection.follower_ids:
        raise RuntimeError(
            "The realized follower set is empty. Reduce leadership_fraction or "
            "change worker construction before running the requested experiment."
        )
    output_loss = F.cross_entropy(
        output_logits,
        targets,
        label_smoothing=smoothing,
    )
    leader_losses = [
        F.cross_entropy(
            local_logits[index],
            targets,
            label_smoothing=smoothing,
        )
        for index in selection.leader_ids
    ]
    leader_loss = torch.stack(leader_losses).mean()

    alignment_space = str(config.get("alignment_space", "probabilities"))
    teacher = local_logits[selection.best_leader_id]
    if alignment_space == "probabilities":
        teacher = teacher.softmax(dim=1)
    elif alignment_space != "logits":
        raise ValueError(f"Unsupported alignment space: {alignment_space}")
    if bool(config.get("detach_teacher", True)):
        teacher = teacher.detach()
    follower_losses = []
    for index in selection.follower_ids:
        student = local_logits[index]
        if alignment_space == "probabilities":
            student = student.softmax(dim=1)
        follower_losses.append(F.mse_loss(student, teacher))
    follower_loss = (
        torch.stack(follower_losses).mean()
        if follower_losses
        else output_loss.new_zeros(())
    )
    total = (
        float(config.get("output_loss_weight", 1.0)) * output_loss
        + float(config.get("leader_loss_weight", 0.5)) * leader_loss
        + float(config.get("follower_loss_weight", 0.5)) * follower_loss
    )
    return LFNNLossOutput(
        total=total,
        output=output_loss,
        leader=leader_loss,
        follower=follower_loss,
        selection=selection,
    )


def worker_gradient_norms(model: Any) -> list[float]:
    values: list[float] = []
    for index in range(len(model.workers)):
        squared = 0.0
        for parameter in model.worker_parameters(index):
            if parameter.grad is not None:
                squared += float(parameter.grad.detach().float().norm(2).item()) ** 2
        values.append(math.sqrt(squared))
    return values

