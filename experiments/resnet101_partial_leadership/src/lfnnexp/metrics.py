from __future__ import annotations

from dataclasses import dataclass

import torch

from .distributed import DistributedContext, reduce_sum


@dataclass
class ClassificationTotals:
    loss_sum: float = 0.0
    correct1: int = 0
    correct5: int = 0
    count: int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        batch = targets.numel()
        max_k = min(5, logits.shape[1])
        prediction = logits.topk(max_k, dim=1, largest=True, sorted=True).indices
        correct = prediction.eq(targets[:, None])
        self.loss_sum += float(loss.detach()) * batch
        self.correct1 += int(correct[:, :1].sum().item())
        self.correct5 += int(correct[:, :max_k].sum().item())
        self.count += batch

    def distributed_values(
        self, context: DistributedContext, device: torch.device
    ) -> dict[str, float]:
        raw = torch.tensor(
            [self.loss_sum, self.correct1, self.correct5, self.count],
            dtype=torch.float64,
            device=device,
        )
        reduced = reduce_sum(raw, context).cpu().tolist()
        loss_sum, correct1, correct5, count = reduced
        if count <= 0:
            raise RuntimeError("No evaluation examples were processed.")
        return {
            "loss": loss_sum / count,
            "top1": 100.0 * correct1 / count,
            "top5": 100.0 * correct5 / count,
            "count": int(count),
        }


def parameter_count(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }

