from __future__ import annotations

import unittest

import torch

from lfnnexp.distributed import DistributedContext
from lfnnexp.lfnn_loss import (
    compute_lfnn_loss,
    realized_leader_count,
    select_global_minibatch_leaders,
)


class LFNNLossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = DistributedContext(
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )

    def test_delta_75_has_one_follower_for_four_workers(self) -> None:
        self.assertEqual(realized_leader_count(4, 0.75, "ceil"), 3)

    def test_delta_90_is_all_leader_for_four_workers(self) -> None:
        self.assertEqual(realized_leader_count(4, 0.90, "ceil"), 4)

    def test_lowest_cross_entropy_worker_is_best_leader(self) -> None:
        targets = torch.tensor([0, 1])
        logits = [
            torch.tensor([[8.0, -8.0], [-8.0, 8.0]]),
            torch.zeros(2, 2),
            torch.tensor([[-8.0, 8.0], [8.0, -8.0]]),
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
        ]
        selection = select_global_minibatch_leaders(
            logits,
            targets,
            fraction=0.75,
            rounding="ceil",
            label_smoothing=0.0,
            context=self.context,
        )
        self.assertEqual(selection.best_leader_id, 0)
        self.assertEqual(len(selection.leader_ids), 3)
        self.assertEqual(len(selection.follower_ids), 1)

    def test_follower_loss_updates_follower_only_through_detached_teacher(self) -> None:
        targets = torch.tensor([0, 1])
        local_logits = [
            torch.tensor([[8.0, -8.0], [-8.0, 8.0]], requires_grad=True),
            torch.zeros(2, 2, requires_grad=True),
            torch.tensor([[-8.0, 8.0], [8.0, -8.0]], requires_grad=True),
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]], requires_grad=True),
        ]
        output = torch.zeros(2, 2, requires_grad=True)
        config = {
            "leadership_fraction": 0.75,
            "leader_rounding": "ceil",
            "require_followers": True,
            "selection_scope": "global_minibatch",
            "alignment_space": "probabilities",
            "detach_teacher": True,
            "output_loss_weight": 1.0,
            "leader_loss_weight": 0.5,
            "follower_loss_weight": 0.5,
            "label_smoothing": 0.0,
        }
        result = compute_lfnn_loss(
            output, local_logits, targets, config, self.context
        )
        result.total.backward()
        follower = result.selection.follower_ids[0]
        self.assertGreater(float(local_logits[follower].grad.norm()), 0.0)
        self.assertGreater(float(result.follower), 0.0)


if __name__ == "__main__":
    unittest.main()

