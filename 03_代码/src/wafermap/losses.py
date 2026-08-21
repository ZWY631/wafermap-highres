from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss for imbalanced classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        if gamma < 0:
            raise ValueError(
                "gamma must be non-negative."
            )

        valid_reductions = {
            "none",
            "mean",
            "sum",
        }

        if reduction not in valid_reductions:
            raise ValueError(
                f"Unknown reduction: {reduction}"
            )

        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        probability = torch.exp(
            -cross_entropy
        )

        focal_factor = (
            1.0 - probability
        ).pow(self.gamma)

        loss = focal_factor * cross_entropy

        if self.reduction == "none":
            return loss

        if self.reduction == "sum":
            return loss.sum()

        return loss.mean()