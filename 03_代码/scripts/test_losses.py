#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from torch import nn


def main():
    torch.manual_seed(42)

    logits = torch.randn(
        4,
        9,
        requires_grad=True,
    )

    targets = torch.tensor(
        [0, 1, 2, 8],
        dtype=torch.long,
    )

    cross_entropy = F.cross_entropy(
        logits,
        targets,
    )

    focal_gamma_zero = FocalLoss(
        gamma=0.0
    )(
        logits,
        targets,
    )

    focal_gamma_two = FocalLoss(
        gamma=2.0
    )(
        logits,
        targets,
    )

    if not torch.allclose(
        cross_entropy,
        focal_gamma_zero,
        atol=1e-6,
    ):
        raise ValueError(
            "Focal Loss with gamma=0 "
            "does not match Cross Entropy."
        )

    if focal_gamma_two > cross_entropy:
        raise ValueError(
            "Focal Loss should not exceed "
            "Cross Entropy for this batch."
        )

    focal_gamma_two.backward()

    if logits.grad is None:
        raise ValueError(
            "No gradient was produced."
        )

    if not torch.isfinite(logits.grad).all():
        raise ValueError(
            "Gradient contains NaN or infinity."
        )

    print(
        f"Cross Entropy: {cross_entropy.item():.6f}"
    )
    print(
        "Focal Loss gamma=0: "
        f"{focal_gamma_zero.item():.6f}"
    )
    print(
        "Focal Loss gamma=2: "
        f"{focal_gamma_two.item():.6f}"
    )
    print("Focal Loss test passed.")


if __name__ == "__main__":
    main()