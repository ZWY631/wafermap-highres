#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.dataset import WaferMapDataset
from torch import nn
from wafermap.models_improved import ShuffleNetV2HighResECA
from wafermap.transforms import build_train_transform


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    device = select_device()
    print(f"Device: {device}")

    dataset = WaferMapDataset(
        "train",
        transform=build_train_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    images, labels = next(iter(loader))

    images = images.to(device)
    labels = labels.to(device)

    model = ShuffleNetV2HighResECA().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise ValueError(
            "Loss is NaN or infinity before backward."
        )

    loss.backward()

    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(
                parameter.grad
            ).all():
                raise ValueError(
                    "Gradient contains NaN or infinity."
                )

    optimizer.step()
    synchronize(device)

    print(f"Batch shape: {tuple(images.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Focal Loss: {loss.item():.6f}")
    print("One-batch training smoke test passed.")


if __name__ == "__main__":
    main()