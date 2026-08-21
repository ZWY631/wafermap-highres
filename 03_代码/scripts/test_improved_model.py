#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES
from wafermap.dataset import WaferMapDataset
from wafermap.models_improved import ShuffleNetV2HighResECA


def main():
    dataset = WaferMapDataset("train")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    model = ShuffleNetV2ECA()
    model.eval()

    with torch.no_grad():
        logits = model(images)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    expected_shape = (4, NUM_CLASSES)

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Unexpected logits shape: {tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "Model output contains NaN or infinity."
        )

    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Label shape: {tuple(labels.shape)}")
    print(f"Parameter count: {parameter_count:,}")
    print(
        "ECA kernel size: "
        f"{model.eca.channel_conv.kernel_size[0]}"
    )
    print("ShuffleNetV2-ECA forward test passed.")


if __name__ == "__main__":
    main()