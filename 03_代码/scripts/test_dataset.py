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

from wafermap.constants import IMAGE_SIZE, NUM_CLASSES
from wafermap.dataset import WaferMapDataset


def main():
    for split_name in ("train", "val", "test"):
        dataset = WaferMapDataset(split_name)

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=(split_name == "train"),
            num_workers=0,
        )

        images, labels = next(iter(loader))

        expected_shape = (
            32,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        if images.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape: {images.shape}"
            )

        if images.dtype != torch.float32:
            raise TypeError(
                f"Unexpected image dtype: {images.dtype}"
            )

        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            raise ValueError("Label id is out of range.")

        print(
            f"{split_name}: "
            f"dataset={len(dataset):,}, "
            f"batch={tuple(images.shape)}, "
            f"labels={labels.tolist()[:8]}"
        )


if __name__ == "__main__":
    main()