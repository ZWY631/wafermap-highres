#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.dataset import WaferMapDataset
from wafermap.transforms import build_train_transform


def main():
    train_transform = build_train_transform()

    dataset = WaferMapDataset(
        "train",
        transform=train_transform,
    )

    image, label = dataset[0]
    unique_values = torch.unique(image)

    if image.shape != (1, 64, 64):
        raise ValueError(
            f"Unexpected transformed shape: {image.shape}"
        )

    if image.min() < 0 or image.max() > 1:
        raise ValueError(
            "Transformed image is outside the [0, 1] range."
        )

    print(f"图像形状：{tuple(image.shape)}")
    print(f"标签编号：{int(label)}")
    print(
        "增强后像素值："
        f"{unique_values.tolist()}"
    )
    print("训练增强测试通过。")


if __name__ == "__main__":
    main()