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

from wafermap.dataset import WaferMapDataset
from torch import nn
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

    model = ResNet18Baseline()
    model.eval()

    with torch.no_grad():
        logits = model(images)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    expected_shape = (4, 9)

    if logits.shape != expected_shape:
        raise ValueError(
            f"Unexpected logits shape: {logits.shape}"
        )

    print(f"输入形状：{tuple(images.shape)}")
    print(f"输出形状：{tuple(logits.shape)}")
    print(f"标签形状：{tuple(labels.shape)}")
    print(f"模型参数量：{parameter_count:,}")
    print("ResNet18 前向传播测试通过。")


if __name__ == "__main__":
    main()
