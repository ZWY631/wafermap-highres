from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, shufflenet_v2_x1_0

from wafermap.constants import NUM_CLASSES


class ResNet18Baseline(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.network = resnet18(
            weights=None
        )

        self.network.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.network.fc = nn.Linear(
            self.network.fc.in_features,
            num_classes,
        )

    def forward(self, images: torch.Tensor):
        return self.network(images)


class ShuffleNetV2Baseline(nn.Module):
    """Standard-downsampling torchvision ShuffleNetV2 x1.0."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.network = shufflenet_v2_x1_0(
            weights=None
        )

        self.network.conv1[0] = nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        self.network.fc = nn.Linear(
            self.network.fc.in_features,
            num_classes,
        )

    def forward(self, images: torch.Tensor):
        return self.network(images)
