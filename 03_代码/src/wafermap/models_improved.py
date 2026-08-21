from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models import shufflenet_v2_x1_0

from wafermap.constants import NUM_CLASSES


class ECALayer(nn.Module):
    """Efficient Channel Attention for a 4-D feature map."""

    def __init__(
        self,
        channels: int,
        gamma: int = 2,
        bias: int = 1,
    ):
        super().__init__()

        kernel_size = int(
            abs(
                (
                    math.log2(channels)
                    + bias
                )
                / gamma
            )
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel_size = max(kernel_size, 3)

        self.average_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

        self.activation = nn.Sigmoid()

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.average_pool(features)

        attention = attention.squeeze(-1)
        attention = attention.transpose(-1, -2)

        attention = self.channel_conv(attention)
        attention = self.activation(attention)

        attention = attention.transpose(-1, -2)
        attention = attention.unsqueeze(-1)

        return features * attention.expand_as(features)


class ShuffleNetV2ECA(nn.Module):
    """Lightweight ShuffleNetV2 classifier with ECA."""

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

        feature_channels = (
            self.network.fc.in_features
        )

        self.eca = ECALayer(
            channels=feature_channels
        )

        self.network.fc = nn.Linear(
            feature_channels,
            num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.network.conv1(images)
        features = self.network.maxpool(features)
        features = self.network.stage2(features)
        features = self.network.stage3(features)
        features = self.network.stage4(features)
        features = self.network.conv5(features)

        features = self.eca(features)

        features = features.mean(
            dim=(2, 3)
        )

        return self.network.fc(features)


class ShuffleNetV2HighResECA(
    ShuffleNetV2ECA
):
    """Preserve spatial details for small wafer patterns."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__(
            num_classes=num_classes
        )

        self.network.conv1[0] = nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.network.maxpool = nn.Identity()

class ShuffleNetV2HighRes(nn.Module):
    """High-resolution ShuffleNetV2 without ECA."""

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
            stride=1,
            padding=1,
            bias=False,
        )

        self.network.maxpool = nn.Identity()

        feature_channels = (
            self.network.fc.in_features
        )

        self.network.fc = nn.Linear(
            feature_channels,
            num_classes,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.network.conv1(images)
        features = self.network.maxpool(features)
        features = self.network.stage2(features)
        features = self.network.stage3(features)
        features = self.network.stage4(features)
        features = self.network.conv5(features)

        features = features.mean(
            dim=(2, 3)
        )

        return self.network.fc(features)        
