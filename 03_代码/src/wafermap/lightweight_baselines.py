from __future__ import annotations

import torch
from torch import nn
from torchvision.models import efficientnet_b0, mobilenet_v3_small

from wafermap.constants import NUM_CLASSES


class MobileNetV3SmallBaseline(nn.Module):
    """Single-channel MobileNetV3-Small trained without pretraining."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.network = mobilenet_v3_small(weights=None)

        first_conv = self.network.features[0][0]
        self.network.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=False,
        )

        final_linear = self.network.classifier[-1]
        self.network.classifier[-1] = nn.Linear(
            final_linear.in_features,
            num_classes,
        )

    def forward(self, images: torch.Tensor):
        return self.network(images)


class EfficientNetB0Baseline(nn.Module):
    """Single-channel EfficientNet-B0 trained without pretraining."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.network = efficientnet_b0(weights=None)

        first_conv = self.network.features[0][0]
        self.network.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=False,
        )

        final_linear = self.network.classifier[-1]
        self.network.classifier[-1] = nn.Linear(
            final_linear.in_features,
            num_classes,
        )

    def forward(self, images: torch.Tensor):
        return self.network(images)
