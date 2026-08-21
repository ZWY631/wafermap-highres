from __future__ import annotations

import torch
from torch import nn
from torchvision.models import shufflenet_v2_x1_0

from wafermap.constants import NUM_CLASSES


class ShuffleNetV2StemAblation(nn.Module):
    """ShuffleNetV2 with only the two input-stem factors varied."""

    def __init__(
        self,
        conv1_stride: int,
        keep_max_pool: bool,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        if conv1_stride not in (1, 2):
            raise ValueError("conv1_stride must be either 1 or 2")

        self.conv1_stride = conv1_stride
        self.keep_max_pool = keep_max_pool
        self.network = shufflenet_v2_x1_0(weights=None)

        self.network.conv1[0] = nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=3,
            stride=conv1_stride,
            padding=1,
            bias=False,
        )

        if not keep_max_pool:
            self.network.maxpool = nn.Identity()

        self.network.fc = nn.Linear(
            self.network.fc.in_features,
            num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class ShuffleNetV2Stride2NoPool(ShuffleNetV2StemAblation):
    """Intermediate stem: stride-two convolution without max pooling."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(
            conv1_stride=2,
            keep_max_pool=False,
            num_classes=num_classes,
        )


class ShuffleNetV2Stride1MaxPool(ShuffleNetV2StemAblation):
    """Intermediate stem: stride-one convolution with max pooling."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__(
            conv1_stride=1,
            keep_max_pool=True,
            num_classes=num_classes,
        )
