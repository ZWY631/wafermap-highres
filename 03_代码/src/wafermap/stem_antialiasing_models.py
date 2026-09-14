"""Anti-aliased stem variants for the WM-811K ShuffleNetV2 ablations.

Motivation
----------
The frozen 2x2 stem ablation (S2P / S2N / S1P / S1N) shows that *both* the
first-convolution stride and the presence of the initial max pooling affect
validation Macro-F1. Two mechanisms can produce that pattern:

1. **Resolution**: a stride-two stem discards spatial samples, and narrow
   wafer patterns (Scratch, Loc, Edge-Loc) lose the structure they need.
2. **Aliasing**: subsampling a signal without low-pass filtering folds
   high-frequency content back into the passband, so a thin scratch can be
   destroyed even when the sampling grid would have been sufficient
   (Zhang, "Making Convolutional Networks Shift-Invariant Again", ICML 2019).

These two explanations make different predictions and are separated by the
variants below. A blur-pool stem downsamples **by exactly the same factor**
as the standard stem but low-pass filters before subsampling. It therefore
holds output resolution constant while removing aliasing:

===========================  ==========  ==================  ==================
Configuration                conv1       downsampling        grid before stage2
===========================  ==========  ==================  ==================
S2P (standard, frozen)       stride 2    maxpool 3x2         16 x 16
S2B (this module)            stride 2    blurpool 3x2        16 x 16
S2N (frozen)                 stride 2    identity            32 x 32
S1P (frozen)                 stride 1    maxpool 3x2         32 x 32
S1N / HighRes (frozen)       stride 1    identity            64 x 64
S1B (this module)            stride 1    blurpool 3x2        32 x 32
===========================  ==========  ==================  ==================

Decision rule
-------------
* If S2B recovers most of the Standard -> HighRes Macro-F1 gap, the effect is
  dominated by **aliasing**, and the paper's mechanism claim must be restated.
* If S2B stays close to S2P, the effect is dominated by **resolution** and the
  original interpretation is supported.

BlurPool adds **no trainable parameters** (the filter is a fixed buffer), so
every configuration here keeps the frozen 1,262,397-parameter count. This is
what makes the comparison a fair control.

Frozen-file policy
------------------
This module is intentionally separate from ``stem_ablation_models.py``. That
file is pinned by SHA-256 in ``run_stem_ablation_training.py`` and its hash
guards the completed S2P/S2N/S1P/S1N runs. Editing it would invalidate that
audit trail, so new configurations live here instead.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import shufflenet_v2_x1_0

from wafermap.constants import NUM_CLASSES


def binomial_kernel(kernel_size: int) -> torch.Tensor:
    """Return normalized binomial coefficients of order ``kernel_size - 1``.

    ``kernel_size=3`` gives [1, 2, 1] / 4, the separable 3-tap filter used by
    Zhang (2019) and the shape-preserving replacement for the torchvision
    ShuffleNetV2 ``MaxPool2d(3, stride=2, padding=1)``.
    """
    if kernel_size < 2:
        raise ValueError("kernel_size must be >= 2")
    coefficients = torch.tensor(
        [float(_binomial(kernel_size - 1, index)) for index in range(kernel_size)],
        dtype=torch.float32,
    )
    return coefficients / coefficients.sum()


def _binomial(order: int, index: int) -> int:
    from math import comb

    return comb(order, index)


class BlurPool2d(nn.Module):
    """Fixed depthwise binomial filter followed by subsampling.

    This is the anti-aliasing operator of Zhang (2019). The filter is
    registered as a buffer rather than a parameter, so it contributes zero
    trainable weights and cannot be altered by the optimizer. With
    ``kernel_size=3, stride=2, padding=1`` it is output-shape identical to
    ``nn.MaxPool2d(3, stride=2, padding=1)``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        taps = binomial_kernel(kernel_size)
        two_dimensional = torch.outer(taps, taps)
        weight = (
            two_dimensional.view(1, 1, kernel_size, kernel_size)
            .expand(channels, 1, kernel_size, kernel_size)
            .contiguous()
        )
        self.register_buffer("weight", weight)
        self.weight.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class ShuffleNetV2StemAntiAliased(nn.Module):
    """ShuffleNetV2 whose input stem varies stride and downsampling filter.

    ``downsampling`` selects what replaces torchvision's initial max pooling:

    ``"maxpool"``   frozen baseline behaviour (S2P when stride is 2)
    ``"identity"``  no initial pooling (S2N when stride is 2)
    ``"blurpool"``  fixed binomial anti-aliasing filter (S2B when stride is 2)
    """

    SUPPORTED_DOWNSAMPLING = ("maxpool", "identity", "blurpool")

    def __init__(
        self,
        conv1_stride: int,
        downsampling: str,
        blur_kernel_size: int = 3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        if conv1_stride not in (1, 2):
            raise ValueError("conv1_stride must be either 1 or 2")
        if downsampling not in self.SUPPORTED_DOWNSAMPLING:
            raise ValueError(
                f"downsampling must be one of {self.SUPPORTED_DOWNSAMPLING}"
            )

        self.conv1_stride = conv1_stride
        self.downsampling = downsampling
        self.blur_kernel_size = blur_kernel_size

        self.network = shufflenet_v2_x1_0(weights=None)

        self.network.conv1[0] = nn.Conv2d(
            in_channels=1,
            out_channels=24,
            kernel_size=3,
            stride=conv1_stride,
            padding=1,
            bias=False,
        )

        stem_channels = self.network.conv1[0].out_channels

        if downsampling == "maxpool":
            # torchvision default: MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.network.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        elif downsampling == "identity":
            self.network.maxpool = nn.Identity()
        else:
            self.network.maxpool = BlurPool2d(
                channels=stem_channels,
                kernel_size=blur_kernel_size,
                stride=2,
                padding=blur_kernel_size // 2,
            )

        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class ShuffleNetV2Stride2BlurPool(ShuffleNetV2StemAntiAliased):
    """S2B: stride-two convolution with an anti-aliased, resolution-matched stem.

    Downsamples to the same 16 x 16 grid as the standard stride-two plus
    max-pooling stem, but low-pass filters before subsampling. This is the
    primary control that separates aliasing from resolution.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__(
            conv1_stride=2,
            downsampling="blurpool",
            num_classes=num_classes,
        )


class ShuffleNetV2Stride1BlurPool(ShuffleNetV2StemAntiAliased):
    """S1B: stride-one convolution with anti-aliased pooling.

    Reaches the same 32 x 32 grid as S2N and S1P but with a low-pass filter,
    isolating aliasing at an intermediate resolution.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__(
            conv1_stride=1,
            downsampling="blurpool",
            num_classes=num_classes,
        )


def _self_check() -> None:
    """Shape, parameter-count, and gradient checks. Run: python -m ..."""

    expected_parameters = 1_262_397
    cases = (
        (ShuffleNetV2Stride2BlurPool, (64, 64), (16, 16)),
        (ShuffleNetV2Stride1BlurPool, (64, 64), (32, 32)),
    )

    for factory, input_size, expected_grid in cases:
        model = factory()
        parameters = model.trainable_parameter_count()
        assert parameters == expected_parameters, (
            f"{factory.__name__}: {parameters} != {expected_parameters}"
        )

        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 1, *input_size))
        assert output.shape == (2, NUM_CLASSES), output.shape

        # Confirm the stem really produces the claimed pre-stage-2 grid.
        stem = model.network
        activations = torch.zeros(1, 1, *input_size)
        activations = stem.conv1(activations)
        activations = stem.maxpool(activations)
        assert tuple(activations.shape[-2:]) == expected_grid, (
            f"{factory.__name__}: stem grid {tuple(activations.shape[-2:])} "
            f"!= {expected_grid}"
        )

        # BlurPool must contribute no trainable parameters.
        blur = stem.maxpool
        assert isinstance(blur, BlurPool2d)
        assert all(not p.requires_grad for p in blur.parameters())

        print(
            f"{factory.__name__}: params={parameters} "
            f"stem_grid={expected_grid} OK"
        )

    print("stem_antialiasing_models self-check passed.")


if __name__ == "__main__":
    _self_check()
