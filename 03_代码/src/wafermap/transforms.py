from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_train_transform():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.RandomVerticalFlip(
                p=0.5
            ),
            transforms.RandomRotation(
                degrees=90,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            ),
        ]
    )