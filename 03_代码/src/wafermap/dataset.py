from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from wafermap.constants import IMAGE_SIZE
from wafermap.paths import PROCESSED_DATA_DIR

PROCESSED_DIR = (
    PROCESSED_DATA_DIR / "wm811k_labeled_64x64"
)

IMAGE_FILE = PROCESSED_DIR / "images_uint8.npy"
METADATA_FILE = PROCESSED_DIR / "metadata.csv"


class WaferMapDataset(Dataset):
    def __init__(
        self,
        split_name: str,
        transform=None,
    ):
        valid_splits = {"train", "val", "test"}

        if split_name not in valid_splits:
            raise ValueError(
                f"Unknown split name: {split_name}"
            )

        self.split_name = split_name
        self.transform = transform

        self.images = np.load(
            IMAGE_FILE,
            mmap_mode="r",
        )

        metadata = pd.read_csv(METADATA_FILE)

        if self.images.shape[0] != len(metadata):
            raise ValueError(
                "Image and metadata row counts do not match."
            )

        if self.images.shape[1:] != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                f"Unexpected image shape: {self.images.shape}"
            )

        split_mask = metadata["split"] == split_name

        self.row_positions = np.flatnonzero(
            split_mask.to_numpy()
        )

        self.metadata = metadata.loc[
            split_mask
        ].reset_index(drop=True)

        self.labels = self.metadata[
            "label_id"
        ].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.row_positions)

    def __getitem__(self, index):
        row_position = int(
            self.row_positions[index]
        )

        image = self.images[
            row_position
        ].copy()

        image_tensor = torch.from_numpy(
            image
        ).to(torch.float32)

        image_tensor = (
            image_tensor.unsqueeze(0) / 2.0
        )

        if self.transform is not None:
            image_tensor = self.transform(
                image_tensor
            )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return image_tensor, label