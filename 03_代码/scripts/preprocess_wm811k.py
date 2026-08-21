#!/usr/bin/env python3
"""Convert labeled WM-811K wafer maps to fixed 64x64 arrays."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inspect_raw_wm811k import (
    DATA_FILE,
    load_legacy_pandas_pickle,
    normalize_nested_value,
)
from wafermap.constants import IMAGE_SIZE, WM811K_CLASS_NAMES
from wafermap.paths import PROCESSED_DATA_DIR, SPLITS_DIR

SPLIT_FILE = (
    SPLITS_DIR / "wm811k_labeled_lot_disjoint.csv"
)

OUTPUT_DIR = (
    PROCESSED_DATA_DIR / "wm811k_labeled_64x64"
)

IMAGE_FILE = OUTPUT_DIR / "images_uint8.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

def load_input_data():
    print(f"正在读取划分清单：{SPLIT_FILE}")

    manifest = pd.read_csv(SPLIT_FILE)

    required_columns = [
        "source_index",
        "lotName",
        "waferIndex",
        "label",
        "split",
    ]

    missing_columns = set(required_columns).difference(
        manifest.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    print(f"划分清单样本数：{len(manifest):,}")
    print(f"正在读取原始数据：{DATA_FILE}")

    source_data = load_legacy_pandas_pickle(DATA_FILE)
    source_data = source_data.reset_index(drop=True)

    source_indices = manifest[
        "source_index"
    ].to_numpy(dtype=np.int64)

    if source_indices.min() < 0:
        raise ValueError("source_index contains a negative value.")

    if source_indices.max() >= len(source_data):
        raise ValueError(
            "source_index exceeds the raw data length."
        )

    selected_data = source_data.iloc[
        source_indices
    ].reset_index(drop=True)

    source_values = {
        "label": selected_data["failureType"].map(
            normalize_nested_value
        ),
        "lotName": selected_data["lotName"].map(
            normalize_nested_value
        ),
    }

    for column_name, source_series in source_values.items():
        manifest_values = manifest[column_name].to_numpy()
        raw_values = source_series.to_numpy()

        if not np.array_equal(raw_values, manifest_values):
            raise ValueError(
                f"Manifest mismatch detected in column: {column_name}"
            )

    print("划分清单与原始数据索引核对通过。")

    return manifest, selected_data
def square_pad_and_resize(wafer_map):
    map_array = np.asarray(wafer_map)

    if map_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D wafer map, got shape {map_array.shape}."
        )

    if not np.isin(map_array, (0, 1, 2)).all():
        raise ValueError(
            "Wafer map contains values outside {0, 1, 2}."
        )

    map_array = map_array.astype(
        np.uint8,
        copy=False,
    )

    height, width = map_array.shape
    square_size = max(height, width)

    square_canvas = np.zeros(
        (square_size, square_size),
        dtype=np.uint8,
    )

    target_top = (square_size - height) // 2
    target_left = (square_size - width) // 2

    square_canvas[
        target_top:target_top + height,
        target_left:target_left + width,
    ] = map_array

    resized_map = cv2.resize(
        square_canvas,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    if not np.isin(resized_map, (0, 1, 2)).all():
        raise ValueError(
            "Resize created an unexpected pixel value."
        )

    return resized_map.astype(
        np.uint8,
        copy=False,
    )

def build_image_array(selected_data):
    image_array = np.empty(
        (
            len(selected_data),
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=np.uint8,
    )

    wafer_maps = selected_data["waferMap"].to_numpy()

    for row_position, wafer_map in enumerate(wafer_maps):
        image_array[row_position] = square_pad_and_resize(
            wafer_map
        )

        if (row_position + 1) % 10_000 == 0:
            print(
                f"已处理晶圆图："
                f"{row_position + 1:,}/{len(wafer_maps):,}"
            )

    return image_array


def save_outputs(manifest, image_array):
    expected_shape = (
        len(manifest),
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if image_array.shape != expected_shape:
        raise ValueError(
            f"Unexpected image array shape: {image_array.shape}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(IMAGE_FILE, image_array)

    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(
            WM811K_CLASS_NAMES
        )
    }

    metadata = manifest[
        [
            "source_index",
            "lotName",
            "waferIndex",
            "label",
            "split",
        ]
    ].copy()

    metadata["label_id"] = metadata["label"].map(
        class_to_id
    ).astype(np.int64)

    metadata.to_csv(
        METADATA_FILE,
        index=False,
    )

    print(f"图像数组已保存：{IMAGE_FILE}")
    print(f"元数据已保存：{METADATA_FILE}")


def main():
    manifest, selected_data = load_input_data()
    image_array = build_image_array(selected_data)
    save_outputs(manifest, image_array)


if __name__ == "__main__":
    main()
