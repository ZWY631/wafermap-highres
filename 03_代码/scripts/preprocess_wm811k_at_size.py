#!/usr/bin/env python3
"""Convert labeled WM-811K wafer maps to fixed NxN arrays for any N.

Why this exists
---------------
The manuscript normalizes every map to 64 x 64 and never justifies that
choice. A reviewer will ask: if preserving spatial resolution is the
mechanism, why stop at 64? This script produces the alternative resolutions
needed to answer that, with the frozen pipeline otherwise unchanged.

Guarantees that keep the comparison fair
----------------------------------------
* The **lot-disjoint split manifest is reused byte-for-byte**
  (``02_数据/splits/wm811k_labeled_lot_disjoint.csv``). No map changes subset
  between resolutions, so a 128 x 128 run is paired with the 64 x 64 run at
  the level of individual wafers.
* Square padding, nearest-neighbour interpolation, and the {0, 1, 2} value
  domain are identical to ``preprocess_wm811k.py``. Only the target side
  length differs.
* Index integrity against the raw pickle is re-verified here, exactly as in
  the frozen script.

Usage
-----
    python scripts/preprocess_wm811k_at_size.py --image-size 128
    python scripts/preprocess_wm811k_at_size.py --image-size 96
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = CODE_DIR / "scripts"
SRC_DIR = CODE_DIR / "src"
PROJECT_ROOT = CODE_DIR.parent

for candidate in (SCRIPTS_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from inspect_raw_wm811k import (  # noqa: E402
    DATA_FILE,
    load_legacy_pandas_pickle,
    normalize_nested_value,
)
from wafermap.constants import WM811K_CLASS_NAMES  # noqa: E402
from wafermap.paths import PROCESSED_DATA_DIR, SPLITS_DIR  # noqa: E402


SPLIT_FILE = SPLITS_DIR / "wm811k_labeled_lot_disjoint.csv"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Preprocess labeled WM-811K maps at a chosen resolution."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=True,
        help="Target square side length in cells (64 reproduces the frozen set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the output directory. Useful for equivalence checks "
            "against the frozen 64 x 64 set without overwriting it."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args(argv)


def output_dir_for(image_size: int) -> Path:
    return PROCESSED_DATA_DIR / f"wm811k_labeled_{image_size}x{image_size}"


RAW_SHA256_FILE = DATA_FILE.parent / "SHA256SUMS.txt"


def verify_raw_data_available() -> None:
    """Fail early, with an actionable message, if the raw pickle is absent.

    The processed arrays under ``02_数据/processed`` are derived artifacts and
    cannot be resampled to a new target resolution: information removed by a
    64 x 64 nearest-neighbour resize is gone. A new resolution therefore
    requires ``LSWMD.pkl`` itself.
    """
    if DATA_FILE.is_file():
        return

    expected = "unknown"
    if RAW_SHA256_FILE.is_file():
        for line in RAW_SHA256_FILE.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == DATA_FILE.name:
                expected = parts[0]

    raise FileNotFoundError(
        "Raw WM-811K pickle is missing, so a new resolution cannot be built.\n"
        f"  expected path : {DATA_FILE}\n"
        f"  expected sha256: {expected}\n"
        "  expected size : 2,095,505,977 bytes\n\n"
        "The processed arrays in 02_数据/processed are derived artifacts; "
        "resampling them cannot recover detail that a 64 x 64 resize already "
        "discarded. Re-download LSWMD.pkl from\n"
        "  https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map "
        "(Version 1, CC0 1.0)\n"
        "place it at the expected path above, and verify it against "
        "SHA256SUMS.txt before re-running this script."
    )


def load_input_data():
    print(f"正在读取划分清单：{SPLIT_FILE}")
    manifest = pd.read_csv(SPLIT_FILE)

    required_columns = ["source_index", "lotName", "waferIndex", "label", "split"]
    missing_columns = set(required_columns).difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    print(f"划分清单样本数：{len(manifest):,}")
    print(f"正在读取原始数据：{DATA_FILE}")

    source_data = load_legacy_pandas_pickle(DATA_FILE).reset_index(drop=True)

    source_indices = manifest["source_index"].to_numpy(dtype=np.int64)
    if source_indices.min() < 0:
        raise ValueError("source_index contains a negative value.")
    if source_indices.max() >= len(source_data):
        raise ValueError("source_index exceeds the raw data length.")

    selected_data = source_data.iloc[source_indices].reset_index(drop=True)

    source_values = {
        "label": selected_data["failureType"].map(normalize_nested_value),
        "lotName": selected_data["lotName"].map(normalize_nested_value),
    }
    for column_name, source_series in source_values.items():
        if not np.array_equal(
            source_series.to_numpy(), manifest[column_name].to_numpy()
        ):
            raise ValueError(f"Manifest mismatch detected in column: {column_name}")

    print("划分清单与原始数据索引核对通过。")
    return manifest, selected_data


def square_pad_and_resize(wafer_map, image_size: int):
    """Square-pad to the larger original side, then nearest-neighbour resize."""
    map_array = np.asarray(wafer_map)

    if map_array.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got shape {map_array.shape}.")
    if not np.isin(map_array, (0, 1, 2)).all():
        raise ValueError("Wafer map contains values outside {0, 1, 2}.")

    map_array = map_array.astype(np.uint8, copy=False)

    height, width = map_array.shape
    square_size = max(height, width)
    square_canvas = np.zeros((square_size, square_size), dtype=np.uint8)

    target_top = (square_size - height) // 2
    target_left = (square_size - width) // 2
    square_canvas[
        target_top : target_top + height,
        target_left : target_left + width,
    ] = map_array

    resized_map = cv2.resize(
        square_canvas,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )

    if not np.isin(resized_map, (0, 1, 2)).all():
        raise ValueError("Resize created an unexpected pixel value.")

    return resized_map.astype(np.uint8, copy=False)


def build_image_array(selected_data, image_size: int):
    image_array = np.empty(
        (len(selected_data), image_size, image_size), dtype=np.uint8
    )
    wafer_maps = selected_data["waferMap"].to_numpy()

    for row_position, wafer_map in enumerate(wafer_maps):
        image_array[row_position] = square_pad_and_resize(wafer_map, image_size)
        if (row_position + 1) % 20_000 == 0:
            print(f"已处理晶圆图：{row_position + 1:,}/{len(wafer_maps):,}")

    return image_array


def save_outputs(manifest, image_array, output_dir: Path, image_size: int):
    expected_shape = (len(manifest), image_size, image_size)
    if image_array.shape != expected_shape:
        raise ValueError(f"Unexpected image array shape: {image_array.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_file = output_dir / "images_uint8.npy"
    metadata_file = output_dir / "metadata.csv"

    np.save(image_file, image_array)

    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(WM811K_CLASS_NAMES)
    }
    metadata = manifest[
        ["source_index", "lotName", "waferIndex", "label", "split"]
    ].copy()
    metadata["label_id"] = metadata["label"].map(class_to_id).astype(np.int64)
    metadata.to_csv(metadata_file, index=False)

    print(f"图像数组已保存：{image_file}")
    print(f"元数据已保存：{metadata_file}")

    # Record the split provenance so downstream runs can prove the manifest
    # was shared with the frozen 64 x 64 pipeline.
    provenance = {
        "image_size": image_size,
        "rows": int(len(manifest)),
        "split_file": str(SPLIT_FILE.relative_to(PROJECT_ROOT)),
        "interpolation": "cv2.INTER_NEAREST",
        "square_padding": "centered to max(original height, width)",
        "split_counts": {
            str(key): int(value)
            for key, value in manifest["split"].value_counts().items()
        },
    }
    (output_dir / "preprocessing_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"预处理溯源信息已保存：{output_dir / 'preprocessing_provenance.json'}")


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")

    output_dir = args.output_dir or output_dir_for(args.image_size)
    output_dir = output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Pass --overwrite to replace it."
        )

    verify_raw_data_available()
    manifest, selected_data = load_input_data()
    image_array = build_image_array(selected_data, args.image_size)
    save_outputs(manifest, image_array, output_dir, args.image_size)


if __name__ == "__main__":
    main()
