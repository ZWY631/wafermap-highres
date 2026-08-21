#!/usr/bin/env python3
"""Create reproducible lot-disjoint splits for labeled WM-811K data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inspect_raw_wm811k import (
    DATA_FILE,
    load_legacy_pandas_pickle,
    normalize_nested_value,
)
from wafermap.constants import RANDOM_SEED, WM811K_CLASS_NAMES
from wafermap.paths import SPLITS_DIR

OUTPUT_FILE = SPLITS_DIR / "wm811k_labeled_lot_disjoint.csv"

def load_labeled_data():
    print(f"正在读取原始数据：{DATA_FILE}")

    data = load_legacy_pandas_pickle(DATA_FILE)
    data = data.reset_index(names="source_index")

    data["label"] = data["failureType"].map(normalize_nested_value)
    data["lotName"] = data["lotName"].map(normalize_nested_value)
    data["waferIndex"] = data["waferIndex"].map(normalize_nested_value)

    labeled_data = data[
        data["label"].isin(WM811K_CLASS_NAMES)
    ].copy()

    required_columns = [
        "source_index",
        "lotName",
        "waferIndex",
        "label",
    ]

    if labeled_data[required_columns].isna().any().any():
        raise ValueError(
            "Required split metadata contains missing values."
        )

    print(f"有标签样本数量：{len(labeled_data):,}")
    print(
        f"参与划分的 lot 数量："
        f"{labeled_data['lotName'].nunique():,}"
    )

    return labeled_data

def assign_splits(labeled_data):
    splitter = StratifiedGroupKFold(
        n_splits=20,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    split_names = pd.Series(
        index=labeled_data.index,
        data="",
        dtype="string",
    )

    for fold_index, (_, holdout_positions) in enumerate(
        splitter.split(
            labeled_data,
            y=labeled_data["label"],
            groups=labeled_data["lotName"],
        )
    ):
        if fold_index in {0, 1, 2}:
            split_name = "val"
        elif fold_index in {3, 4, 5}:
            split_name = "test"
        else:
            split_name = "train"

        split_names.iloc[holdout_positions] = split_name

    if (split_names == "").any():
        raise RuntimeError(
            "Some samples were not assigned to a split."
        )

    result = labeled_data.copy()
    result["split"] = split_names.to_numpy()

    return result

def validate_and_save(manifest):
    if len(manifest) != 172_950:
        raise ValueError(
            f"Expected 172,950 rows, got {len(manifest):,}."
        )

    if manifest["source_index"].duplicated().any():
        raise ValueError("Duplicate source_index detected.")

    expected_splits = {"train", "val", "test"}
    actual_splits = set(manifest["split"].dropna().unique())

    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected split names: {sorted(actual_splits)}"
        )

    lot_split_counts = (
        manifest.groupby("lotName")["split"].nunique()
    )

    if lot_split_counts.max() != 1:
        raise ValueError(
            "At least one lot appears in multiple splits."
        )

    split_order = ["train", "val", "test"]

    split_counts = (
        manifest["split"]
        .value_counts()
        .reindex(split_order)
    )

    class_counts = pd.crosstab(
        manifest["split"],
        manifest["label"],
    ).reindex(
        index=split_order,
        columns=WM811K_CLASS_NAMES,
        fill_value=0,
    )

    if (class_counts == 0).any().any():
        raise ValueError(
            "At least one class is missing from a split."
        )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "source_index",
        "lotName",
        "waferIndex",
        "label",
        "split",
    ]

    manifest[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nSplit sizes:")
    print(split_counts.to_string())
    print("\nClass counts:")
    print(class_counts.to_string())
    print(f"\n划分清单已保存：{OUTPUT_FILE}")


def main():
    labeled_data = load_labeled_data()
    manifest = assign_splits(labeled_data)
    validate_and_save(manifest)


if __name__ == "__main__":
    main()