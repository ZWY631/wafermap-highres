#!/usr/bin/env python3
"""Inspect the original WM-811K pickle without modifying it."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.compat import pickle_compat


# 让 scripts 目录中的脚本可以导入本项目的 wafermap 包。
CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import WM811K_CLASS_NAMES  # noqa: E402
from wafermap.paths import EXPERIMENT_DIR, RAW_DATA_DIR  # noqa: E402


# The raw release directory can be overridden with WM811K_RAW_DIR so that an
# alternative distribution of the same records (for example the MIRLab mirror
# used for the input-resolution ablation) can be used without disturbing the
# canonical Kaggle directory and its recorded SHA-256.
DATASET_DIR = Path(
    os.environ.get("WM811K_RAW_DIR", RAW_DATA_DIR / "wm811k_kaggle_qingyi_v1")
)
DATA_FILE = DATASET_DIR / "LSWMD.pkl"
LOG_FILE = EXPERIMENT_DIR / "logs" / "raw_data_inspection.txt"


def build_logger() -> logging.Logger:
    """Create one logger that writes to both the terminal and a text file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("wm811k_raw_inspection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    return logger


def normalize_nested_value(value: Any) -> str | None:
    """Convert WM-811K's nested array labels into a plain string or None."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None

    text = str(value).strip()
    return text if text else None


def summarize_value(value: Any) -> str:
    """Describe one cell without printing a complete wafer map array."""
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    return f"{type(value).__name__}: {value!r}"


def load_legacy_pandas_pickle(path: Path) -> Any:
    """Load a pandas 0.x pickle with current pandas module locations."""
    module_aliases = {
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.range": "pandas.core.indexes.range",
    }

    class WM811KUnpickler(pickle_compat.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            current_module = module_aliases.get(module, module)
            return super().find_class(current_module, name)

    with path.open("rb") as file_handle:
        return WM811KUnpickler(file_handle, encoding="latin1").load()


def log_value_counts(
    logger: logging.Logger,
    title: str,
    values: pd.Series,
) -> None:
    """Print counts and percentages in a stable, readable format."""
    logger.info("\n%s", title)
    logger.info("%-16s %12s %12s", "value", "count", "percent")
    logger.info("%-16s %12s %12s", "-" * 16, "-" * 12, "-" * 12)

    counts = values.value_counts(dropna=False)
    total = len(values)
    for value, count in counts.items():
        display_value = "<unlabeled>" if pd.isna(value) else str(value)
        logger.info(
            "%-16s %12s %11.4f%%",
            display_value,
            f"{int(count):,}",
            100.0 * int(count) / total,
        )


def main() -> None:
    logger = build_logger()

    logger.info("WM-811K RAW DATA INSPECTION")
    logger.info("=" * 72)
    logger.info("Data file : %s", DATA_FILE)
    logger.info("Log file  : %s", LOG_FILE)

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"WM-811K data file not found: {DATA_FILE}")

    file_size = DATA_FILE.stat().st_size
    logger.info("File size : %s bytes (%.2f GiB)", f"{file_size:,}", file_size / 1024**3)
    logger.info("pandas    : %s", pd.__version__)
    logger.info("NumPy     : %s", np.__version__)

    logger.info("\n[1/5] Loading the original pickle. This can take several minutes...")
    start_time = time.perf_counter()
    data = load_legacy_pandas_pickle(DATA_FILE)
    load_seconds = time.perf_counter() - start_time
    logger.info("Load completed in %.2f seconds.", load_seconds)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(data).__name__}")

    logger.info("\n[2/5] Checking table structure")
    logger.info("Object type     : %s", type(data).__name__)
    logger.info("Rows             : %s", f"{len(data):,}")
    logger.info("Columns          : %s", data.shape[1])
    logger.info("Column names     : %s", list(data.columns))
    logger.info("Shallow memory   : %.2f MiB", data.memory_usage(deep=False).sum() / 1024**2)
    logger.info("Column dtypes:\n%s", data.dtypes.to_string())

    expected_columns = {
        "waferMap",
        "dieSize",
        "lotName",
        "waferIndex",
        "trianTestLabel",
        "failureType",
    }
    missing_columns = sorted(expected_columns.difference(data.columns))
    logger.info("Expected columns : %s", "PASS" if not missing_columns else "FAIL")
    if missing_columns:
        logger.info("Missing columns  : %s", missing_columns)

    logger.info("\nFirst-row cell summaries:")
    for column in data.columns:
        logger.info("  %-16s %s", column, summarize_value(data.iloc[0][column]))

    logger.info("\n[3/5] Checking failure labels")
    if "failureType" not in data.columns:
        raise KeyError("Required column 'failureType' is missing")

    normalized_labels = data["failureType"].map(normalize_nested_value)
    log_value_counts(logger, "Failure-type distribution (all rows)", normalized_labels)

    labeled_mask = normalized_labels.notna()
    known_class_mask = normalized_labels.isin(WM811K_CLASS_NAMES)
    unexpected_labels = sorted(
        set(normalized_labels[labeled_mask].unique()).difference(WM811K_CLASS_NAMES)
    )
    logger.info("\nLabeled rows     : %s", f"{int(labeled_mask.sum()):,}")
    logger.info("Unlabeled rows   : %s", f"{int((~labeled_mask).sum()):,}")
    logger.info("Known-class rows : %s", f"{int(known_class_mask.sum()):,}")
    logger.info("Unexpected labels: %s", unexpected_labels if unexpected_labels else "None")

    logger.info("\n[4/5] Checking lots, split labels, and wafer-map sizes")
    if "lotName" in data.columns:
        logger.info("Unique lots      : %s", f"{data['lotName'].nunique(dropna=True):,}")

    if "trianTestLabel" in data.columns:
        normalized_splits = data["trianTestLabel"].map(normalize_nested_value)
        log_value_counts(logger, "Original train/test marker distribution", normalized_splits)

    if "waferMap" in data.columns:
        wafer_shapes = data["waferMap"].map(
            lambda value: tuple(value.shape) if isinstance(value, np.ndarray) else None
        )
        logger.info("\nMost common wafer-map shapes:")
        for shape, count in wafer_shapes.value_counts(dropna=False).head(10).items():
            logger.info("  %-16s %12s", str(shape), f"{int(count):,}")

    logger.info("\n[5/5] Inspection conclusion")
    checks_passed = not missing_columns and not unexpected_labels
    logger.info("Schema and labels : %s", "PASS" if checks_passed else "REVIEW REQUIRED")
    logger.info("Raw data modified : NO")
    logger.info("Inspection log    : %s", LOG_FILE)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
