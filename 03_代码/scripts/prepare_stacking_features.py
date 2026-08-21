#!/usr/bin/env python3
"""Prepare Kang and Kang's 59 handcrafted features for the fixed split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import skimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import WM811K_CLASS_NAMES
from wafermap.stacking_features import (
    FEATURE_NAMES,
    NUM_HANDCRAFTED_FEATURES,
    extract_handcrafted_features,
    load_legacy_pandas_pickle,
)


RAW_FILE = PROJECT_ROOT / "02_数据/raw/wm811k_kaggle_qingyi_v1/LSWMD.pkl"
METADATA_FILE = PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/metadata.csv"
SPLIT_FILE = PROJECT_ROOT / "02_数据/splits/wm811k_labeled_lot_disjoint.csv"
OUTPUT_DIR = PROJECT_ROOT / "02_数据/processed/wm811k_stacking_features"
FEATURE_FILE = OUTPUT_DIR / "handcrafted_features_float32.npy"
MANIFEST_FILE = OUTPUT_DIR / "run_manifest.json"

EXPECTED_SAMPLES = 172_950
EXPECTED_SPLIT_SHA256 = "c429a4a73121e7907e33c9c88314993717cee7d80261c87afa9a1bdb18d5c473"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--benchmark", type=int, metavar="SAMPLES")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args(argv)


def load_and_validate_inputs():
    for path in (RAW_FILE, METADATA_FILE, SPLIT_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(SPLIT_FILE) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Fixed lot-disjoint split hash mismatch.")

    metadata = pd.read_csv(METADATA_FILE)
    if len(metadata) != EXPECTED_SAMPLES:
        raise ValueError(f"Unexpected metadata count: {len(metadata)}")
    if metadata["source_index"].duplicated().any():
        raise ValueError("Duplicate source_index in processed metadata.")
    if set(metadata["label"]) != set(WM811K_CLASS_NAMES):
        raise ValueError("Processed metadata class set mismatch.")

    raw = load_legacy_pandas_pickle(RAW_FILE).reset_index(drop=True)
    positions = metadata["source_index"].to_numpy(dtype=np.int64)
    selected = raw.iloc[positions]
    raw_shapes = selected["waferMap"].map(lambda item: np.asarray(item).shape)
    tiny_mask = raw_shapes.map(min) < 5
    tiny_source_indices = metadata.loc[
        tiny_mask.to_numpy(), "source_index"
    ].to_numpy(dtype=np.int64)
    return metadata, selected["waferMap"].to_numpy(), tiny_source_indices


def calculate_features(wafer_maps, workers: int, progress: bool) -> np.ndarray:
    if workers < 1:
        raise ValueError("workers must be positive")
    result = np.empty((len(wafer_maps), NUM_HANDCRAFTED_FEATURES), dtype=np.float32)
    if workers == 1:
        iterator = map(extract_handcrafted_features, wafer_maps)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(extract_handcrafted_features, wafer_maps, chunksize=32)
    try:
        for index, features in enumerate(iterator):
            result[index] = features
            if progress and (index + 1) % 5_000 == 0:
                print(f"FEATURE_PROGRESS {index + 1}/{len(wafer_maps)}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    if not np.isfinite(result).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return result


def ensure_outputs_available():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")


def main(argv=None):
    args = parse_args(argv)
    ensure_outputs_available()
    print("Loading and validating the 2.0 GiB raw WM-811K pickle...", flush=True)
    load_start = time.perf_counter()
    metadata, wafer_maps, tiny_source_indices = load_and_validate_inputs()
    load_seconds = time.perf_counter() - load_start

    if args.preflight:
        print("STACKING_FEATURE_PREFLIGHT_PASSED")
        print(f"samples={len(metadata)}")
        print(f"raw_load_seconds={load_seconds:.3f}")
        print(f"workers={args.workers}")
        print(f"tiny_maps_requiring_padding={len(tiny_source_indices)}")
        print(f"planned_output={OUTPUT_DIR}")
        return

    sample_count = args.benchmark if args.benchmark is not None else len(wafer_maps)
    if not 1 <= sample_count <= len(wafer_maps):
        raise ValueError("benchmark sample count is out of range")
    start = time.perf_counter()
    features = calculate_features(wafer_maps[:sample_count], args.workers, args.execute)
    feature_seconds = time.perf_counter() - start

    if args.benchmark is not None:
        projected = feature_seconds * len(wafer_maps) / sample_count
        print("STACKING_FEATURE_BENCHMARK_COMPLETE")
        print(f"samples={sample_count}")
        print(f"workers={args.workers}")
        print(f"feature_seconds={feature_seconds:.3f}")
        print(f"projected_full_seconds={projected:.3f}")
        print(f"features_shape={features.shape}")
        return

    staging = Path(tempfile.mkdtemp(prefix="stacking_features_", dir=OUTPUT_DIR.parent))
    try:
        np.save(staging / FEATURE_FILE.name, features)
        payload = {
            "schema_version": 1,
            "completed_at": datetime.now().astimezone().isoformat(),
            "method": "Kang and Kang 2021 handcrafted wafer features",
            "doi": "10.1016/j.compind.2021.103450",
            "samples": len(features),
            "feature_count": NUM_HANDCRAFTED_FEATURES,
            "feature_names": list(FEATURE_NAMES),
            "feature_source": "raw variable-size waferMap selected by source_index",
            "tiny_map_adaptation": {
                "rule": "center zero-pad each dimension to at least 5 for the author-excluded tiny maps",
                "count": len(tiny_source_indices),
                "source_indices": tiny_source_indices.tolist(),
            },
            "cnn_source": "separate fixed 64x64 nearest-neighbor maps",
            "workers": args.workers,
            "durations_seconds": {
                "raw_load_and_validation": load_seconds,
                "feature_extraction": feature_seconds,
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_image": skimage.__version__,
            },
            "input_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in (RAW_FILE, METADATA_FILE, SPLIT_FILE)
            },
            "script_sha256": sha256_file(Path(__file__)),
            "module_sha256": sha256_file(
                SRC_DIR / "wafermap" / "stacking_features.py"
            ),
            "artifact_sha256": {
                FEATURE_FILE.name: sha256_file(staging / FEATURE_FILE.name)
            },
        }
        (staging / MANIFEST_FILE.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if OUTPUT_DIR.exists():
            raise FileExistsError(OUTPUT_DIR)
        staging.rename(OUTPUT_DIR)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("STACKING_FEATURE_EXTRACTION_COMPLETE")
    print(f"output={OUTPUT_DIR}")
    print(f"feature_seconds={feature_seconds:.3f}")


if __name__ == "__main__":
    main()
