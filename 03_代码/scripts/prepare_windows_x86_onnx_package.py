#!/usr/bin/env python3
"""Build, validate, freeze, and archive the Windows x86-64 ONNX package."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ID = "20260801_windows_x86_onnx"
PACKAGE_DIR = PROJECT_ROOT / "08_归档" / "20260801_windows_x86_onnx_deployment"
ARCHIVE_PATH = PROJECT_ROOT / "08_归档" / f"{PACKAGE_ID}_deployment.zip"
SOURCE_IMAGES = (
    PROJECT_ROOT
    / "02_数据"
    / "processed"
    / "wm811k_labeled_64x64"
    / "images_uint8.npy"
)
SOURCE_METADATA = (
    PROJECT_ROOT
    / "02_数据"
    / "processed"
    / "wm811k_labeled_64x64"
    / "metadata.csv"
)
SOURCE_SPLIT = (
    PROJECT_ROOT / "02_数据" / "splits" / "wm811k_labeled_lot_disjoint.csv"
)
SOURCE_CHECKPOINT = (
    PROJECT_ROOT
    / "04_实验"
    / "checkpoints"
    / "shufflenet_v2_highres_ce_full"
    / "best.pt"
)
SOURCE_PROTOCOL = (
    PROJECT_ROOT / "00_项目管理" / "20260801_Windows_x86_ONNX部署冻结协议.md"
)

EXPECTED_SOURCE_HASHES = {
    SOURCE_IMAGES: "ec496dd8a5cdb2f83c54a048497d1f6d77e64ae74efd495c36456629c9cf9aa3",
    SOURCE_METADATA: "b6b1e0c2d819c60aa879e50015408161324616cbe5d928f4001d6609a7c9e681",
    SOURCE_SPLIT: "c429a4a73121e7907e33c9c88314993717cee7d80261c87afa9a1bdb18d5c473",
    SOURCE_CHECKPOINT: "cafebb6a76ef566ea05fa47e7745677c35cb3abbec87b1de468c0898b02c67bc",
}

CALIBRATION_SEED = 20260801
CALIBRATION_SAMPLES = 2048
CALIBRATION_PER_CLASS_FIRST = 64
EXPECTED_TEST_SAMPLES = 25943
EXPECTED_TOTAL_SAMPLES = 172950
EXPECTED_CLASS_IDS = set(range(9))

GENERATED_FILES = (
    PACKAGE_DIR / "checkpoint" / "best.pt",
    PACKAGE_DIR / "data" / "calibration_train_uint8.npz",
    PACKAGE_DIR / "data" / "fixed_test_uint8.npz",
    PACKAGE_DIR / "models" / "highres_shufflenetv2_fp32.onnx",
    PACKAGE_DIR / "models" / "fp32_export_validation.json",
    PACKAGE_DIR / "models" / "highres_shufflenetv2_int8_qdq.onnx",
    PACKAGE_DIR / "models" / "int8_quantization_validation.json",
    PACKAGE_DIR / "FROZEN_PROTOCOL.md",
    PACKAGE_DIR / "package_manifest.json",
    ARCHIVE_PATH,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_files() -> None:
    for path, expected_hash in EXPECTED_SOURCE_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen source file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen source hash mismatch: {path}\n"
                f"expected={expected_hash}\nactual={actual_hash}"
            )
    if not SOURCE_PROTOCOL.is_file():
        raise FileNotFoundError(f"Missing frozen protocol: {SOURCE_PROTOCOL}")


def refuse_overwrite() -> None:
    existing = [path for path in GENERATED_FILES if path.exists()]
    if existing:
        lines = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite generated package files:\n{lines}")


def read_metadata() -> dict[str, np.ndarray]:
    columns: dict[str, list] = {
        "source_index": [],
        "label_id": [],
        "split": [],
    }
    with SOURCE_METADATA.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = set(columns)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Metadata columns do not contain {sorted(required)}")
        for row in reader:
            columns["source_index"].append(int(row["source_index"]))
            columns["label_id"].append(int(row["label_id"]))
            columns["split"].append(row["split"])

    if len(columns["source_index"]) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError("Unexpected metadata sample count.")
    source_indices = np.asarray(columns["source_index"], dtype=np.int64)
    labels = np.asarray(columns["label_id"], dtype=np.int64)
    splits = np.asarray(columns["split"])
    if len(np.unique(source_indices)) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError("Metadata source_index values are not unique.")
    if not np.all(np.diff(source_indices) > 0):
        raise ValueError("Metadata source_index values are not strictly increasing.")
    if set(np.unique(labels).tolist()) != EXPECTED_CLASS_IDS:
        raise ValueError("Unexpected class IDs in metadata.")
    if set(np.unique(splits).tolist()) != {"train", "val", "test"}:
        raise ValueError("Unexpected split labels in metadata.")
    return {
        "source_index": source_indices,
        "label_id": labels,
        "split": splits,
    }


def choose_calibration_row_positions(metadata: dict[str, np.ndarray]) -> np.ndarray:
    train_positions = np.flatnonzero(metadata["split"] == "train")
    train_labels = metadata["label_id"][train_positions]
    rng = np.random.default_rng(CALIBRATION_SEED)
    selected_parts = []
    for class_id in sorted(EXPECTED_CLASS_IDS):
        candidates = train_positions[train_labels == class_id]
        take = min(CALIBRATION_PER_CLASS_FIRST, len(candidates))
        selected_parts.append(rng.choice(candidates, size=take, replace=False))
    first_pass = np.concatenate(selected_parts).astype(np.int64)
    remaining_pool = np.setdiff1d(train_positions, first_pass, assume_unique=False)
    remaining_count = CALIBRATION_SAMPLES - len(first_pass)
    if remaining_count < 0 or remaining_count > len(remaining_pool):
        raise ValueError("Invalid calibration sampling request.")
    second_pass = rng.choice(remaining_pool, size=remaining_count, replace=False)
    selected = np.concatenate([first_pass, second_pass]).astype(np.int64)
    if len(selected) != CALIBRATION_SAMPLES or len(np.unique(selected)) != len(selected):
        raise RuntimeError("Calibration sample selection is invalid.")
    rng.shuffle(selected)
    return selected


def create_data_packages(metadata: dict[str, np.ndarray]) -> None:
    source_images = np.load(SOURCE_IMAGES, mmap_mode="r")
    if source_images.shape != (EXPECTED_TOTAL_SAMPLES, 64, 64):
        raise ValueError(f"Unexpected source image shape: {source_images.shape}")
    if source_images.dtype != np.uint8:
        raise ValueError(f"Unexpected source image dtype: {source_images.dtype}")

    calibration_positions = choose_calibration_row_positions(metadata)
    test_positions = np.flatnonzero(metadata["split"] == "test").astype(np.int64)
    if len(test_positions) != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"Unexpected fixed test size: {len(test_positions)}")
    if np.intersect1d(calibration_positions, test_positions).size:
        raise RuntimeError("Calibration and fixed-test samples overlap.")

    calibration_images = np.asarray(source_images[calibration_positions], dtype=np.uint8)
    calibration_labels = metadata["label_id"][calibration_positions].astype(np.int64)
    test_images = np.asarray(source_images[test_positions], dtype=np.uint8)
    test_labels = metadata["label_id"][test_positions].astype(np.int64)

    np.savez_compressed(
        PACKAGE_DIR / "data" / "calibration_train_uint8.npz",
        images_uint8=calibration_images,
        labels=calibration_labels,
        row_positions=calibration_positions,
        source_indices=metadata["source_index"][calibration_positions],
        random_seed=np.asarray(CALIBRATION_SEED, dtype=np.int64),
    )
    np.savez_compressed(
        PACKAGE_DIR / "data" / "fixed_test_uint8.npz",
        images_uint8=test_images,
        labels=test_labels,
        row_positions=test_positions,
        source_indices=metadata["source_index"][test_positions],
    )


def run_checked(script_name: str) -> None:
    subprocess.run(
        [sys.executable, str(PACKAGE_DIR / script_name)],
        cwd=PACKAGE_DIR,
        check=True,
    )


def create_manifest() -> None:
    relative_paths = [
        "README_CN.md",
        "FROZEN_PROTOCOL.md",
        "benchmark_windows.py",
        "export_fp32.py",
        "model_definition.py",
        "quantize_int8.py",
        "requirements_windows.txt",
        "run_formal_benchmark.ps1",
        "setup_and_preflight.ps1",
        "checkpoint/best.pt",
        "data/calibration_train_uint8.npz",
        "data/fixed_test_uint8.npz",
        "models/highres_shufflenetv2_fp32.onnx",
        "models/fp32_export_validation.json",
        "models/highres_shufflenetv2_int8_qdq.onnx",
        "models/int8_quantization_validation.json",
    ]
    files = []
    for relative_path in relative_paths:
        path = PACKAGE_DIR / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing package file: {path}")
        files.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "package_id": PACKAGE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            str(path.relative_to(PROJECT_ROOT)): digest
            for path, digest in EXPECTED_SOURCE_HASHES.items()
        },
        "calibration": {
            "split": "train",
            "samples": CALIBRATION_SAMPLES,
            "random_seed": CALIBRATION_SEED,
            "per_class_first": CALIBRATION_PER_CLASS_FIRST,
        },
        "fixed_test_samples": EXPECTED_TEST_SAMPLES,
        "files": files,
    }
    (PACKAGE_DIR / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_archive() -> None:
    excluded_names = {".DS_Store", "__pycache__", ".venv", "results"}
    with zipfile.ZipFile(
        ARCHIVE_PATH, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            relative = path.relative_to(PACKAGE_DIR)
            if any(part in excluded_names for part in relative.parts) or not path.is_file():
                continue
            archive_name = Path(PACKAGE_DIR.name) / relative
            archive.write(path, archive_name.as_posix())


def main() -> None:
    print("[1/8] Verifying frozen source files...", flush=True)
    verify_source_files()
    refuse_overwrite()
    for directory in (PACKAGE_DIR / "checkpoint", PACKAGE_DIR / "data", PACKAGE_DIR / "models"):
        directory.mkdir(parents=True, exist_ok=True)

    print("[2/8] Copying the frozen checkpoint and protocol...", flush=True)
    shutil.copy2(SOURCE_CHECKPOINT, PACKAGE_DIR / "checkpoint" / "best.pt")
    shutil.copy2(SOURCE_PROTOCOL, PACKAGE_DIR / "FROZEN_PROTOCOL.md")

    print("[3/8] Reading and validating metadata...", flush=True)
    metadata = read_metadata()
    print("[4/8] Creating deterministic calibration and fixed-test packages...", flush=True)
    create_data_packages(metadata)

    print("[5/8] Exporting and validating FP32 ONNX...", flush=True)
    run_checked("export_fp32.py")
    print("[6/8] Quantizing and validating static INT8 ONNX...", flush=True)
    run_checked("quantize_int8.py")

    print("[7/8] Creating the package manifest...", flush=True)
    create_manifest()
    print("[8/8] Creating the Windows transfer ZIP...", flush=True)
    create_archive()
    print(f"Package directory: {PACKAGE_DIR}")
    print(f"Transfer archive: {ARCHIVE_PATH}")
    print(f"Archive SHA-256: {sha256_file(ARCHIVE_PATH)}")


if __name__ == "__main__":
    main()
