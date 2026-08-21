#!/usr/bin/env python3
"""Train one seed of the lot-grouped Kang and Kang stacking adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sklearn
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.stacking_reproduction import (
    CNN_BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    META_RIDGE_ALPHA,
    MFE_BATCH_SIZE,
    NUM_OOF_FOLDS,
    SUPPORTED_SEEDS,
    WEIGHT_DECAY,
    FeatureDataset,
    HandcraftedFNN,
    VGG16WaferCNN,
    WaferBinaryDataset,
    build_cnn_train_transform,
    classification_metrics,
    fit_feature_scaler,
    fit_meta_ridge,
    make_lot_grouped_oof_folds,
    predict_probabilities,
    seed_everything,
    select_device,
    train_classifier,
)


METHOD_ID = "kang_kang_2021_stacking_lot_adaptation"
IMAGES_FILE = PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/images_uint8.npy"
METADATA_FILE = PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/metadata.csv"
FEATURE_DIR = PROJECT_ROOT / "02_数据/processed/wm811k_stacking_features"
FEATURE_FILE = FEATURE_DIR / "handcrafted_features_float32.npy"
FEATURE_MANIFEST = FEATURE_DIR / "run_manifest.json"
SPLIT_FILE = PROJECT_ROOT / "02_数据/splits/wm811k_labeled_lot_disjoint.csv"
PROTOCOL_FILE = PROJECT_ROOT / "00_项目管理/20260807_领域专用方法同协议复现冻结协议.md"
CHECKPOINT_ROOT = PROJECT_ROOT / "04_实验/checkpoints/kang_kang_stacking_lot_reproduction"
STOPPABLE_STAGES = (
    "fold0_mfe",
    "fold0_cnn",
    "fold1_mfe",
    "fold1_cnn",
    "final_mfe",
    "final_cnn",
)

EXPECTED_SPLIT_SHA256 = "c429a4a73121e7907e33c9c88314993717cee7d80261c87afa9a1bdb18d5c473"
EXPECTED_COUNTS = {"train": 121_069, "val": 25_938, "test": 25_943}
SOURCE_FILES = (
    Path("03_代码/src/wafermap/stacking_features.py"),
    Path("03_代码/src/wafermap/stacking_reproduction.py"),
    Path("03_代码/scripts/train_kang_stacking_seed.py"),
    Path("00_项目管理/20260807_领域专用方法同协议复现冻结协议.md"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SUPPORTED_SEEDS, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--benchmark", type=int, metavar="TRAIN_SAMPLES")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=STOPPABLE_STAGES,
        help="With --execute, stop cleanly after this completed and hashed stage.",
    )
    args = parser.parse_args(argv)
    if args.stop_after is not None and not args.execute:
        parser.error("--stop-after requires --execute")
    return args


def expected_source_hashes():
    return {
        str(path): sha256_file(PROJECT_ROOT / path)
        for path in SOURCE_FILES
    }


def load_and_validate_inputs():
    for path in (
        IMAGES_FILE,
        METADATA_FILE,
        FEATURE_FILE,
        FEATURE_MANIFEST,
        SPLIT_FILE,
        PROTOCOL_FILE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(SPLIT_FILE) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Fixed lot split hash mismatch.")

    feature_manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    expected_feature_hash = feature_manifest["artifact_sha256"][FEATURE_FILE.name]
    if sha256_file(FEATURE_FILE) != expected_feature_hash:
        raise ValueError("Handcrafted feature file hash mismatch.")
    if feature_manifest["samples"] != sum(EXPECTED_COUNTS.values()):
        raise ValueError("Handcrafted feature sample count mismatch.")

    metadata = pd.read_csv(METADATA_FILE)
    counts = metadata["split"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Split counts mismatch: {counts}")
    if metadata["source_index"].duplicated().any():
        raise ValueError("Duplicate source_index in metadata.")
    if set(metadata["label"]) != set(WM811K_CLASS_NAMES):
        raise ValueError("Class set mismatch.")
    lot_counts = metadata.groupby("lotName")["split"].nunique()
    if int(lot_counts.max()) != 1:
        raise ValueError("A lot appears in more than one outer split.")

    images = np.load(IMAGES_FILE, mmap_mode="r")
    features = np.load(FEATURE_FILE, mmap_mode="r")
    if images.shape != (len(metadata), 64, 64) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected image array: {images.shape} {images.dtype}")
    if features.shape != (len(metadata), 59) or features.dtype != np.float32:
        raise ValueError(f"Unexpected feature array: {features.shape} {features.dtype}")
    return metadata, images, features, feature_manifest


def run_directory(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"seed{seed}"


def create_or_validate_run_protocol(seed: int, run_dir: Path, feature_manifest):
    protocol_path = run_dir / "run_protocol.json"
    payload = {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "seed": seed,
        "oof_folds": NUM_OOF_FOLDS,
        "selection_metric": "validation Macro-F1",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "mfe_batch_size": MFE_BATCH_SIZE,
        "cnn_batch_size": CNN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "meta_ridge_alpha": META_RIDGE_ALPHA,
        "cnn_initialization": "from scratch; torchvision vgg16 weights=None",
        "cnn_input": "failed-die mask only, centered to {-0.5, 0.5}",
        "mfe_input": "59 author-defined features from raw variable-size wafer maps",
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "feature_sha256": feature_manifest["artifact_sha256"][FEATURE_FILE.name],
        "source_sha256": expected_source_hashes(),
        "test_access": "forbidden in this training script",
    }
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "Run protocol differs from current frozen code/configuration; "
                f"refusing to resume {run_dir}"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def build_loader(dataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        generator=generator if shuffle else None,
    )


def write_history(rows, path: Path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalize_stage(staging: Path, stage_dir: Path, stage_payload: dict):
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(staging.iterdir())
        if path.is_file() and path.name != "stage_manifest.json"
    }
    stage_payload["artifact_sha256"] = artifact_hashes
    (staging / "stage_manifest.json").write_text(
        json.dumps(stage_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if stage_dir.exists():
        raise FileExistsError(stage_dir)
    staging.rename(stage_dir)


def validate_stage(stage_dir: Path):
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete stage: {stage_dir}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in payload["artifact_sha256"].items():
        path = stage_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Stage artifact mismatch: {path}")
    return payload


def stage_arrays(stage_dir: Path):
    validate_stage(stage_dir)
    return (
        np.load(stage_dir / "prediction_positions.npy"),
        np.load(stage_dir / "probabilities.npy"),
    )


def train_mfe_stage(
    stage_dir,
    features,
    labels,
    train_positions,
    prediction_positions,
    validation_positions,
    seed,
    device,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
):
    if stage_dir.exists():
        validate_stage(stage_dir)
        print(f"STAGE_ALREADY_COMPLETE {stage_dir.name}", flush=True)
        return
    staging = Path(tempfile.mkdtemp(prefix=f".{stage_dir.name}_", dir=stage_dir.parent))
    start = time.perf_counter()
    try:
        scaler = fit_feature_scaler(features[train_positions])
        train_x = scaler.transform(features[train_positions]).astype(np.float32)
        validation_x = scaler.transform(features[validation_positions]).astype(np.float32)
        prediction_x = scaler.transform(features[prediction_positions]).astype(np.float32)
        train_loader = build_loader(
            FeatureDataset(train_x, labels[train_positions]),
            MFE_BATCH_SIZE,
            True,
            seed,
        )
        validation_loader = build_loader(
            FeatureDataset(validation_x, labels[validation_positions]),
            EVAL_BATCH_SIZE,
            False,
            seed,
        )
        prediction_loader = build_loader(
            FeatureDataset(prediction_x, labels[prediction_positions]),
            EVAL_BATCH_SIZE,
            False,
            seed,
        )
        seed_everything(seed)
        model = HandcraftedFNN()
        result = train_classifier(
            model,
            train_loader,
            validation_loader,
            device,
            max_epochs=max_epochs,
            patience=patience,
        )
        model.load_state_dict(result.best_state_dict)
        model = model.to(device)
        probabilities, observed_labels = predict_probabilities(
            model, prediction_loader, device
        )
        if not np.array_equal(observed_labels, labels[prediction_positions]):
            raise RuntimeError("MFE prediction order mismatch.")
        torch.save(
            {
                "model": "HandcraftedFNN",
                "state_dict": result.best_state_dict,
                "best_epoch": result.best_epoch,
                "validation_metrics": result.best_validation_metrics,
            },
            staging / "best.pt",
        )
        np.savez(
            staging / "scaler.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            var=scaler.var_,
        )
        np.save(staging / "prediction_positions.npy", prediction_positions)
        np.save(staging / "probabilities.npy", probabilities.astype(np.float32))
        write_history(result.history, staging / "history.csv")
        finalize_stage(
            staging,
            stage_dir,
            {
                "stage": stage_dir.name,
                "branch": "mfe",
                "seed": seed,
                "train_samples": len(train_positions),
                "prediction_samples": len(prediction_positions),
                "validation_samples": len(validation_positions),
                "best_epoch": result.best_epoch,
                "validation_metrics": result.best_validation_metrics,
                "seconds": time.perf_counter() - start,
            },
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def train_cnn_stage(
    stage_dir,
    images,
    labels,
    train_positions,
    prediction_positions,
    validation_positions,
    seed,
    device,
    max_epochs=MAX_EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
):
    if stage_dir.exists():
        validate_stage(stage_dir)
        print(f"STAGE_ALREADY_COMPLETE {stage_dir.name}", flush=True)
        return
    staging = Path(tempfile.mkdtemp(prefix=f".{stage_dir.name}_", dir=stage_dir.parent))
    start = time.perf_counter()
    try:
        train_loader = build_loader(
            WaferBinaryDataset(
                images,
                train_positions,
                labels[train_positions],
                transform=build_cnn_train_transform(),
            ),
            CNN_BATCH_SIZE,
            True,
            seed,
        )
        validation_loader = build_loader(
            WaferBinaryDataset(
                images, validation_positions, labels[validation_positions]
            ),
            EVAL_BATCH_SIZE,
            False,
            seed,
        )
        prediction_loader = build_loader(
            WaferBinaryDataset(
                images, prediction_positions, labels[prediction_positions]
            ),
            EVAL_BATCH_SIZE,
            False,
            seed,
        )
        seed_everything(seed)
        model = VGG16WaferCNN()
        result = train_classifier(
            model,
            train_loader,
            validation_loader,
            device,
            max_epochs=max_epochs,
            patience=patience,
        )
        model.load_state_dict(result.best_state_dict)
        model = model.to(device)
        probabilities, observed_labels = predict_probabilities(
            model, prediction_loader, device
        )
        if not np.array_equal(observed_labels, labels[prediction_positions]):
            raise RuntimeError("CNN prediction order mismatch.")
        torch.save(
            {
                "model": "VGG16WaferCNN",
                "initialization": "weights=None",
                "state_dict": result.best_state_dict,
                "best_epoch": result.best_epoch,
                "validation_metrics": result.best_validation_metrics,
            },
            staging / "best.pt",
        )
        np.save(staging / "prediction_positions.npy", prediction_positions)
        np.save(staging / "probabilities.npy", probabilities.astype(np.float32))
        write_history(result.history, staging / "history.csv")
        finalize_stage(
            staging,
            stage_dir,
            {
                "stage": stage_dir.name,
                "branch": "cnn",
                "seed": seed,
                "train_samples": len(train_positions),
                "prediction_samples": len(prediction_positions),
                "validation_samples": len(validation_positions),
                "best_epoch": result.best_epoch,
                "validation_metrics": result.best_validation_metrics,
                "seconds": time.perf_counter() - start,
            },
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def fit_and_save_meta(run_dir, train_positions, validation_positions, labels, folds):
    meta_dir = run_dir / "meta"
    if meta_dir.exists():
        validate_stage(meta_dir)
        print("STAGE_ALREADY_COMPLETE meta", flush=True)
        return
    staging = Path(tempfile.mkdtemp(prefix=".meta_", dir=run_dir))
    start = time.perf_counter()
    try:
        train_lookup = {int(position): index for index, position in enumerate(train_positions)}
        oof_mfe = np.full((len(train_positions), NUM_CLASSES), np.nan, dtype=np.float32)
        oof_cnn = np.full_like(oof_mfe, np.nan)
        coverage = np.zeros(len(train_positions), dtype=np.int64)
        for fold_id, (_, relative_holdout) in enumerate(folds):
            expected_positions = train_positions[relative_holdout]
            for branch, target in (("mfe", oof_mfe), ("cnn", oof_cnn)):
                positions, probabilities = stage_arrays(
                    run_dir / "stages" / f"fold{fold_id}_{branch}"
                )
                if not np.array_equal(positions, expected_positions):
                    raise RuntimeError(f"Fold {fold_id} {branch} positions mismatch.")
                target_indices = np.asarray(
                    [train_lookup[int(position)] for position in positions], dtype=np.int64
                )
                target[target_indices] = probabilities
            coverage[relative_holdout] += 1
        if not np.all(coverage == 1):
            raise RuntimeError("OOF coverage is not exactly one.")
        if not np.isfinite(oof_mfe).all() or not np.isfinite(oof_cnn).all():
            raise RuntimeError("OOF probabilities are incomplete.")

        meta_x = np.concatenate((oof_mfe, oof_cnn), axis=1)
        meta_model = fit_meta_ridge(meta_x, labels[train_positions])
        mfe_positions, val_mfe = stage_arrays(run_dir / "stages/final_mfe")
        cnn_positions, val_cnn = stage_arrays(run_dir / "stages/final_cnn")
        if not np.array_equal(mfe_positions, validation_positions) or not np.array_equal(
            cnn_positions, validation_positions
        ):
            raise RuntimeError("Final validation prediction positions mismatch.")
        val_x = np.concatenate((val_mfe, val_cnn), axis=1)
        val_predictions = meta_model.predict(val_x)
        validation_metrics = classification_metrics(
            labels[validation_positions], val_predictions
        )

        np.savez(
            staging / "meta_ridge.npz",
            coefficient=meta_model.coefficient,
            intercept=meta_model.intercept,
            alpha=np.asarray(meta_model.alpha),
        )
        np.savez_compressed(
            staging / "oof_probabilities.npz",
            positions=train_positions,
            labels=labels[train_positions],
            mfe=oof_mfe,
            cnn=oof_cnn,
        )
        frame = pd.DataFrame(
            {
                "image_row": validation_positions,
                "true_label_id": labels[validation_positions],
                "predicted_label_id": val_predictions,
            }
        )
        frame["true_label"] = [WM811K_CLASS_NAMES[index] for index in frame.true_label_id]
        frame["predicted_label"] = [
            WM811K_CLASS_NAMES[index] for index in frame.predicted_label_id
        ]
        frame["correct"] = frame.true_label_id == frame.predicted_label_id
        frame.to_csv(staging / "validation_predictions.csv", index=False)
        (staging / "validation_summary.json").write_text(
            json.dumps(
                {
                    "samples": len(validation_positions),
                    "metrics": validation_metrics,
                    "test_accessed": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        finalize_stage(
            staging,
            meta_dir,
            {
                "stage": "meta",
                "branch": "multi_response_ridge",
                "train_oof_samples": len(train_positions),
                "validation_samples": len(validation_positions),
                "validation_metrics": validation_metrics,
                "seconds": time.perf_counter() - start,
                "test_accessed": False,
            },
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def complete_run_manifest(run_dir, seed, protocol_payload):
    path = run_dir / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"Run is already complete: {path}")
    stage_directories = sorted(
        list((run_dir / "stages").iterdir()) + [run_dir / "meta"]
    )
    stage_hashes = {}
    for stage_dir in stage_directories:
        validate_stage(stage_dir)
        manifest = stage_dir / "stage_manifest.json"
        stage_hashes[str(manifest.relative_to(run_dir))] = sha256_file(manifest)
    payload = {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "status": "training_complete_test_not_accessed",
        "seed": seed,
        "protocol": protocol_payload,
        "stage_manifest_sha256": stage_hashes,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(select_device()),
        },
        "peak_process_rss_bytes": psutil.Process().memory_info().rss,
        "test_accessed": False,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def preflight(seed, metadata, images, features, feature_manifest):
    train = metadata["split"].eq("train").to_numpy()
    train_positions = np.flatnonzero(train)
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    groups = metadata.loc[train, "lotName"].to_numpy()
    folds = make_lot_grouped_oof_folds(labels[train_positions], groups, seed)
    run_dir = run_directory(seed)
    if (run_dir / "run_manifest.json").exists():
        state = "complete"
    elif run_dir.exists():
        state = "partial_resumable_only_if_protocol_matches"
    else:
        state = "not_started"
    print("KANG_STACKING_PREFLIGHT_PASSED")
    print(f"seed={seed}")
    print(f"device={select_device()}")
    print(f"train_samples={len(train_positions)}")
    print(f"validation_samples={int(metadata['split'].eq('val').sum())}")
    print(f"oof_folds={len(folds)}")
    print(f"feature_sha256={feature_manifest['artifact_sha256'][FEATURE_FILE.name]}")
    print(f"run_state={state}")
    print(f"planned_output={run_dir}")
    print("test_inference_planned=NO")


def smoke(seed, metadata, images, features):
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    train_positions = []
    validation_positions = []
    for class_id in range(NUM_CLASSES):
        class_train = np.flatnonzero(
            metadata["split"].eq("train").to_numpy() & (labels == class_id)
        )
        class_val = np.flatnonzero(
            metadata["split"].eq("val").to_numpy() & (labels == class_id)
        )
        train_positions.extend(class_train[:4])
        validation_positions.extend(class_val[:2])
    train_positions = np.asarray(train_positions, dtype=np.int64)
    validation_positions = np.asarray(validation_positions, dtype=np.int64)
    device = select_device()
    with tempfile.TemporaryDirectory(prefix="kang_stacking_smoke_") as directory:
        root = Path(directory)
        train_mfe_stage(
            root / "mfe",
            features,
            labels,
            train_positions,
            validation_positions,
            validation_positions,
            seed,
            device,
            max_epochs=1,
            patience=1,
        )
        train_cnn_stage(
            root / "cnn",
            images,
            labels,
            train_positions,
            validation_positions,
            validation_positions,
            seed,
            device,
            max_epochs=1,
            patience=1,
        )
        mfe_positions, mfe_probabilities = stage_arrays(root / "mfe")
        cnn_positions, cnn_probabilities = stage_arrays(root / "cnn")
        if not np.array_equal(mfe_positions, cnn_positions):
            raise RuntimeError("Smoke prediction positions differ.")
        duplicated = np.concatenate((mfe_probabilities, cnn_probabilities), axis=1)
        meta = fit_meta_ridge(duplicated, labels[mfe_positions])
        predictions = meta.predict(duplicated)
        metrics = classification_metrics(labels[mfe_positions], predictions)
    print("KANG_STACKING_SMOKE_PASSED")
    print(f"device={device}")
    print(f"train_samples={len(train_positions)}")
    print(f"validation_samples={len(validation_positions)}")
    print(f"metrics_pipeline_check_only={metrics}")
    print("formal_outputs_created=NO")


def benchmark(seed, sample_count, metadata, images, features):
    if sample_count < 256:
        raise ValueError("Benchmark requires at least 256 training samples.")
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    all_train = np.flatnonzero(metadata["split"].eq("train").to_numpy())
    all_validation = np.flatnonzero(metadata["split"].eq("val").to_numpy())
    if sample_count > len(all_train):
        raise ValueError("Benchmark sample count exceeds the training split.")
    rng = np.random.default_rng(seed)
    train_positions = np.sort(rng.choice(all_train, size=sample_count, replace=False))
    validation_count = min(2048, len(all_validation))
    validation_positions = np.sort(
        rng.choice(all_validation, size=validation_count, replace=False)
    )
    device = select_device()
    with tempfile.TemporaryDirectory(prefix="kang_stacking_benchmark_") as directory:
        root = Path(directory)
        train_mfe_stage(
            root / "mfe",
            features,
            labels,
            train_positions,
            validation_positions,
            validation_positions,
            seed,
            device,
            max_epochs=1,
            patience=1,
        )
        train_cnn_stage(
            root / "cnn",
            images,
            labels,
            train_positions,
            validation_positions,
            validation_positions,
            seed,
            device,
            max_epochs=1,
            patience=1,
        )
        mfe = validate_stage(root / "mfe")
        cnn = validate_stage(root / "cnn")
    fold_train_samples = round(len(all_train) * (NUM_OOF_FOLDS - 1) / NUM_OOF_FOLDS)
    print("KANG_STACKING_BENCHMARK_COMPLETE")
    print(f"device={device}")
    print(f"benchmark_train_samples={sample_count}")
    print(f"benchmark_validation_samples={validation_count}")
    print(f"mfe_stage_seconds={mfe['seconds']:.3f}")
    print(f"cnn_stage_seconds={cnn['seconds']:.3f}")
    print(f"projected_mfe_fold_epoch_seconds={mfe['seconds'] * fold_train_samples / sample_count:.3f}")
    print(f"projected_cnn_fold_epoch_seconds={cnn['seconds'] * fold_train_samples / sample_count:.3f}")
    print(f"projected_cnn_final_epoch_seconds={cnn['seconds'] * len(all_train) / sample_count:.3f}")
    print(f"peak_process_rss_bytes={psutil.Process().memory_info().rss}")
    print("projection_limit=includes validation and prediction overhead; formal early-stop epoch count unknown")
    print("formal_outputs_created=NO")


def pause_after_stage(stage_name, stop_after):
    if stage_name != stop_after:
        return False
    print(f"FORMAL_STACKING_PAUSED_AFTER stage={stage_name}", flush=True)
    print("resume=rerun_same_seed_with_a_later_stop_after_or_without_stop_after")
    print("test_accessed=NO")
    return True


def execute(seed, metadata, images, features, feature_manifest, stop_after=None):
    run_dir = run_directory(seed)
    if (run_dir / "run_manifest.json").exists():
        raise FileExistsError(f"Completed run already exists: {run_dir}")
    protocol_payload = create_or_validate_run_protocol(seed, run_dir, feature_manifest)
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    train_positions = np.flatnonzero(metadata["split"].eq("train").to_numpy())
    validation_positions = np.flatnonzero(metadata["split"].eq("val").to_numpy())
    groups = metadata.loc[metadata["split"].eq("train"), "lotName"].to_numpy()
    folds = make_lot_grouped_oof_folds(labels[train_positions], groups, seed)
    device = select_device()
    print(f"FORMAL_STACKING_START seed={seed} device={device}", flush=True)

    for fold_id, (relative_train, relative_holdout) in enumerate(folds):
        fold_seed = seed + fold_id * 10_000
        fold_train = train_positions[relative_train]
        fold_holdout = train_positions[relative_holdout]
        train_mfe_stage(
            stages_dir / f"fold{fold_id}_mfe",
            features,
            labels,
            fold_train,
            fold_holdout,
            validation_positions,
            fold_seed,
            device,
        )
        if pause_after_stage(f"fold{fold_id}_mfe", stop_after):
            return
        train_cnn_stage(
            stages_dir / f"fold{fold_id}_cnn",
            images,
            labels,
            fold_train,
            fold_holdout,
            validation_positions,
            fold_seed,
            device,
        )
        if pause_after_stage(f"fold{fold_id}_cnn", stop_after):
            return

    train_mfe_stage(
        stages_dir / "final_mfe",
        features,
        labels,
        train_positions,
        validation_positions,
        validation_positions,
        seed + 90_000,
        device,
    )
    if pause_after_stage("final_mfe", stop_after):
        return
    train_cnn_stage(
        stages_dir / "final_cnn",
        images,
        labels,
        train_positions,
        validation_positions,
        validation_positions,
        seed + 90_000,
        device,
    )
    if pause_after_stage("final_cnn", stop_after):
        return
    fit_and_save_meta(run_dir, train_positions, validation_positions, labels, folds)
    complete_run_manifest(run_dir, seed, protocol_payload)
    print(f"FORMAL_STACKING_TRAINING_COMPLETE seed={seed}")
    print("test_accessed=NO")


def main(argv=None):
    args = parse_args(argv)
    metadata, images, features, feature_manifest = load_and_validate_inputs()
    if args.preflight:
        preflight(args.seed, metadata, images, features, feature_manifest)
    elif args.smoke:
        smoke(args.seed, metadata, images, features)
    elif args.benchmark is not None:
        benchmark(args.seed, args.benchmark, metadata, images, features)
    else:
        execute(
            args.seed,
            metadata,
            images,
            features,
            feature_manifest,
            stop_after=args.stop_after,
        )


if __name__ == "__main__":
    main()
