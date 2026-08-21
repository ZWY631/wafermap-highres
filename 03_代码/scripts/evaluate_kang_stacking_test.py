#!/usr/bin/env python3
"""One-time fixed-test evaluation for completed Kang stacking seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码/src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.stacking_reproduction import (
    EVAL_BATCH_SIZE,
    MetaRidgeModel,
    FeatureDataset,
    HandcraftedFNN,
    VGG16WaferCNN,
    WaferBinaryDataset,
    classification_metrics,
    predict_probabilities,
    select_device,
)


SEEDS = (42, 123, 2026)
CHECKPOINT_ROOT = PROJECT_ROOT / "04_实验/checkpoints/kang_kang_stacking_lot_reproduction"
OUTPUT_DIR = PROJECT_ROOT / "04_实验/metrics/20260807_kang_kang_stacking_lot_reproduction"
IMAGES_FILE = PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/images_uint8.npy"
METADATA_FILE = PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/metadata.csv"
FEATURE_FILE = PROJECT_ROOT / "02_数据/processed/wm811k_stacking_features/handcrafted_features_float32.npy"
EXPECTED_TEST_SAMPLES = 25_943


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def completed_run_paths():
    paths = {}
    for seed in SEEDS:
        run_dir = CHECKPOINT_ROOT / f"seed{seed}"
        manifest = run_dir / "run_manifest.json"
        paths[seed] = (run_dir, manifest)
    return paths


def validate_completed_runs(require_all=True):
    states = {}
    for seed, (run_dir, manifest_path) in completed_run_paths().items():
        if not manifest_path.is_file():
            states[seed] = "missing"
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "training_complete_test_not_accessed":
            raise ValueError(f"Unexpected run status for seed {seed}")
        if payload.get("test_accessed") is not False:
            raise ValueError(f"Training manifest indicates test access for seed {seed}")
        for relative, expected in payload["stage_manifest_sha256"].items():
            path = run_dir / relative
            if sha256_file(path) != expected:
                raise ValueError(f"Stage manifest hash mismatch: {path}")
        states[seed] = "complete"
    if require_all and set(states.values()) != {"complete"}:
        raise FileNotFoundError(f"All three training runs are required: {states}")
    return states


def build_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def load_mfe(run_dir, features, positions, labels, device):
    stage = run_dir / "stages/final_mfe"
    scaler = np.load(stage / "scaler.npz")
    scaled = ((features[positions] - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    loader = build_loader(FeatureDataset(scaled, labels[positions]))
    checkpoint = torch.load(stage / "best.pt", map_location="cpu", weights_only=False)
    model = HandcraftedFNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return predict_probabilities(model, loader, device)[0]


def load_cnn(run_dir, images, positions, labels, device):
    stage = run_dir / "stages/final_cnn"
    loader = build_loader(WaferBinaryDataset(images, positions, labels[positions]))
    checkpoint = torch.load(stage / "best.pt", map_location="cpu", weights_only=False)
    model = VGG16WaferCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return predict_probabilities(model, loader, device)[0]


def load_meta(run_dir):
    payload = np.load(run_dir / "meta/meta_ridge.npz")
    return MetaRidgeModel(
        coefficient=payload["coefficient"],
        intercept=payload["intercept"],
        alpha=float(payload["alpha"]),
    )


def evaluate_seed(seed, metadata, images, features, positions, labels, device, staging):
    run_dir = CHECKPOINT_ROOT / f"seed{seed}"
    start = time.perf_counter()
    mfe = load_mfe(run_dir, features, positions, labels, device)
    cnn = load_cnn(run_dir, images, positions, labels, device)
    meta = load_meta(run_dir)
    base = np.concatenate((mfe, cnn), axis=1)
    predictions = meta.predict(base)
    metrics = classification_metrics(labels[positions], predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels[positions],
        predictions,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )
    frame = metadata.iloc[positions][
        ["source_index", "lotName", "waferIndex", "label", "label_id"]
    ].reset_index(drop=True)
    frame["predicted_label_id"] = predictions
    frame["predicted_label"] = [WM811K_CLASS_NAMES[index] for index in predictions]
    frame["correct"] = predictions == labels[positions]
    frame.to_csv(staging / f"predictions_seed{seed}.csv", index=False)
    per_class = pd.DataFrame(
        {
            "seed": seed,
            "label_id": np.arange(NUM_CLASSES),
            "label": WM811K_CLASS_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    return (
        {"seed": seed, **metrics, "seconds": time.perf_counter() - start},
        per_class,
        confusion_matrix(labels[positions], predictions, labels=np.arange(NUM_CLASSES)),
    )


def execute():
    validate_completed_runs(require_all=True)
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing repeat test evaluation: {OUTPUT_DIR}")
    metadata = pd.read_csv(METADATA_FILE)
    images = np.load(IMAGES_FILE, mmap_mode="r")
    features = np.load(FEATURE_FILE, mmap_mode="r")
    positions = np.flatnonzero(metadata["split"].eq("test").to_numpy())
    if len(positions) != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"Unexpected test sample count: {len(positions)}")
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    device = select_device()
    staging = Path(tempfile.mkdtemp(prefix="kang_stacking_test_", dir=OUTPUT_DIR.parent))
    try:
        metric_rows = []
        class_frames = []
        matrices = {}
        for seed in SEEDS:
            metrics, per_class, matrix = evaluate_seed(
                seed, metadata, images, features, positions, labels, device, staging
            )
            metric_rows.append(metrics)
            class_frames.append(per_class)
            matrices[str(seed)] = matrix.tolist()
        per_seed = pd.DataFrame(metric_rows)
        per_seed.to_csv(staging / "per_seed_metrics.csv", index=False)
        pd.concat(class_frames, ignore_index=True).to_csv(
            staging / "per_class_metrics.csv", index=False
        )
        (staging / "confusion_matrices.json").write_text(
            json.dumps(matrices, indent=2) + "\n", encoding="utf-8"
        )
        aggregate = []
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = per_seed[metric].tolist()
            aggregate.append(
                {
                    "metric": metric,
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values),
                }
            )
        pd.DataFrame(aggregate).to_csv(staging / "aggregate_metrics.csv", index=False)
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "status": "fixed_test_evaluation_complete",
                    "samples": len(positions),
                    "seeds": list(SEEDS),
                    "device": str(device),
                    "platform": platform.platform(),
                    "training_manifest_sha256": {
                        str(seed): sha256_file(
                            CHECKPOINT_ROOT / f"seed{seed}/run_manifest.json"
                        )
                        for seed in SEEDS
                    },
                    "script_sha256": sha256_file(Path(__file__)),
                    "artifact_sha256": artifact_hashes,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.rename(OUTPUT_DIR)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("KANG_STACKING_FIXED_TEST_COMPLETE")
    print(f"output={OUTPUT_DIR}")


def main(argv=None):
    args = parse_args(argv)
    if args.preflight:
        states = validate_completed_runs(require_all=False)
        print("KANG_STACKING_TEST_PREFLIGHT")
        print(f"training_states={states}")
        print(f"output_exists={OUTPUT_DIR.exists()}")
        print("test_labels_read=NO")
        print("test_inference_started=NO")
    else:
        execute()


if __name__ == "__main__":
    main()

