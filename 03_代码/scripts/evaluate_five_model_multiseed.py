#!/usr/bin/env python3
"""Evaluate five frozen model families without repeating existing inference."""

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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.dataset import WaferMapDataset
from wafermap.lightweight_baselines import (
    EfficientNetB0Baseline,
    MobileNetV3SmallBaseline,
)
from wafermap.models import ResNet18Baseline, ShuffleNetV2Baseline
from wafermap.models_improved import ShuffleNetV2HighRes
from wafermap.paths import EXPERIMENT_DIR, RESULT_DIR


EVALUATION_ID = "20260801_five_model_multiseed_unified_test"
FREEZE_MANIFEST = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260801_五模型统一评估冻结清单.json"
)
EXPECTED_FREEZE_SHA256 = (
    "2a97b07434239589da586aaa5b836d3b1aaba8d957c40cc8f5d5e54fd1139d5f"
)
OUTPUT_DIR = EXPERIMENT_DIR / "metrics" / EVALUATION_ID
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
RESULT_TABLE = RESULT_DIR / "tables" / "table_five_model_multiseed_test.csv"
RESULT_PER_CLASS = (
    RESULT_DIR / "tables" / "table_five_model_multiseed_per_class.csv"
)
ATTEMPT_RECORD = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260801_五模型统一测试运行审计.json"
)
LOCK_FILE = PROJECT_ROOT / "tmp" / f"{EVALUATION_ID}.lock"

BATCH_SIZE = 256
EXPECTED_TEST_SAMPLES = 25943
SEEDS = (42, 123, 2026)

MODEL_FACTORIES = {
    "highres_shufflenet_v2": ShuffleNetV2HighRes,
    "standard_shufflenet_v2": ShuffleNetV2Baseline,
    "resnet18": ResNet18Baseline,
    "mobilenet_v3_small": MobileNetV3SmallBaseline,
    "efficientnet_b0": EfficientNetB0Baseline,
}

METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
)


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-once", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_manifest() -> dict:
    if sha256_file(FREEZE_MANIFEST) != EXPECTED_FREEZE_SHA256:
        raise ValueError("Freeze manifest hash mismatch.")
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "status": "frozen_before_unified_test",
        "evaluation_id": EVALUATION_ID,
        "selection_split": "validation",
        "seeds": list(SEEDS),
        "test_tuning_allowed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Freeze manifest mismatch: {key}")
    if len(manifest.get("models", [])) != 5:
        raise ValueError("Freeze manifest must contain five models.")
    return manifest


def ensure_outputs_absent():
    existing = [
        path
        for path in (OUTPUT_DIR, RESULT_TABLE, RESULT_PER_CLASS, ATTEMPT_RECORD)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to repeat or overwrite unified test outputs:\n"
            + "\n".join(f"- {path}" for path in existing)
        )
    if LOCK_FILE.exists():
        raise FileExistsError(f"Evaluation lock exists: {LOCK_FILE}")


def validate_frozen_files(manifest: dict):
    for group in ("frozen_code", "dataset"):
        for item in manifest[group]:
            path = PROJECT_ROOT / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Frozen file mismatch: {path}")


def instantiate_and_validate_checkpoint(model_info: dict, checkpoint_info: dict):
    checkpoint_file = PROJECT_ROOT / checkpoint_info["checkpoint_path"]
    if sha256_file(checkpoint_file) != checkpoint_info["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = MODEL_FACTORIES[model_info["model_id"]]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != model_info["parameter_count"]:
        raise ValueError(f"Parameter count mismatch: {checkpoint_file}")
    if int(checkpoint["epoch"]) != int(checkpoint_info["best_epoch"]):
        raise ValueError(f"Best epoch mismatch: {checkpoint_file}")
    if int(checkpoint["random_seed"]) != int(checkpoint_info["seed"]):
        raise ValueError(f"Seed mismatch: {checkpoint_file}")
    return model, checkpoint


def validate_all(manifest: dict):
    validate_frozen_files(manifest)
    actions = []
    for model_info in manifest["models"]:
        for checkpoint_info in model_info["checkpoints"]:
            model, _ = instantiate_and_validate_checkpoint(
                model_info, checkpoint_info
            )
            del model
            reused = checkpoint_info["reused_predictions_path"]
            if reused:
                path = PROJECT_ROOT / reused
                if sha256_file(path) != checkpoint_info["reused_predictions_sha256"]:
                    raise ValueError(f"Reused prediction hash mismatch: {path}")
            actions.append(checkpoint_info["evaluation_action"])
    if actions.count("infer_once") != 11 or actions.count(
        "reuse_frozen_predictions"
    ) != 4:
        raise ValueError("Unexpected evaluation action counts.")


def build_test_dataset() -> WaferMapDataset:
    dataset = WaferMapDataset("test")
    if len(dataset) != EXPECTED_TEST_SAMPLES or dataset.transform is not None:
        raise ValueError("Unexpected fixed test dataset.")
    expected_counts = np.asarray([644, 83, 778, 1452, 539, 21, 130, 179, 22117])
    actual_counts = np.bincount(dataset.labels, minlength=NUM_CLASSES)
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError("Test class counts changed.")
    return dataset


def infer(model, dataset, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    model = model.to(device).eval()
    probabilities = []
    labels = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(target.numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    probability_array = np.concatenate(probabilities).astype(np.float32)
    label_array = np.concatenate(labels).astype(np.int64)
    if not np.array_equal(label_array, dataset.labels):
        raise RuntimeError("Test sample order changed.")
    if probability_array.shape != (EXPECTED_TEST_SAMPLES, NUM_CLASSES):
        raise RuntimeError("Unexpected probability array shape.")
    if not np.isfinite(probability_array).all() or not np.allclose(
        probability_array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise FloatingPointError("Invalid probabilities.")
    return probability_array, elapsed


def prediction_frame(dataset, probabilities: np.ndarray) -> pd.DataFrame:
    y_pred = probabilities.argmax(axis=1).astype(np.int64)
    frame = dataset.metadata.copy()
    frame["true_label_id"] = dataset.labels
    frame["predicted_label_id"] = y_pred
    frame["predicted_label"] = [WM811K_CLASS_NAMES[index] for index in y_pred]
    frame["confidence"] = probabilities.max(axis=1)
    frame["correct"] = y_pred == dataset.labels
    for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
        slug = class_name.lower().replace("-", "_")
        frame[f"prob_{class_id}_{slug}"] = probabilities[:, class_id]
    return frame


def normalize_reused_predictions(path: Path, dataset) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != len(dataset):
        raise ValueError(f"Reused prediction row count mismatch: {path}")
    if not np.array_equal(frame["source_index"].to_numpy(), dataset.metadata["source_index"]):
        raise ValueError(f"Reused prediction order mismatch: {path}")
    true_column = "true_label_id" if "true_label_id" in frame else "label_id"
    if not np.array_equal(frame[true_column].to_numpy(), dataset.labels):
        raise ValueError(f"Reused labels mismatch: {path}")
    required = ["source_index", true_column, "predicted_label_id"]
    result = frame[required].copy()
    result = result.rename(columns={true_column: "true_label_id"})
    result["predicted_label_id"] = result["predicted_label_id"].astype(int)
    return result


def metric_rows(model_info: dict, checkpoint_info: dict, frame: pd.DataFrame):
    y_true = frame["true_label_id"].to_numpy(dtype=np.int64)
    y_pred = frame["predicted_label_id"].to_numpy(dtype=np.int64)
    labels = np.arange(NUM_CLASSES)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    summary = {
        "model_id": model_info["model_id"],
        "model": model_info["display_name"],
        "seed": int(checkpoint_info["seed"]),
        "run_name": checkpoint_info["run_name"],
        "best_epoch": int(checkpoint_info["best_epoch"]),
        "parameter_count": int(model_info["parameter_count"]),
        "evaluation_action": checkpoint_info["evaluation_action"],
        "test_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    class_rows = []
    for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
        class_rows.append(
            {
                "model_id": model_info["model_id"],
                "model": model_info["display_name"],
                "seed": int(checkpoint_info["seed"]),
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1_score": float(class_f1[class_id]),
                "support": int(support[class_id]),
            }
        )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, class_rows, matrix


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, model), group in per_seed.groupby(
        ["model_id", "model"], sort=False
    ):
        row = {
            "model_id": model_id,
            "model": model,
            "training_runs": len(group),
            "parameter_count": int(group["parameter_count"].iloc[0]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def write_audit(status: str, **extra):
    record = {
        "evaluation_id": EVALUATION_ID,
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
        "freeze_manifest": str(FREEZE_MANIFEST.relative_to(PROJECT_ROOT)),
        "freeze_manifest_sha256": EXPECTED_FREEZE_SHA256,
        "evaluator_sha256": sha256_file(Path(__file__)),
        **extra,
    }
    ATTEMPT_RECORD.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(manifest: dict):
    device = select_device()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    write_audit("started", device=str(device))

    staging = Path(
        tempfile.mkdtemp(prefix=f"{EVALUATION_ID}_", dir=PROJECT_ROOT / "tmp")
    )
    staging_predictions = staging / "predictions"
    staging_predictions.mkdir(parents=True)
    dataset = build_test_dataset()
    summaries = []
    class_rows = []
    matrices = {}
    durations = {}

    try:
        for model_info in manifest["models"]:
            for checkpoint_info in model_info["checkpoints"]:
                seed = int(checkpoint_info["seed"])
                key = f"{model_info['model_id']}_seed{seed}"
                if checkpoint_info["evaluation_action"] == "reuse_frozen_predictions":
                    frame = normalize_reused_predictions(
                        PROJECT_ROOT / checkpoint_info["reused_predictions_path"],
                        dataset,
                    )
                    durations[key] = None
                    print(f"REUSE {key}", flush=True)
                else:
                    model, _ = instantiate_and_validate_checkpoint(
                        model_info, checkpoint_info
                    )
                    probabilities, elapsed = infer(model, dataset, device)
                    frame = prediction_frame(dataset, probabilities)
                    durations[key] = elapsed
                    print(f"INFER {key}: {elapsed:.2f}s", flush=True)
                    del model, probabilities
                    if device.type == "mps":
                        torch.mps.empty_cache()
                prediction_file = staging_predictions / f"{key}.csv"
                frame.to_csv(prediction_file, index=False)
                summary, rows, matrix = metric_rows(
                    model_info, checkpoint_info, frame
                )
                summaries.append(summary)
                class_rows.extend(rows)
                matrices[key] = matrix.tolist()

        per_seed = pd.DataFrame(summaries)
        if len(per_seed) != 15:
            raise RuntimeError("Unified evaluation did not produce 15 rows.")
        aggregate_table = aggregate(per_seed)
        per_class = pd.DataFrame(class_rows)
        per_seed.to_csv(staging / "per_seed_metrics.csv", index=False)
        aggregate_table.to_csv(staging / "aggregate_metrics.csv", index=False)
        per_class.to_csv(staging / "per_class_metrics.csv", index=False)
        (staging / "confusion_matrices.json").write_text(
            json.dumps(matrices, indent=2) + "\n", encoding="utf-8"
        )
        run_manifest = {
            "evaluation_id": EVALUATION_ID,
            "completed_at": datetime.now().astimezone().isoformat(),
            "device": str(device),
            "test_samples": len(dataset),
            "new_inference_runs": 11,
            "reused_prediction_runs": 4,
            "durations_seconds": durations,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        (staging / "run_manifest.json").write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        shutil.move(str(staging), str(OUTPUT_DIR))
        RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_DIR / "aggregate_metrics.csv", RESULT_TABLE)
        shutil.copy2(OUTPUT_DIR / "per_class_metrics.csv", RESULT_PER_CLASS)

        artifact_hashes = {
            str(path.relative_to(OUTPUT_DIR)): sha256_file(path)
            for path in sorted(OUTPUT_DIR.rglob("*"))
            if path.is_file()
        }
        write_audit(
            "completed",
            device=str(device),
            output_dir=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
            artifact_hashes=artifact_hashes,
        )
        print(f"Unified evaluation completed: {OUTPUT_DIR}")
        print(aggregate_table.to_string(index=False))
    except Exception as error:
        write_audit("failed", device=str(device), error=repr(error))
        raise
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def main():
    args = parse_args()
    ensure_outputs_absent()
    manifest = load_manifest()
    validate_all(manifest)
    if args.preflight:
        print("Unified evaluation preflight: PASS")
        print("Models=5 checkpoints=15 infer_once=11 reuse=4")
        return
    run(manifest)


if __name__ == "__main__":
    main()
