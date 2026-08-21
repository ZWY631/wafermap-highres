#!/usr/bin/env python3
"""Run the single frozen multi-seed test evaluation for the final model."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from torch import nn
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import IMAGE_SIZE, NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.dataset import WaferMapDataset
from wafermap.models_improved import ShuffleNetV2HighRes
from wafermap.paths import EXPERIMENT_DIR, PROJECT_ROOT, RESULT_DIR


EVALUATION_ID = "20260729_highres_ce_multiseed_final_test"
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "c682922b1c291f03a41a81e5d15ebfe08e4456e77c816f7943041fc11fcef8db"
)
FREEZE_MANIFEST = (
    PROJECT_ROOT / "00_项目管理" / "20260729_最终模型冻结清单.json"
)
COMPLETION_RECORD = (
    PROJECT_ROOT / "00_项目管理" / "20260729_最终测试完成记录.md"
)
STARTED_RECORD = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260729_最终测试运行审计"
    / "attempt_started.json"
)
AUDIT_DIR = STARTED_RECORD.parent

RAW_OUTPUT_DIR = EXPERIMENT_DIR / "metrics" / EVALUATION_ID
PREDICTIONS_DIR = RESULT_DIR / "predictions" / EVALUATION_ID
FIGURE_PNG = (
    RESULT_DIR
    / "figures"
    / "model_results"
    / "final_highres_ce_multiseed_confusion_matrix_mean.png"
)
FIGURE_PDF = FIGURE_PNG.with_suffix(".pdf")
PAPER_SUMMARY_TABLE = (
    RESULT_DIR / "tables" / "table_final_highres_ce_multiseed_test.csv"
)
PAPER_PER_CLASS_TABLE = (
    RESULT_DIR / "tables" / "table_final_highres_ce_per_class_test.csv"
)
LOCK_FILE = PROJECT_ROOT / "tmp" / f"{EVALUATION_ID}.lock"

BATCH_SIZE = 256
NUM_WORKERS = 0
EXPECTED_TEST_SAMPLES = 25943
EXPECTED_SEEDS = (42, 123, 2026)

EXPECTED_CHECKPOINTS = (
    {
        "seed": 42,
        "run_name": "shufflenet_v2_highres_ce_full",
        "path": "04_实验/checkpoints/shufflenet_v2_highres_ce_full/best.pt",
        "sha256": (
            "cafebb6a76ef566ea05fa47e7745677c35cb3abbec87b1de468c0898b02c67bc"
        ),
        "best_epoch": 28,
        "validation_macro_f1": 0.9053773248524236,
    },
    {
        "seed": 123,
        "run_name": "shufflenet_v2_highres_ce_full_seed123",
        "path": (
            "04_实验/checkpoints/"
            "shufflenet_v2_highres_ce_full_seed123/best.pt"
        ),
        "sha256": (
            "189785ae00a95ee3c342b5f29e5b8306bf864256016259b19e80be333226929f"
        ),
        "best_epoch": 27,
        "validation_macro_f1": 0.9072335751434788,
    },
    {
        "seed": 2026,
        "run_name": "shufflenet_v2_highres_ce_full_seed2026",
        "path": (
            "04_实验/checkpoints/"
            "shufflenet_v2_highres_ce_full_seed2026/best.pt"
        ),
        "sha256": (
            "15791aaddb863abc45f1de55c0c1e8781420238ae7b97c0b6a84740d155c0c7b"
        ),
        "best_epoch": 26,
        "validation_macro_f1": 0.9020327874707612,
    },
)

AGGREGATE_METRICS = (
    "test_loss",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "elapsed_seconds",
    "samples_per_second",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the three frozen HighRes ShuffleNetV2 + CE checkpoints "
            "on the fixed WM-811K test split."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the frozen protocol without performing inference.",
    )
    mode.add_argument(
        "--run-final-test",
        action="store_true",
        help=(
            "Perform the one-time final test. Existing outputs cause an "
            "immediate refusal."
        ),
    )
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


def ensure_outputs_are_available(check_lock: bool = True):
    output_paths = (
        RAW_OUTPUT_DIR,
        PREDICTIONS_DIR,
        FIGURE_PNG,
        FIGURE_PDF,
        PAPER_SUMMARY_TABLE,
        PAPER_PER_CLASS_TABLE,
        AUDIT_DIR,
        COMPLETION_RECORD,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to repeat or overwrite the frozen final test. "
            "Existing outputs:\n"
            f"{formatted}"
        )

    if check_lock and LOCK_FILE.exists():
        raise FileExistsError(
            f"A final-test lock already exists: {LOCK_FILE}. "
            "Confirm that no evaluation process is running before removing it."
        )


def load_and_validate_manifest() -> tuple[dict, str]:
    if not FREEZE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing freeze manifest: {FREEZE_MANIFEST}")

    manifest_hash = sha256_file(FREEZE_MANIFEST)
    if manifest_hash != EXPECTED_FREEZE_MANIFEST_SHA256:
        raise ValueError(
            "Freeze manifest hash differs from the digest anchored in the "
            "final evaluator. Refusing to evaluate."
        )
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))

    expected_fields = {
        "status": "frozen_before_test",
        "final_evaluation_id": EVALUATION_ID,
        "selection_split": "validation",
        "current_paired_selection_used_test_predictions": False,
        "test_split_was_untouched_during_entire_project": False,
        "selected_architecture_id": "highres_ce_no_eca",
        "model_name": "ShuffleNetV2HighRes",
        "loss_name": "CrossEntropyLoss",
        "parameter_count": 1262397,
        "image_size": IMAGE_SIZE,
        "input_channels": 1,
        "class_names": list(WM811K_CLASS_NAMES),
        "seeds": list(EXPECTED_SEEDS),
    }
    for key, expected_value in expected_fields.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Freeze manifest mismatch: {key}={actual_value!r}, "
                f"expected {expected_value!r}."
            )

    protocol = manifest.get("protocol", {})
    expected_protocol = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "inference_mode": True,
        "one_evaluation_per_seed": True,
        "post_test_model_tuning_allowed": False,
        "overwrite_existing_outputs": False,
    }
    for key, expected_value in expected_protocol.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"Freeze protocol mismatch: {key}={protocol.get(key)!r}, "
                f"expected {expected_value!r}."
            )

    return manifest, manifest_hash


def validate_frozen_code_and_environment(manifest: dict):
    frozen_code = manifest.get("frozen_code", [])
    if not frozen_code:
        raise ValueError("Freeze manifest does not contain frozen source hashes.")

    for item in frozen_code:
        source_file = PROJECT_ROOT / item["path"]
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing frozen source file: {source_file}")
        if sha256_file(source_file) != item["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {source_file}")

    actual_environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if manifest.get("frozen_environment") != actual_environment:
        raise ValueError(
            "Runtime package versions differ from the frozen environment: "
            f"actual={actual_environment!r}"
        )

    historical_access = manifest.get("historical_test_access", [])
    if len(historical_access) != 2:
        raise ValueError("Historical test access is not fully documented.")
    for item in historical_access:
        result_file = PROJECT_ROOT / item["result"]
        if not result_file.is_file():
            raise FileNotFoundError(
                f"Missing documented historical test record: {result_file}"
            )
        if sha256_file(result_file) != item["sha256"]:
            raise ValueError(
                f"Historical test record hash mismatch: {result_file}"
            )


def validate_dataset(manifest: dict) -> WaferMapDataset:
    dataset_info = manifest["dataset"]
    images_file = PROJECT_ROOT / dataset_info["processed_images"]
    metadata_file = PROJECT_ROOT / dataset_info["metadata"]

    for path in (images_file, metadata_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen dataset file: {path}")

    if sha256_file(images_file) != dataset_info["processed_images_sha256"]:
        raise ValueError("Processed image file hash differs from freeze manifest.")
    if sha256_file(metadata_file) != dataset_info["metadata_sha256"]:
        raise ValueError("Metadata file hash differs from freeze manifest.")

    dataset = WaferMapDataset("test")
    if dataset.transform is not None:
        raise ValueError("Final test dataset must not use random transforms.")
    if len(dataset) != dataset_info["test_samples"]:
        raise ValueError("Test sample count differs from freeze manifest.")
    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, found {len(dataset)}."
        )

    class_counts = np.bincount(dataset.labels, minlength=NUM_CLASSES).tolist()
    if class_counts != dataset_info["test_class_counts"]:
        raise ValueError("Test class counts differ from freeze manifest.")

    expected_label_names = np.asarray(WM811K_CLASS_NAMES)[dataset.labels]
    actual_label_names = dataset.metadata["label"].astype(str).to_numpy()
    if not np.array_equal(expected_label_names, actual_label_names):
        raise ValueError("Test label IDs and class names are inconsistent.")

    full_metadata = pd.read_csv(metadata_file)
    lot_sets = {
        split: set(full_metadata.loc[full_metadata["split"] == split, "lotName"])
        for split in ("train", "val", "test")
    }
    if not lot_sets["train"].isdisjoint(lot_sets["val"]):
        raise ValueError("Train and validation lots overlap.")
    if not lot_sets["train"].isdisjoint(lot_sets["test"]):
        raise ValueError("Train and test lots overlap.")
    if not lot_sets["val"].isdisjoint(lot_sets["test"]):
        raise ValueError("Validation and test lots overlap.")

    return dataset


def load_and_validate_checkpoint(
    checkpoint_info: dict,
    manifest: dict,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint_file = PROJECT_ROOT / checkpoint_info["path"]
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint_file}")
    if sha256_file(checkpoint_file) != checkpoint_info["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    expected_fields = {
        "run_name": checkpoint_info["run_name"],
        "model_name": manifest["model_name"],
        "loss_name": manifest["loss_name"],
        "image_size": IMAGE_SIZE,
        "parameter_count": manifest["parameter_count"],
        "epoch": checkpoint_info["best_epoch"],
        "class_names": list(WM811K_CLASS_NAMES),
        "random_seed": checkpoint_info["seed"],
    }
    for key, expected_value in expected_fields.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint mismatch in {checkpoint_file}: "
                f"{key}={actual_value!r}, expected {expected_value!r}."
            )

    if not np.isclose(
        float(checkpoint["best_val_macro_f1"]),
        float(checkpoint_info["validation_macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Frozen validation Macro-F1 mismatch: {checkpoint_file}"
        )

    model = ShuffleNetV2HighRes(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != manifest["parameter_count"]:
        raise ValueError("Instantiated model parameter count is incorrect.")

    model = model.to(device)
    model.eval()
    return model, checkpoint


def validate_all_checkpoints(manifest: dict):
    checkpoints = manifest.get("checkpoints", [])
    if checkpoints != list(EXPECTED_CHECKPOINTS):
        raise ValueError(
            "Checkpoint definitions differ from the evaluator's independently "
            "anchored checkpoint list."
        )
    if tuple(item["seed"] for item in checkpoints) != EXPECTED_SEEDS:
        raise ValueError("Checkpoint seed order differs from frozen seed order.")

    for checkpoint_info in checkpoints:
        model, _ = load_and_validate_checkpoint(
            checkpoint_info,
            manifest,
            torch.device("cpu"),
        )
        del model


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    dataset: WaferMapDataset,
    device: torch.device,
) -> dict:
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    synchronize(device)
    start_time = time.perf_counter()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)

        total_loss += loss.item() * labels.shape[0]
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    y_true = np.concatenate(all_labels).astype(np.int64, copy=False)
    probabilities = np.concatenate(all_probabilities).astype(
        np.float32, copy=False
    )

    if not np.isfinite(total_loss):
        raise FloatingPointError("Test loss became NaN or infinite.")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Predicted probabilities contain NaN or Inf.")
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise FloatingPointError("Predicted class probabilities do not sum to one.")

    y_pred = probabilities.argmax(axis=1).astype(np.int64, copy=False)
    confidences = probabilities.max(axis=1)

    if len(y_true) != len(dataset):
        raise RuntimeError("Prediction count does not match test dataset size.")
    if not np.array_equal(y_true, dataset.labels):
        raise RuntimeError("Test DataLoader order changed unexpectedly.")
    if probabilities.shape != (len(dataset), NUM_CLASSES):
        raise RuntimeError(f"Unexpected probability shape: {probabilities.shape}")

    return {
        "test_loss": total_loss / len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "confidences": confidences,
        "elapsed_seconds": elapsed_seconds,
    }


def calculate_metrics(
    results: dict,
    checkpoint_info: dict,
    checkpoint: dict,
    device: torch.device,
) -> tuple[dict, list[dict], np.ndarray]:
    labels = np.arange(NUM_CLASSES)
    y_true = results["y_true"]
    y_pred = results["y_pred"]

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    summary = {
        "seed": int(checkpoint_info["seed"]),
        "run_name": checkpoint_info["run_name"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameter_count": int(checkpoint["parameter_count"]),
        "device": str(device),
        "test_samples": int(len(y_true)),
        "test_loss": float(results["test_loss"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "elapsed_seconds": float(results["elapsed_seconds"]),
        "samples_per_second": float(len(y_true) / results["elapsed_seconds"]),
    }

    numeric_summary = [
        value
        for key, value in summary.items()
        if key
        not in {
            "seed",
            "run_name",
            "checkpoint_epoch",
            "parameter_count",
            "device",
            "test_samples",
        }
    ]
    if not np.isfinite(np.asarray(numeric_summary, dtype=np.float64)).all():
        raise FloatingPointError("Calculated test metrics contain NaN or Inf.")
    for values, name in (
        (precision, "per-class precision"),
        (recall, "per-class recall"),
        (class_f1, "per-class F1"),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} contains NaN or Inf.")

    per_class_rows = []
    for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
        per_class_rows.append(
            {
                "seed": int(checkpoint_info["seed"]),
                "run_name": checkpoint_info["run_name"],
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1_score": float(class_f1[class_id]),
                "support": int(support[class_id]),
            }
        )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, per_class_rows, matrix


def normalized_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def aggregate_rows(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary_json = {
        "evaluation_id": EVALUATION_ID,
        "n_seeds": len(per_seed),
        "seeds": per_seed["seed"].astype(int).tolist(),
        "sample_standard_deviation_ddof": 1,
        "metrics": {},
    }
    for metric in AGGREGATE_METRICS:
        values = per_seed[metric].astype(float)
        row = {
            "metric": metric,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
        rows.append(row)
        summary_json["metrics"][metric] = {
            key: value for key, value in row.items() if key != "metric"
        }
    return pd.DataFrame(rows), summary_json


def aggregate_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (class_id, class_name), group in per_class.groupby(
        ["class_id", "class_name"], sort=True
    ):
        row = {
            "class_id": int(class_id),
            "class_name": class_name,
            "support_per_seed": int(group["support"].iloc[0]),
        }
        for metric in ("precision", "recall", "f1_score"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def save_predictions(
    path: Path,
    dataset: WaferMapDataset,
    results: dict,
):
    predictions = dataset.metadata.copy()
    predictions["true_label_id"] = results["y_true"]
    predictions["predicted_label_id"] = results["y_pred"]
    predictions["predicted_label"] = [
        WM811K_CLASS_NAMES[class_id] for class_id in results["y_pred"]
    ]
    predictions["confidence"] = results["confidences"]
    predictions["correct"] = results["y_true"] == results["y_pred"]

    for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
        safe_name = class_name.lower().replace("-", "_")
        predictions[f"prob_{class_id}_{safe_name}"] = results[
            "probabilities"
        ][:, class_id]

    predictions.to_csv(path, index=False)


def save_confusion_figure(
    mean_matrix: np.ndarray,
    output_png: Path,
    output_pdf: Path,
):
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        mean_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=WM811K_CLASS_NAMES,
        yticklabels=WM811K_CLASS_NAMES,
        cbar_kws={"label": "Mean row-normalized proportion"},
        ax=axis,
    )
    axis.set_title(
        "Final Test: High-Resolution ShuffleNetV2 + CE\n"
        "Mean Normalized Confusion Matrix Across Three Seeds"
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    plt.setp(axis.get_yticklabels(), rotation=0)
    figure.tight_layout()
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def build_paper_summary_table(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    aggregate_index = aggregate.set_index("metric")
    row = {
        "model": "HighRes ShuffleNetV2 + CE (no ECA)",
        "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
        "test_samples_per_seed": EXPECTED_TEST_SAMPLES,
        "parameters": int(per_seed["parameter_count"].iloc[0]),
    }
    for metric, column_name in (
        ("accuracy", "accuracy_percent_mean_sd"),
        ("macro_f1", "macro_f1_percent_mean_sd"),
        ("weighted_f1", "weighted_f1_percent_mean_sd"),
        ("balanced_accuracy", "balanced_accuracy_percent_mean_sd"),
    ):
        mean = aggregate_index.loc[metric, "mean"] * 100.0
        std = aggregate_index.loc[metric, "sample_std"] * 100.0
        row[column_name] = f"{mean:.4f} +/- {std:.4f}"
    return pd.DataFrame([row])


def build_paper_per_class_table(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in per_class.iterrows():
        output_row = {
            "class": row["class_name"],
            "support_per_seed": int(row["support_per_seed"]),
        }
        for metric in ("precision", "recall", "f1_score"):
            mean = row[f"{metric}_mean"] * 100.0
            std = row[f"{metric}_sample_std"] * 100.0
            output_row[f"{metric}_percent_mean_sd"] = (
                f"{mean:.4f} +/- {std:.4f}"
            )
        rows.append(output_row)
    return pd.DataFrame(rows)


def completion_markdown(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    device: torch.device,
) -> str:
    aggregate_index = aggregate.set_index("metric")

    def percent(metric: str) -> str:
        mean = aggregate_index.loc[metric, "mean"] * 100.0
        std = aggregate_index.loc[metric, "sample_std"] * 100.0
        return f"{mean:.4f}% +/- {std:.4f}%"

    rows = []
    for _, row in per_seed.iterrows():
        rows.append(
            f"| {int(row['seed'])} | {int(row['checkpoint_epoch'])} | "
            f"{row['accuracy'] * 100:.4f}% | {row['macro_f1'] * 100:.4f}% | "
            f"{row['balanced_accuracy'] * 100:.4f}% |"
        )

    return "\n".join(
        [
            "# 最终测试完成记录",
            "",
            f"- 完成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- 评估编号：`{EVALUATION_ID}`",
            f"- 设备：`{device}`",
            "- 冻结模型：HighRes ShuffleNetV2 + CrossEntropyLoss（无 ECA）",
            "- 测试样本：25,943 张/seed，固定 lot 隔离测试集",
            "- 重要约束：本次测试完成后，不再根据测试结果修改模型。",
            "- 方法学说明：该 benchmark test split 此前用于两个历史基准模型，",
            "  因此不能称为整个研究过程中从未访问过的全新 holdout。",
            "",
            "## 三个随机种子的结果",
            "",
            "| Seed | 最佳 epoch | Accuracy | Macro-F1 | Balanced Accuracy |",
            "|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 最终报告值",
            "",
            f"- Accuracy：{percent('accuracy')}",
            f"- Macro-F1：{percent('macro_f1')}",
            f"- Weighted-F1：{percent('weighted_f1')}",
            f"- Balanced Accuracy：{percent('balanced_accuracy')}",
            "",
            "标准差为三个随机种子的样本标准差（ddof=1）。不能只挑其中最好的一次报告。",
            "",
            "## 文件位置",
            "",
            f"- 原始指标：`04_实验/metrics/{EVALUATION_ID}`",
            f"- 逐样本预测：`05_结果/predictions/{EVALUATION_ID}`",
            "- 论文表格：`05_结果/tables`",
            "- 混淆矩阵图：`05_结果/figures/model_results`",
            "",
        ]
    )


def write_run_manifest(
    path: Path,
    freeze_manifest_hash: str,
    device: torch.device,
    per_seed: pd.DataFrame,
):
    run_manifest = {
        "evaluation_id": EVALUATION_ID,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze_manifest": str(FREEZE_MANIFEST.relative_to(PROJECT_ROOT)),
        "freeze_manifest_sha256": freeze_manifest_hash,
        "irreversible_start_record": str(STARTED_RECORD.relative_to(PROJECT_ROOT)),
        "irreversible_start_record_sha256": sha256_file(STARTED_RECORD),
        "evaluation_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "device": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "test_samples_per_seed": EXPECTED_TEST_SAMPLES,
        "seeds": list(EXPECTED_SEEDS),
        "per_seed_checkpoint_sha256": {
            str(int(row["seed"])): row["checkpoint_sha256"]
            for _, row in per_seed.iterrows()
        },
        "per_seed_audit_record_sha256": {
            str(seed): sha256_file(
                AUDIT_DIR / f"seed{seed}_inference_completed.json"
            )
            for seed in EXPECTED_SEEDS
        },
        "test_set_used_once_per_checkpoint": True,
        "test_split_was_untouched_during_entire_project": False,
        "post_test_tuning_allowed": False,
    }
    path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def acquire_lock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(f"pid={os.getpid()}\n")


def create_irreversible_start_record(freeze_manifest_hash: str):
    """Persist the test start before the first test-set inference batch."""
    AUDIT_DIR.mkdir(parents=False, exist_ok=False)
    record = {
        "evaluation_id": EVALUATION_ID,
        "state": "final_test_started_do_not_rerun_automatically",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "freeze_manifest": str(FREEZE_MANIFEST.relative_to(PROJECT_ROOT)),
        "freeze_manifest_sha256": freeze_manifest_hash,
        "evaluation_script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
        "seeds": list(EXPECTED_SEEDS),
        "test_samples_per_seed": EXPECTED_TEST_SAMPLES,
        "rule": (
            "This marker is intentionally retained after success or failure. "
            "Do not delete it or rerun the final test without a documented "
            "scientific audit of the interrupted run."
        ),
    }

    descriptor = os.open(STARTED_RECORD, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_seed_audit_record(summary: dict, checkpoint_info: dict):
    seed = int(summary["seed"])
    record_file = AUDIT_DIR / f"seed{seed}_inference_completed.json"
    record = {
        "evaluation_id": EVALUATION_ID,
        "state": "test_inference_completed_for_seed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "run_name": summary["run_name"],
        "checkpoint_sha256": checkpoint_info["sha256"],
        "test_samples": int(summary["test_samples"]),
        "accuracy": float(summary["accuracy"]),
        "macro_f1": float(summary["macro_f1"]),
        "balanced_accuracy": float(summary["balanced_accuracy"]),
        "note": (
            "This persistent audit record proves that this checkpoint has "
            "already read the frozen test split."
        ),
    }
    with record_file.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_completion_audit_record(
    aggregate: pd.DataFrame,
    artifact_manifest_hash: str,
):
    aggregate_index = aggregate.set_index("metric")
    record = {
        "evaluation_id": EVALUATION_ID,
        "state": "final_test_completed_and_published",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "canonical_bundle": str(RAW_OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "artifact_manifest_sha256": artifact_manifest_hash,
        "macro_f1_mean": float(aggregate_index.loc["macro_f1", "mean"]),
        "macro_f1_sample_std": float(
            aggregate_index.loc["macro_f1", "sample_std"]
        ),
    }
    record_file = AUDIT_DIR / "evaluation_completed.json"
    with record_file.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_interruption_audit_record(error: BaseException, temp_root: Path):
    record = {
        "evaluation_id": EVALUATION_ID,
        "state": "final_test_interrupted_do_not_rerun_automatically",
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "staging_directory": str(temp_root),
        "canonical_bundle_exists": RAW_OUTPUT_DIR.exists(),
        "rule": (
            "Retain this record and inspect the persistent per-seed audit files "
            "before deciding how the interrupted evaluation should be handled."
        ),
    }
    record_file = AUDIT_DIR / "evaluation_interrupted.json"
    with record_file.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")


def release_lock():
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


def expected_bundle_paths() -> set[str]:
    paths = {
        "test_per_seed_summary.csv",
        "test_aggregate_summary.csv",
        "test_aggregate_summary.json",
        "test_per_class_by_seed.csv",
        "test_per_class_aggregate.csv",
        "confusion_matrix_normalized_mean.csv",
        "confusion_matrix_normalized_sample_std.csv",
        "run_manifest.json",
        f"figures/{FIGURE_PNG.name}",
        f"figures/{FIGURE_PDF.name}",
        f"tables/{PAPER_SUMMARY_TABLE.name}",
        f"tables/{PAPER_PER_CLASS_TABLE.name}",
        f"records/{COMPLETION_RECORD.name}",
    }
    for seed in EXPECTED_SEEDS:
        paths.add(f"confusion_matrix_seed{seed}.csv")
        paths.add(f"confusion_matrix_normalized_seed{seed}.csv")
        paths.add(f"predictions/predictions_seed{seed}.csv")
    return paths


def write_artifact_manifest(bundle_dir: Path) -> Path:
    manifest_file = bundle_dir / "artifact_manifest.json"
    files = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path == manifest_file:
            continue
        files.append(
            {
                "path": str(path.relative_to(bundle_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    actual_paths = {item["path"] for item in files}
    if actual_paths != expected_bundle_paths():
        missing = sorted(expected_bundle_paths() - actual_paths)
        unexpected = sorted(actual_paths - expected_bundle_paths())
        raise RuntimeError(
            "Staged evaluation bundle inventory mismatch. "
            f"missing={missing}, unexpected={unexpected}"
        )

    payload = {
        "evaluation_id": EVALUATION_ID,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_count_excluding_this_manifest": len(files),
        "files": files,
    }
    manifest_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_file


def verify_artifact_manifest(bundle_dir: Path) -> str:
    manifest_file = bundle_dir / "artifact_manifest.json"
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    declared = {item["path"]: item for item in payload["files"]}
    actual_paths = {
        str(path.relative_to(bundle_dir))
        for path in bundle_dir.rglob("*")
        if path.is_file() and path != manifest_file
    }
    if actual_paths != set(declared) or actual_paths != expected_bundle_paths():
        raise RuntimeError("Canonical bundle file inventory is incomplete.")

    for relative_path, item in declared.items():
        path = bundle_dir / relative_path
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Artifact size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Artifact hash mismatch: {path}")

    return sha256_file(manifest_file)


def copy_file_exclusive(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Published artifact hash mismatch: {destination}")


def publish_derived_outputs(canonical_bundle: Path):
    canonical_predictions = canonical_bundle / "predictions"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=False)
    for source in sorted(canonical_predictions.iterdir()):
        if source.is_file():
            copy_file_exclusive(source, PREDICTIONS_DIR / source.name)

    copy_file_exclusive(
        canonical_bundle / "figures" / FIGURE_PNG.name,
        FIGURE_PNG,
    )
    copy_file_exclusive(
        canonical_bundle / "figures" / FIGURE_PDF.name,
        FIGURE_PDF,
    )
    copy_file_exclusive(
        canonical_bundle / "tables" / PAPER_SUMMARY_TABLE.name,
        PAPER_SUMMARY_TABLE,
    )
    copy_file_exclusive(
        canonical_bundle / "tables" / PAPER_PER_CLASS_TABLE.name,
        PAPER_PER_CLASS_TABLE,
    )
    copy_file_exclusive(
        canonical_bundle / "records" / COMPLETION_RECORD.name,
        COMPLETION_RECORD,
    )


def commit_outputs(staged_bundle: Path) -> str:
    RAW_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RAW_OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Canonical result bundle already exists: {RAW_OUTPUT_DIR}"
        )

    # One same-filesystem rename makes the complete scientific result canonical.
    os.rename(staged_bundle, RAW_OUTPUT_DIR)
    artifact_manifest_hash = verify_artifact_manifest(RAW_OUTPUT_DIR)
    publish_derived_outputs(RAW_OUTPUT_DIR)
    return artifact_manifest_hash


def run_final_test(
    manifest: dict,
    freeze_manifest_hash: str,
    dataset: WaferMapDataset,
):
    device = select_device()
    print(f"正式测试设备：{device}", flush=True)
    print(
        f"固定测试集：{len(dataset)} 张；共 {len(EXPECTED_SEEDS)} 个 seed",
        flush=True,
    )

    temp_root = Path(
        tempfile.mkdtemp(prefix=f"{EVALUATION_ID}_", dir=PROJECT_ROOT / "tmp")
    )
    staged_bundle = temp_root / EVALUATION_ID
    staged_metrics = staged_bundle
    staged_predictions = staged_bundle / "predictions"
    staged_figures = staged_bundle / "figures"
    staged_tables = staged_bundle / "tables"
    staged_records = staged_bundle / "records"
    for directory in (
        staged_bundle,
        staged_predictions,
        staged_figures,
        staged_tables,
        staged_records,
    ):
        directory.mkdir()

    per_seed_rows = []
    all_per_class_rows = []
    normalized_matrices = []

    committed = False
    try:
        for checkpoint_info in manifest["checkpoints"]:
            seed = int(checkpoint_info["seed"])
            print(f"开始 seed {seed} 的一次性测试推理...", flush=True)
            model, checkpoint = load_and_validate_checkpoint(
                checkpoint_info,
                manifest,
                device,
            )
            results = collect_predictions(model, dataset, device)
            summary, per_class_rows, matrix = calculate_metrics(
                results,
                checkpoint_info,
                checkpoint,
                device,
            )
            summary["checkpoint_sha256"] = checkpoint_info["sha256"]
            per_seed_rows.append(summary)
            all_per_class_rows.extend(per_class_rows)

            matrix_frame = pd.DataFrame(
                matrix,
                index=WM811K_CLASS_NAMES,
                columns=WM811K_CLASS_NAMES,
            )
            matrix_frame.to_csv(
                staged_metrics / f"confusion_matrix_seed{seed}.csv",
                index_label="true_class",
            )
            normalized = normalized_confusion_matrix(matrix)
            normalized_matrices.append(normalized)
            pd.DataFrame(
                normalized,
                index=WM811K_CLASS_NAMES,
                columns=WM811K_CLASS_NAMES,
            ).to_csv(
                staged_metrics / f"confusion_matrix_normalized_seed{seed}.csv",
                index_label="true_class",
            )
            save_predictions(
                staged_predictions / f"predictions_seed{seed}.csv",
                dataset,
                results,
            )
            write_seed_audit_record(summary, checkpoint_info)

            print(
                f"seed {seed} 完成：Accuracy={summary['accuracy'] * 100:.4f}% "
                f"Macro-F1={summary['macro_f1'] * 100:.4f}% "
                f"Balanced Accuracy={summary['balanced_accuracy'] * 100:.4f}%",
                flush=True,
            )

            del model, checkpoint, results
            if device.type == "mps":
                torch.mps.empty_cache()

        per_seed = pd.DataFrame(per_seed_rows)
        per_class = pd.DataFrame(all_per_class_rows)
        aggregate, aggregate_json = aggregate_rows(per_seed)
        per_class_aggregate = aggregate_per_class(per_class)

        per_seed.to_csv(staged_metrics / "test_per_seed_summary.csv", index=False)
        aggregate.to_csv(staged_metrics / "test_aggregate_summary.csv", index=False)
        (staged_metrics / "test_aggregate_summary.json").write_text(
            json.dumps(aggregate_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        per_class.to_csv(staged_metrics / "test_per_class_by_seed.csv", index=False)
        per_class_aggregate.to_csv(
            staged_metrics / "test_per_class_aggregate.csv", index=False
        )

        matrix_stack = np.stack(normalized_matrices, axis=0)
        mean_matrix = matrix_stack.mean(axis=0)
        std_matrix = matrix_stack.std(axis=0, ddof=1)
        for name, matrix in (
            ("confusion_matrix_normalized_mean.csv", mean_matrix),
            ("confusion_matrix_normalized_sample_std.csv", std_matrix),
        ):
            pd.DataFrame(
                matrix,
                index=WM811K_CLASS_NAMES,
                columns=WM811K_CLASS_NAMES,
            ).to_csv(staged_metrics / name, index_label="true_class")

        save_confusion_figure(
            mean_matrix,
            staged_figures / FIGURE_PNG.name,
            staged_figures / FIGURE_PDF.name,
        )
        build_paper_summary_table(per_seed, aggregate).to_csv(
            staged_tables / PAPER_SUMMARY_TABLE.name, index=False
        )
        build_paper_per_class_table(per_class_aggregate).to_csv(
            staged_tables / PAPER_PER_CLASS_TABLE.name, index=False
        )
        (staged_records / COMPLETION_RECORD.name).write_text(
            completion_markdown(per_seed, aggregate, device),
            encoding="utf-8",
        )
        write_run_manifest(
            staged_metrics / "run_manifest.json",
            freeze_manifest_hash,
            device,
            per_seed,
        )
        write_artifact_manifest(staged_bundle)
        artifact_manifest_hash = commit_outputs(staged_bundle)
        write_completion_audit_record(aggregate, artifact_manifest_hash)
        committed = True
    except BaseException as error:
        try:
            write_interruption_audit_record(error, temp_root)
        except Exception as audit_error:
            print(
                f"警告：写入中断审计记录失败：{audit_error}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"正式测试未正常完成；临时文件保留在：{temp_root}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if committed:
            shutil.rmtree(temp_root, ignore_errors=True)

    aggregate_index = aggregate.set_index("metric")
    print("三随机种子最终测试全部完成。", flush=True)
    print(
        "最终 Macro-F1："
        f"{aggregate_index.loc['macro_f1', 'mean'] * 100:.4f}% +/- "
        f"{aggregate_index.loc['macro_f1', 'sample_std'] * 100:.4f}%",
        flush=True,
    )
    print(f"原始指标：{RAW_OUTPUT_DIR}", flush=True)
    print(f"逐样本预测：{PREDICTIONS_DIR}", flush=True)


def validate_frozen_inputs() -> tuple[dict, str, WaferMapDataset]:
    manifest, manifest_hash = load_and_validate_manifest()
    validate_frozen_code_and_environment(manifest)
    dataset = validate_dataset(manifest)
    validate_all_checkpoints(manifest)
    return manifest, manifest_hash, dataset


def print_preflight_summary(dataset: WaferMapDataset):
    print("冻结协议预检查通过")
    print(f"评估编号：{EVALUATION_ID}")
    print("模型：HighRes ShuffleNetV2 + CE（无 ECA）")
    print(f"seeds：{list(EXPECTED_SEEDS)}")
    print(f"测试样本：{len(dataset)}")
    print("输出路径均为空，不会覆盖已有结果。")


def main():
    args = parse_args()

    if args.preflight:
        ensure_outputs_are_available(check_lock=True)
        _, _, dataset = validate_frozen_inputs()
        print_preflight_summary(dataset)
        print("预检查模式：没有执行测试推理，也没有创建结果文件。")
        return

    acquire_lock()
    try:
        # The lock is acquired before the second output check to close the
        # check-then-run concurrency window.
        ensure_outputs_are_available(check_lock=False)
        manifest, manifest_hash, dataset = validate_frozen_inputs()
        print_preflight_summary(dataset)
        create_irreversible_start_record(manifest_hash)
        run_final_test(manifest, manifest_hash, dataset)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
