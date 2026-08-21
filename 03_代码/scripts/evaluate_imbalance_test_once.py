#!/usr/bin/env python3
"""Run the frozen one-time test comparison for three imbalance strategies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.dataset import WaferMapDataset
from wafermap.paths import EXPERIMENT_DIR, RESULT_DIR


EVALUATION_ID = "20260807_imbalance_test_evaluation"
PROTOCOL_FILE = (
    PROJECT_ROOT / "00_项目管理" / "20260807_类别不平衡一次性测试评估协议.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "08d6a6c3cb49d59ef1d0728430d6c0914c4461030e3f54daa87fed34ac91be5d"
)
VALIDATION_CONCLUSION = (
    PROJECT_ROOT / "00_项目管理" / "20260807_类别不平衡验证结论冻结清单.json"
)
EXPECTED_VALIDATION_CONCLUSION_SHA256 = (
    "5a71d5c50afbc60a2f2ec4bf553cebc378cf5d03b224b555b7a4c11f7b32f53b"
)
ARCHITECTURE_FREEZE = (
    PROJECT_ROOT / "00_项目管理" / "20260805_stem_2x2_架构选择冻结清单.json"
)
EXPECTED_ARCHITECTURE_FREEZE_SHA256 = (
    "26e9f93dfdae615fbdf91da612cb877637cb13b18e6572b934b9d3690d705ead"
)
VALIDATION_EVALUATOR = (
    PROJECT_ROOT / "03_代码" / "scripts" / "evaluate_imbalance_validation.py"
)
EXPECTED_VALIDATION_EVALUATOR_SHA256 = (
    "d3525c4c8aa9d0dfd881cf3de65eb93aa98df99f12ebd96d24ef5fc0584520ab"
)
PROCESSED_IMAGES = (
    PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64" / "images_uint8.npy"
)
METADATA_FILE = (
    PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64" / "metadata.csv"
)
EXPECTED_IMAGES_SHA256 = (
    "ec496dd8a5cdb2f83c54a048497d1f6d77e64ae74efd495c36456629c9cf9aa3"
)
EXPECTED_METADATA_SHA256 = (
    "b6b1e0c2d819c60aa879e50015408161324616cbe5d928f4001d6609a7c9e681"
)
OLD_TEST_RUN_MANIFEST = (
    EXPERIMENT_DIR
    / "metrics"
    / "20260729_highres_ce_multiseed_final_test"
    / "run_manifest.json"
)
EXPECTED_OLD_TEST_RUN_MANIFEST_SHA256 = (
    "42918affb27aa0dd72926682af781b8d1e06ac4df421ea4d8ba9676e2b035d9f"
)
OLD_TEST_SUMMARY = (
    EXPERIMENT_DIR
    / "metrics"
    / "20260729_highres_ce_multiseed_final_test"
    / "test_per_seed_summary.csv"
)
EXPECTED_OLD_TEST_SUMMARY_SHA256 = (
    "4a1669cac01e0c3e7be4c2d5ef0ff8a856228767ecc063b98b5a3177e1ab930c"
)

OUTPUT_DIR = EXPERIMENT_DIR / "metrics" / EVALUATION_ID
PAPER_TABLE_DIR = RESULT_DIR / "tables"
PAPER_FIGURE_DIR = RESULT_DIR / "figures" / "model_results"
PAPER_PER_SEED_TABLE = PAPER_TABLE_DIR / "table_imbalance_test_per_seed.csv"
PAPER_STRATEGY_TABLE = PAPER_TABLE_DIR / "table_imbalance_test_summary.csv"
PAPER_PER_CLASS_TABLE = PAPER_TABLE_DIR / "table_imbalance_test_per_class.csv"
PAPER_DELTA_TABLE = PAPER_TABLE_DIR / "table_imbalance_test_delta_vs_ordinary_ce.csv"
PAPER_ERROR_FLOW_TABLE = PAPER_TABLE_DIR / "table_imbalance_test_error_flows.csv"
PAPER_FIGURE_PNG = PAPER_FIGURE_DIR / "imbalance_test_confusion_matrices.png"
PAPER_FIGURE_PDF = PAPER_FIGURE_DIR / "imbalance_test_confusion_matrices.pdf"
AUDIT_RECORD = (
    PROJECT_ROOT / "00_项目管理" / "20260807_类别不平衡一次性测试评估运行审计.json"
)
LOCK_FILE = PROJECT_ROOT / "tmp" / f"{EVALUATION_ID}.lock"

BATCH_SIZE = 256
SEEDS = (42, 123, 2026)
EXPECTED_TEST_SAMPLES = 25943
EXPECTED_TEST_COUNTS = np.asarray(
    [644, 83, 778, 1452, 539, 21, 130, 179, 22117], dtype=np.int64
)
AGGREGATE_METRICS = (
    "test_loss",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
)

REUSED_ORDINARY_PREDICTIONS = {
    42: {
        "path": "05_结果/predictions/20260729_highres_ce_multiseed_final_test/predictions_seed42.csv",
        "sha256": "da5e5f43339e538efd40797de3e2cd6abbacb7c5fec07cac956d72aa32cdfc21",
    },
    123: {
        "path": "05_结果/predictions/20260729_highres_ce_multiseed_final_test/predictions_seed123.csv",
        "sha256": "ac79d156b6aabd54a33c9447c2b8dba94b6815447c6320b46672ba6722357dac",
    },
    2026: {
        "path": "05_结果/predictions/20260729_highres_ce_multiseed_final_test/predictions_seed2026.csv",
        "sha256": "1242267517014cf7ca42bddea74e04b8a2e20db207d34b26610a3d0785cfe40d",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen one-time WM-811K imbalance test comparison."
    )
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


def require_file_hash(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def load_validation_evaluator():
    require_file_hash(
        VALIDATION_EVALUATOR,
        EXPECTED_VALIDATION_EVALUATOR_SHA256,
        "validation evaluator",
    )
    spec = importlib.util.spec_from_file_location(
        "frozen_imbalance_validation_evaluator", VALIDATION_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator: {VALIDATION_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


def output_paths():
    return (
        OUTPUT_DIR,
        PAPER_PER_SEED_TABLE,
        PAPER_STRATEGY_TABLE,
        PAPER_PER_CLASS_TABLE,
        PAPER_DELTA_TABLE,
        PAPER_ERROR_FLOW_TABLE,
        PAPER_FIGURE_PNG,
        PAPER_FIGURE_PDF,
        AUDIT_RECORD,
    )


def ensure_outputs_absent():
    existing = [path for path in output_paths() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to repeat or overwrite imbalance test outputs:\n"
            + "\n".join(f"- {path}" for path in existing)
        )
    if LOCK_FILE.exists():
        raise FileExistsError(f"Evaluation lock exists: {LOCK_FILE}")


def validate_frozen_inputs(validation_module):
    require_file_hash(PROTOCOL_FILE, EXPECTED_PROTOCOL_SHA256, "test protocol")
    require_file_hash(
        VALIDATION_CONCLUSION,
        EXPECTED_VALIDATION_CONCLUSION_SHA256,
        "validation conclusion",
    )
    require_file_hash(
        ARCHITECTURE_FREEZE,
        EXPECTED_ARCHITECTURE_FREEZE_SHA256,
        "architecture freeze",
    )
    require_file_hash(PROCESSED_IMAGES, EXPECTED_IMAGES_SHA256, "processed images")
    require_file_hash(METADATA_FILE, EXPECTED_METADATA_SHA256, "metadata")
    require_file_hash(
        OLD_TEST_RUN_MANIFEST,
        EXPECTED_OLD_TEST_RUN_MANIFEST_SHA256,
        "ordinary CE test manifest",
    )
    require_file_hash(
        OLD_TEST_SUMMARY,
        EXPECTED_OLD_TEST_SUMMARY_SHA256,
        "ordinary CE test summary",
    )
    for item in REUSED_ORDINARY_PREDICTIONS.values():
        require_file_hash(
            PROJECT_ROOT / item["path"], item["sha256"], "ordinary CE predictions"
        )

    conclusion = json.loads(VALIDATION_CONCLUSION.read_text(encoding="utf-8"))
    if conclusion.get("status") != "frozen_after_validation_only_evaluation":
        raise ValueError("Validation conclusion is not frozen.")
    if conclusion["frozen_decision"].get("retained_training_strategy") != "ordinary_ce":
        raise ValueError("Unexpected frozen training-strategy decision.")

    entries = list(validation_module.INPUTS)
    if len(entries) != 9:
        raise ValueError("Expected nine frozen checkpoint entries.")
    for entry in entries:
        validation_module.validate_history(entry)
        validation_module.validate_run_manifest(entry)
        model, _ = validation_module.load_and_validate_checkpoint(entry)
        del model
    return entries


def build_test_dataset() -> WaferMapDataset:
    dataset = WaferMapDataset("test")
    if dataset.transform is not None:
        raise ValueError("Test dataset must not use augmentation.")
    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError("Test sample count changed.")
    counts = np.bincount(dataset.labels, minlength=NUM_CLASSES)
    if not np.array_equal(counts, EXPECTED_TEST_COUNTS):
        raise ValueError("Test class counts changed.")
    return dataset


def validate_reused_prediction_frame(path: Path, dataset) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "source_index",
        "split",
        "true_label_id",
        "predicted_label_id",
        "confidence",
    }
    probability_columns = [
        f"prob_{class_id}_{class_name.lower().replace('-', '_')}"
        for class_id, class_name in enumerate(WM811K_CLASS_NAMES)
    ]
    if not required.issubset(frame.columns) or not set(probability_columns).issubset(
        frame.columns
    ):
        raise ValueError(f"Frozen prediction columns changed: {path}")
    if len(frame) != len(dataset):
        raise ValueError(f"Frozen prediction row count changed: {path}")
    if not np.array_equal(
        frame["source_index"].to_numpy(), dataset.metadata["source_index"].to_numpy()
    ):
        raise ValueError(f"Frozen prediction order changed: {path}")
    if not np.array_equal(
        frame["true_label_id"].to_numpy(dtype=np.int64), dataset.labels
    ):
        raise ValueError(f"Frozen true labels changed: {path}")
    if not (frame["split"].astype(str) == "test").all():
        raise ValueError(f"Frozen predictions are not test rows: {path}")
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError(f"Frozen probabilities are invalid: {path}")
    return frame


def infer(model, dataset, device: torch.device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model = model.to(device).eval()
    total_loss = 0.0
    probabilities = []
    labels = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, target).item())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(target.cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    probability_array = np.concatenate(probabilities).astype(np.float32)
    label_array = np.concatenate(labels).astype(np.int64)
    if not np.array_equal(label_array, dataset.labels):
        raise RuntimeError("Test sample order changed during inference.")
    if probability_array.shape != (EXPECTED_TEST_SAMPLES, NUM_CLASSES):
        raise RuntimeError("Unexpected test probability shape.")
    return probability_array, total_loss / len(label_array), elapsed


def create_prediction_frame(dataset, probabilities: np.ndarray) -> pd.DataFrame:
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


def metric_rows(entry, frame, test_loss, evaluation_action, elapsed):
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
        "strategy_id": entry["strategy_id"],
        "strategy": entry["strategy"],
        "seed": entry["seed"],
        "run_name": entry["run_name"],
        "best_epoch": entry["best_epoch"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "evaluation_action": evaluation_action,
        "test_samples": len(y_true),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "elapsed_seconds": None if elapsed is None else float(elapsed),
    }
    class_rows = []
    for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
        true_mask = y_true == class_id
        predicted_mask = y_pred == class_id
        class_rows.append(
            {
                "strategy_id": entry["strategy_id"],
                "strategy": entry["strategy"],
                "seed": entry["seed"],
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1_score": float(class_f1[class_id]),
                "support": int(support[class_id]),
                "predicted_count": int(predicted_mask.sum()),
                "true_positive": int((true_mask & predicted_mask).sum()),
                "false_positive": int((~true_mask & predicted_mask).sum()),
                "false_negative": int((true_mask & ~predicted_mask).sum()),
                "high_uncertainty": class_name == "Near-full",
            }
        )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return summary, class_rows, matrix, y_true, y_pred


def aggregate_strategy_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy_id, strategy), group in per_seed.groupby(
        ["strategy_id", "strategy"], sort=False
    ):
        if set(group["seed"]) != set(SEEDS):
            raise ValueError(f"Missing seed for {strategy_id}.")
        row = {
            "strategy_id": strategy_id,
            "strategy": strategy,
            "runs": len(group),
            "test_samples_per_seed": int(group["test_samples"].iloc[0]),
        }
        for metric in AGGREGATE_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def write_confusion_figure(mean_matrices: dict, destination: Path):
    order = ("ordinary_ce", "weighted_ce", "balanced_sampler")
    titles = {
        "ordinary_ce": "Ordinary CE",
        "weighted_ce": "Weighted CE",
        "balanced_sampler": "Balanced sampler",
    }
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for axis, strategy_id in zip(axes, order):
        matrix = mean_matrices[strategy_id]
        normalized = np.divide(
            matrix,
            EXPECTED_TEST_COUNTS[:, None],
            out=np.zeros_like(matrix, dtype=np.float64),
            where=EXPECTED_TEST_COUNTS[:, None] != 0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(titles[strategy_id])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(
            np.arange(NUM_CLASSES), WM811K_CLASS_NAMES, rotation=50, ha="right"
        )
        axis.set_yticks(np.arange(NUM_CLASSES), WM811K_CLASS_NAMES)
        for row_index in range(NUM_CLASSES):
            for column_index in range(NUM_CLASSES):
                value = normalized[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.50 else "black",
                    fontsize=7,
                )
        fig.colorbar(
            image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized count"
        )
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_audit(status: str, **extra):
    payload = {
        "evaluation_id": EVALUATION_ID,
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
        "frozen_decision": "ordinary_ce",
        "post_test_tuning_allowed": False,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "evaluator_sha256": sha256_file(Path(__file__)),
        **extra,
    }
    AUDIT_RECORD.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_manifests(staging: Path, device, actions, durations):
    artifact_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in {"run_manifest.json", "artifact_manifest.json"}
    }
    run_manifest = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "frozen_decision_before_test": "ordinary_ce",
        "post_test_tuning_allowed": False,
        "reused_prediction_runs": 3,
        "new_inference_runs": 6,
        "test_samples": EXPECTED_TEST_SAMPLES,
        "device": str(device),
        "actions": actions,
        "durations_seconds": durations,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifact_sha256": artifact_hashes,
    }
    (staging / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    full_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    artifact_manifest = {
        "evaluation_id": EVALUATION_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "artifact_sha256": full_hashes,
    }
    (staging / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def publish_paper_artifacts():
    pairs = (
        (OUTPUT_DIR / "per_seed_metrics.csv", PAPER_PER_SEED_TABLE),
        (OUTPUT_DIR / "strategy_summary.csv", PAPER_STRATEGY_TABLE),
        (OUTPUT_DIR / "per_class_strategy_summary.csv", PAPER_PER_CLASS_TABLE),
        (OUTPUT_DIR / "per_class_delta_vs_ordinary_ce.csv", PAPER_DELTA_TABLE),
        (OUTPUT_DIR / "error_flow_summary.csv", PAPER_ERROR_FLOW_TABLE),
        (OUTPUT_DIR / "mean_confusion_matrices.png", PAPER_FIGURE_PNG),
        (OUTPUT_DIR / "mean_confusion_matrices.pdf", PAPER_FIGURE_PDF),
    )
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run_once(validation_module, entries):
    device = select_device()
    dataset = build_test_dataset()
    old_summary = pd.read_csv(OLD_TEST_SUMMARY).set_index("seed")
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    write_audit("started", device=str(device))
    staging = Path(
        tempfile.mkdtemp(prefix=f"{EVALUATION_ID}_", dir=PROJECT_ROOT / "tmp")
    )
    predictions_dir = staging / "predictions"
    predictions_dir.mkdir(parents=True)
    summaries = []
    class_rows = []
    error_flow_rows = []
    matrices = {}
    mean_matrices = {
        strategy: []
        for strategy in ("ordinary_ce", "weighted_ce", "balanced_sampler")
    }
    actions = {}
    durations = {}

    try:
        for entry in entries:
            key = f"{entry['strategy_id']}_seed{entry['seed']}"
            if entry["strategy_id"] == "ordinary_ce":
                source_info = REUSED_ORDINARY_PREDICTIONS[entry["seed"]]
                source_path = PROJECT_ROOT / source_info["path"]
                frame = validate_reused_prediction_frame(source_path, dataset)
                test_loss = float(old_summary.loc[entry["seed"], "test_loss"])
                elapsed = None
                action = "reuse_frozen_predictions"
                shutil.copy2(source_path, predictions_dir / f"{key}.csv")
            else:
                model, _ = validation_module.load_and_validate_checkpoint(entry)
                probabilities, test_loss, elapsed = infer(model, dataset, device)
                frame = create_prediction_frame(dataset, probabilities)
                action = "infer_once"
                frame.to_csv(predictions_dir / f"{key}.csv", index=False)
                del model, probabilities
                if device.type == "mps":
                    torch.mps.empty_cache()

            summary, rows, matrix, y_true, y_pred = metric_rows(
                entry, frame, test_loss, action, elapsed
            )
            if action == "reuse_frozen_predictions":
                source_row = old_summary.loc[entry["seed"]]
                for metric in (
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                    "balanced_accuracy",
                ):
                    if not np.isclose(summary[metric], source_row[metric], atol=1e-12):
                        raise RuntimeError(f"Frozen metric mismatch: {key} {metric}")
            summaries.append(summary)
            class_rows.extend(rows)
            error_flow_rows.append(validation_module.error_flow_row(entry, y_true, y_pred))
            matrices[key] = matrix.tolist()
            mean_matrices[entry["strategy_id"]].append(matrix.astype(np.float64))
            actions[key] = action
            durations[key] = elapsed
            print(f"{action.upper()} {key}", flush=True)

        per_seed = pd.DataFrame(summaries)
        per_class = pd.DataFrame(class_rows)
        error_flows = pd.DataFrame(error_flow_rows)
        if len(per_seed) != 9 or len(per_class) != 81 or len(error_flows) != 9:
            raise RuntimeError("Unexpected output row count.")
        strategy_summary = aggregate_strategy_metrics(per_seed)
        per_class_summary = validation_module.aggregate_per_class(per_class)
        delta_table = validation_module.delta_vs_ordinary_ce(per_class_summary)
        error_flow_summary = validation_module.aggregate_error_flows(error_flows)
        mean_matrices = {
            strategy: np.mean(np.stack(strategy_matrices), axis=0)
            for strategy, strategy_matrices in mean_matrices.items()
        }

        per_seed.to_csv(staging / "per_seed_metrics.csv", index=False, float_format="%.10f")
        strategy_summary.to_csv(
            staging / "strategy_summary.csv", index=False, float_format="%.10f"
        )
        per_class.to_csv(staging / "per_class_metrics.csv", index=False, float_format="%.10f")
        per_class_summary.to_csv(
            staging / "per_class_strategy_summary.csv", index=False, float_format="%.10f"
        )
        delta_table.to_csv(
            staging / "per_class_delta_vs_ordinary_ce.csv",
            index=False,
            float_format="%.10f",
        )
        error_flows.to_csv(
            staging / "error_flows_per_seed.csv", index=False, float_format="%.10f"
        )
        error_flow_summary.to_csv(
            staging / "error_flow_summary.csv", index=False, float_format="%.10f"
        )
        (staging / "confusion_matrices.json").write_text(
            json.dumps(matrices, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_confusion_figure(mean_matrices, staging / "mean_confusion_matrices.png")
        write_confusion_figure(mean_matrices, staging / "mean_confusion_matrices.pdf")
        create_manifests(staging, device, actions, durations)

        shutil.move(str(staging), str(OUTPUT_DIR))
        publish_paper_artifacts()
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
        print("One-time imbalance test evaluation completed.")
        print(strategy_summary.to_string(index=False))
    except Exception as error:
        write_audit("failed", device=str(device), error=repr(error))
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        LOCK_FILE.unlink(missing_ok=True)


def main():
    args = parse_args()
    ensure_outputs_absent()
    validation_module = load_validation_evaluator()
    entries = validate_frozen_inputs(validation_module)
    if args.preflight:
        print("One-time imbalance test preflight: PASS")
        print("reused_predictions=3 new_inference_runs=6 test_inference_started=false")
        return
    run_once(validation_module, entries)


if __name__ == "__main__":
    main()
