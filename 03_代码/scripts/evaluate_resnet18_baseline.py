#!/usr/bin/env python3
"""Evaluate the best ResNet18 baseline on the test set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import (
    NUM_CLASSES,
    WM811K_CLASS_NAMES,
)
from wafermap.dataset import WaferMapDataset
from wafermap.models import ResNet18Baseline
from wafermap.paths import EXPERIMENT_DIR, RESULT_DIR


RUN_NAME = "resnet18_baseline_full"

CHECKPOINT_FILE = (
    EXPERIMENT_DIR
    / "checkpoints"
    / RUN_NAME
    / "best.pt"
)

METRICS_DIR = EXPERIMENT_DIR / "metrics"
FIGURE_DIR = (
    RESULT_DIR / "figures" / "model_results"
)

SUMMARY_FILE = (
    METRICS_DIR / f"{RUN_NAME}_test_summary.json"
)
PER_CLASS_FILE = (
    METRICS_DIR / f"{RUN_NAME}_per_class.csv"
)
PREDICTIONS_FILE = (
    METRICS_DIR / f"{RUN_NAME}_predictions.csv"
)
CONFUSION_MATRIX_FILE = (
    FIGURE_DIR / f"{RUN_NAME}_confusion_matrix.png"
)

BATCH_SIZE = 256
NUM_WORKERS = 0


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_best_model(device):
    if not CHECKPOINT_FILE.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
    )

    checkpoint_classes = checkpoint[
        "class_names"
    ]

    if checkpoint_classes != list(WM811K_CLASS_NAMES):
        raise ValueError(
            "Checkpoint class order does not match constants."
        )

    model = ResNet18Baseline()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def collect_test_predictions(
    model,
    device,
):
    dataset = WaferMapDataset("test")

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
    all_predictions = []
    all_confidences = []

    synchronize(device)
    start_time = time.perf_counter()

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )
        confidences, predictions = (
            probabilities.max(dim=1)
        )

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size

        all_labels.extend(
            labels.cpu().numpy()
        )
        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_confidences.extend(
            confidences.cpu().numpy()
        )

    synchronize(device)
    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    y_true = np.asarray(
        all_labels,
        dtype=np.int64,
    )
    y_pred = np.asarray(
        all_predictions,
        dtype=np.int64,
    )
    confidences = np.asarray(
        all_confidences,
        dtype=np.float32,
    )

    if len(y_true) != len(dataset):
        raise ValueError(
            "Prediction count does not match test dataset."
        )

    results = {
        "loss": total_loss / len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
        "elapsed_seconds": elapsed_seconds,
    }

    return dataset, results

def save_metric_tables(
    dataset,
    results,
    checkpoint,
    device,
):
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    labels = np.arange(NUM_CLASSES)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(WM811K_CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    summary = {
        "run_name": RUN_NAME,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "test_samples": int(len(y_true)),
        "test_loss": float(results["loss"]),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "elapsed_seconds": float(
            results["elapsed_seconds"]
        ),
        "end_to_end_samples_per_second": float(
            len(y_true) / results["elapsed_seconds"]
        ),
    }

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    per_class_rows = []

    for class_id, class_name in enumerate(
        WM811K_CLASS_NAMES
    ):
        class_metrics = report[class_name]

        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1-score"],
                "support": int(
                    class_metrics["support"]
                ),
            }
        )

    pd.DataFrame(
        per_class_rows
    ).to_csv(
        PER_CLASS_FILE,
        index=False,
    )

    predictions = dataset.metadata.copy()

    predictions["predicted_label_id"] = y_pred
    predictions["predicted_label"] = [
        WM811K_CLASS_NAMES[class_id]
        for class_id in y_pred
    ]
    predictions["confidence"] = results[
        "confidences"
    ]
    predictions["correct"] = (
        y_true == y_pred
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    matrix_file = (
        METRICS_DIR
        / f"{RUN_NAME}_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
        index=WM811K_CLASS_NAMES,
        columns=WM811K_CLASS_NAMES,
    ).to_csv(
        matrix_file,
        index_label="true_class",
    )

    return matrix, summary

def plot_confusion_matrix(matrix):
    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_sums != 0,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        normalized_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=WM811K_CLASS_NAMES,
        yticklabels=WM811K_CLASS_NAMES,
        cbar_kws={
            "label": "Row-normalized proportion"
        },
        ax=axis,
    )

    axis.set_title(
        "ResNet18 Baseline: Test Confusion Matrix"
    )
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")

    plt.setp(
        axis.get_xticklabels(),
        rotation=40,
        ha="right",
    )
    plt.setp(
        axis.get_yticklabels(),
        rotation=0,
    )

    figure.tight_layout()
    figure.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    device = select_device()
    print(f"评估设备：{device}")

    model, checkpoint = load_best_model(
        device
    )

    print(
        f"加载最佳模型：epoch "
        f"{checkpoint['epoch']}"
    )

    dataset, results = collect_test_predictions(
        model,
        device,
    )

    matrix, summary = save_metric_tables(
        dataset,
        results,
        checkpoint,
        device,
    )

    plot_confusion_matrix(matrix)

    print(
        f"测试准确率："
        f"{summary['accuracy']:.4f}"
    )
    print(
        f"测试 macro-F1："
        f"{summary['macro_f1']:.4f}"
    )
    print(
        f"测试平衡准确率："
        f"{summary['balanced_accuracy']:.4f}"
    )
    print(
        f"端到端吞吐量："
        f"{summary['end_to_end_samples_per_second']:.1f} "
        f"samples/s"
    )
    print(f"总体指标：{SUMMARY_FILE}")
    print(f"逐类指标：{PER_CLASS_FILE}")
    print(f"预测结果：{PREDICTIONS_FILE}")
    print(f"混淆矩阵图：{CONFUSION_MATRIX_FILE}")


if __name__ == "__main__":
    main()
