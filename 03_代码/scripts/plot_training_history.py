#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORY_FILE = (
    PROJECT_ROOT
    / "04_实验"
    / "metrics"
    / "resnet18_baseline_full_history.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "05_结果"
    / "figures"
    / "model_results"
)

OUTPUT_FILE = (
    FIGURE_DIR
    / "resnet18_baseline_full_training_curves.png"
)

REQUIRED_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "train_macro_f1",
    "val_macro_f1",
    "train_balanced_accuracy",
    "val_balanced_accuracy",
]


def main():
    if not HISTORY_FILE.is_file():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_FILE}"
        )

    history = pd.read_csv(HISTORY_FILE)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in history.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    history = (
        history
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    if len(history) < 2:
        raise ValueError(
            "At least two epochs are required for plotting."
        )

    epochs = history["epoch"]

    best_index = history["val_macro_f1"].idxmax()
    best_epoch = int(history.loc[best_index, "epoch"])
    best_macro_f1 = float(
        history.loc[best_index, "val_macro_f1"]
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    axes[0, 0].plot(
        epochs,
        history["train_loss"],
        label="Train",
        linewidth=2,
    )
    axes[0, 0].plot(
        epochs,
        history["val_loss"],
        label="Validation",
        linewidth=2,
    )
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(
        epochs,
        history["train_accuracy"],
        label="Train",
        linewidth=2,
    )
    axes[0, 1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation",
        linewidth=2,
    )
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(
        epochs,
        history["train_macro_f1"],
        label="Train",
        linewidth=2,
    )
    axes[1, 0].plot(
        epochs,
        history["val_macro_f1"],
        label="Validation",
        linewidth=2,
    )
    axes[1, 0].scatter(
        best_epoch,
        best_macro_f1,
        color="black",
        zorder=3,
        label=f"Best epoch: {best_epoch}",
    )
    axes[1, 0].set_title("Macro-F1")
    axes[1, 0].set_ylabel("Macro-F1")
    axes[1, 0].legend()

    axes[1, 1].plot(
        epochs,
        history["train_balanced_accuracy"],
        label="Train",
        linewidth=2,
    )
    axes[1, 1].plot(
        epochs,
        history["val_balanced_accuracy"],
        label="Validation",
        linewidth=2,
    )
    axes[1, 1].set_title("Balanced Accuracy")
    axes[1, 1].set_ylabel("Balanced Accuracy")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(
            True,
            alpha=0.3,
        )

    figure.suptitle(
        "ResNet18 Baseline Training Curves",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Training history: {HISTORY_FILE}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print(f"Figure saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()