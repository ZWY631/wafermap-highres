#!/usr/bin/env python3
"""Train ShuffleNetV2-ECA with Focal Loss on WM-811K."""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import (
    NUM_CLASSES,
    RANDOM_SEED,
    WM811K_CLASS_NAMES,
)
from wafermap.dataset import WaferMapDataset
from torch import nn
from wafermap.models_improved import ShuffleNetV2HighResECA
from wafermap.paths import EXPERIMENT_DIR
from wafermap.transforms import build_train_transform


RUN_NAME = "shufflenet_v2_highres_eca_ce_full"
MODEL_NAME = "ShuffleNetV2HighResECA"

CHECKPOINT_DIR = (
    EXPERIMENT_DIR / "checkpoints" / RUN_NAME
)
METRICS_DIR = EXPERIMENT_DIR / "metrics"
LOG_DIR = EXPERIMENT_DIR / "logs"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last.pt"
METRICS_FILE = (
    METRICS_DIR / f"{RUN_NAME}_history.csv"
)
LOG_FILE = LOG_DIR / f"{RUN_NAME}.txt"

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
NUM_EPOCHS = 30
MAX_TRAIN_BATCHES = None

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def build_dataloaders():
    train_dataset = WaferMapDataset(
        "train",
        transform=build_train_transform(),
    )

    val_dataset = WaferMapDataset("val")
    test_dataset = WaferMapDataset("test")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_index, (images, labels) in enumerate(
        data_loader
    ):
        if (
            MAX_TRAIN_BATCHES is not None
            and batch_index >= MAX_TRAIN_BATCHES
        ):
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )
        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    if not all_labels:
        raise RuntimeError(
            "No training batches were processed."
        )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    metrics = {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }

    return metrics


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(
            labels.cpu().numpy()
        )

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    metrics = {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }

    return metrics

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    val_metrics,
    best_val_macro_f1,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_metrics": val_metrics,
        "best_val_macro_f1": best_val_macro_f1,
        "class_names": list(WM811K_CLASS_NAMES),
        "random_seed": RANDOM_SEED,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(checkpoint, path)


def initialize_metrics_file():
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "train_balanced_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "epoch_seconds",
    ]

    with METRICS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()


def append_metrics(row):
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "train_balanced_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "epoch_seconds",
    ]

    with METRICS_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writerow(row)

def main():
    seed_everything(RANDOM_SEED)

    device = select_device()
    print(f"训练设备：{device}")

    train_loader, val_loader, test_loader = (
        build_dataloaders()
    )

    model = ShuffleNetV2HighResECA().to(device)
    criterion = FocalLoss(
    gamma=FOCAL_GAMMA
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    initialize_metrics_file()

    log_lines = [
        "WM-811K HighRes ShuffleNetV2-ECA Focal training",
        f"run_name={RUN_NAME}",
        f"model={MODEL_NAME}",
        f"focal_gamma={FOCAL_GAMMA}",
        f"device={device}",
        f"seed={RANDOM_SEED}",
        f"train_batch_size={TRAIN_BATCH_SIZE}",
        f"eval_batch_size={EVAL_BATCH_SIZE}",
        f"num_epochs={NUM_EPOCHS}",
        f"max_train_batches={MAX_TRAIN_BATCHES}",
        f"learning_rate={LEARNING_RATE}",
        f"weight_decay={WEIGHT_DECAY}",
        "loss=FocalLoss",
        f"focal_gamma={FOCAL_GAMMA}",
"class_weight=None",
    ]

    best_val_macro_f1 = float("-inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        learning_rate = optimizer.param_groups[0][
            "lr"
        ]

        epoch_seconds = (
            time.perf_counter() - epoch_start
        )

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics[
                "balanced_accuracy"
            ],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "epoch_seconds": epoch_seconds,
        }

        append_metrics(row)

        save_checkpoint(
            LAST_CHECKPOINT,
            epoch,
            model,
            optimizer,
            scheduler,
            val_metrics,
            best_val_macro_f1,
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics[
                "macro_f1"
            ]

            save_checkpoint(
                BEST_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scheduler,
                val_metrics,
                best_val_macro_f1,
            )

        scheduler.step()

        message = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_balanced_acc="
            f"{val_metrics['balanced_accuracy']:.4f} "
            f"seconds={epoch_seconds:.1f}"
        )

        print(message)
        log_lines.append(message)

    LOG_FILE.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    print(f"最佳模型已保存：{BEST_CHECKPOINT}")
    print(f"最后模型已保存：{LAST_CHECKPOINT}")
    print(f"训练历史已保存：{METRICS_FILE}")
    print(f"训练日志已保存：{LOG_FILE}")


if __name__ == "__main__":
    main()       