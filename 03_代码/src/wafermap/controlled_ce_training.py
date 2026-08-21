from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wafermap.constants import (
    IMAGE_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
    WM811K_CLASS_NAMES,
)
from wafermap.dataset import WaferMapDataset
from wafermap.paths import EXPERIMENT_DIR
from wafermap.transforms import build_train_transform


SUPPORTED_SEEDS = (42, 123, 2026)

TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
NUM_EPOCHS = 30
MAX_TRAIN_BATCHES = None

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

METRIC_FIELDNAMES = (
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
)


@dataclass(frozen=True)
class ExperimentSpec:
    base_run_name: str
    model_name: str
    log_title: str
    model_factory: Callable[[], nn.Module]


@dataclass(frozen=True)
class RunPaths:
    checkpoint_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    metrics_file: Path
    log_file: Path


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled WM-811K Cross Entropy experiment."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SUPPORTED_SEEDS,
        default=RANDOM_SEED,
        help="Paired experiment seed (42, 123, or 2026).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate output paths and exit without training or "
            "creating files."
        ),
    )
    return parser.parse_args(argv)


def run_name_for_seed(base_run_name: str, seed: int) -> str:
    if seed == RANDOM_SEED:
        return base_run_name
    return f"{base_run_name}_seed{seed}"


def build_run_paths(
    run_name: str,
    experiment_dir: Path = EXPERIMENT_DIR,
) -> RunPaths:
    checkpoint_dir = experiment_dir / "checkpoints" / run_name
    return RunPaths(
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=checkpoint_dir / "best.pt",
        last_checkpoint=checkpoint_dir / "last.pt",
        metrics_file=(
            experiment_dir / "metrics" / f"{run_name}_history.csv"
        ),
        log_file=experiment_dir / "logs" / f"{run_name}.txt",
    )


def ensure_output_paths_are_available(paths: RunPaths):
    output_paths = (
        paths.checkpoint_dir,
        paths.metrics_file,
        paths.log_file,
    )
    existing_paths = [
        path for path in output_paths if path.exists()
    ]

    if existing_paths:
        formatted_paths = "\n".join(
            f"- {path}" for path in existing_paths
        )
        raise FileExistsError(
            "Refusing to overwrite an existing run:\n"
            f"{formatted_paths}\n"
            "Use a planned seed whose run has not already completed."
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(seed: int):
    train_dataset = WaferMapDataset(
        "train",
        transform=build_train_transform(),
    )
    val_dataset = WaferMapDataset("val")

    generator = torch.Generator()
    generator.manual_seed(seed)

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
    return train_loader, val_loader


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

    for batch_index, (images, labels) in enumerate(data_loader):
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
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    if not all_labels:
        raise RuntimeError("No training batches were processed.")

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    return calculate_metrics(total_loss, y_true, y_pred)


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
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    return calculate_metrics(total_loss, y_true, y_pred)


def calculate_metrics(total_loss, y_true, y_pred):
    return {
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


def save_checkpoint(
    path,
    run_name,
    spec,
    seed,
    epoch,
    model,
    optimizer,
    scheduler,
    val_metrics,
    best_val_macro_f1,
):
    checkpoint = {
        "run_name": run_name,
        "model_name": spec.model_name,
        "loss_name": "CrossEntropyLoss",
        "image_size": IMAGE_SIZE,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
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
        "random_seed": seed,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def initialize_metrics_file(metrics_file: Path):
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDNAMES,
        )
        writer.writeheader()


def append_metrics(metrics_file: Path, row):
    with metrics_file.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDNAMES,
        )
        writer.writerow(row)


def print_preflight(
    spec: ExperimentSpec,
    seed: int,
    run_name: str,
    paths: RunPaths,
):
    print("Controlled training preflight passed")
    print(f"run_name={run_name}")
    print(f"model={spec.model_name}")
    print(f"seed={seed}")
    print(f"checkpoint_dir={paths.checkpoint_dir}")
    print(f"metrics_file={paths.metrics_file}")
    print(f"log_file={paths.log_file}")


def run_experiment(
    spec: ExperimentSpec,
    seed: int,
    preflight: bool = False,
):
    run_name = run_name_for_seed(spec.base_run_name, seed)
    paths = build_run_paths(run_name)
    ensure_output_paths_are_available(paths)

    if preflight:
        print_preflight(spec, seed, run_name, paths)
        return

    seed_everything(seed)

    device = select_device()
    print(f"Training device: {device}")
    train_loader, val_loader = build_dataloaders(seed)

    model = spec.model_factory().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    initialize_metrics_file(paths.metrics_file)

    log_lines = [
        spec.log_title,
        f"run_name={run_name}",
        f"model={spec.model_name}",
        f"device={device}",
        f"seed={seed}",
        f"train_batch_size={TRAIN_BATCH_SIZE}",
        f"eval_batch_size={EVAL_BATCH_SIZE}",
        f"num_epochs={NUM_EPOCHS}",
        f"max_train_batches={MAX_TRAIN_BATCHES}",
        f"learning_rate={LEARNING_RATE}",
        f"weight_decay={WEIGHT_DECAY}",
        "loss=CrossEntropyLoss",
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

        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start
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
        append_metrics(paths.metrics_file, row)

        save_checkpoint(
            paths.last_checkpoint,
            run_name,
            spec,
            seed,
            epoch,
            model,
            optimizer,
            scheduler,
            val_metrics,
            best_val_macro_f1,
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                paths.best_checkpoint,
                run_name,
                spec,
                seed,
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
            "val_balanced_acc="
            f"{val_metrics['balanced_accuracy']:.4f} "
            f"seconds={epoch_seconds:.1f}"
        )
        print(message)
        log_lines.append(message)

    paths.log_file.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Best checkpoint: {paths.best_checkpoint}")
    print(f"Last checkpoint: {paths.last_checkpoint}")
    print(f"Training history: {paths.metrics_file}")
    print(f"Training log: {paths.log_file}")


def main(
    spec: ExperimentSpec,
    argv: Sequence[str] | None = None,
):
    args = parse_args(argv)
    run_experiment(
        spec=spec,
        seed=args.seed,
        preflight=args.preflight,
    )
