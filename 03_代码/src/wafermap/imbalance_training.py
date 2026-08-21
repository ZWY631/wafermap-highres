from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Sequence

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
from torch.utils.data import DataLoader, WeightedRandomSampler

from wafermap.constants import (
    IMAGE_SIZE,
    NUM_CLASSES,
    WM811K_CLASS_NAMES,
)
from wafermap.controlled_ce_training import (
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_TRAIN_BATCHES,
    NUM_EPOCHS,
    NUM_WORKERS,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    build_run_paths,
    ensure_output_paths_are_available,
)
from wafermap.dataset import WaferMapDataset
from wafermap.models_improved import ShuffleNetV2HighRes
from wafermap.paths import EXPERIMENT_DIR, PROJECT_ROOT
from wafermap.transforms import build_train_transform


SUPPORTED_SEEDS = (42, 123, 2026)
SUPPORTED_STRATEGIES = ("weighted_ce", "balanced_sampler")

FREEZE_FILE = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260805_stem_2x2_架构选择冻结清单.json"
)
EXPECTED_FREEZE_SHA256 = (
    "26e9f93dfdae615fbdf91da612cb877637cb13b18e6572b934b9d3690d705ead"
)

BASE_RUN_NAMES = {
    "weighted_ce": "shufflenet_v2_highres_weighted_ce_full",
    "balanced_sampler": "shufflenet_v2_highres_balanced_sampler_full",
}

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


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run a frozen WM-811K imbalance strategy experiment."
    )
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_STRATEGIES,
        required=True,
        help="Imbalance strategy to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=SUPPORTED_SEEDS,
        default=42,
        help="Training seed (42, 123, or 2026).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate the frozen protocol and output paths only.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selected_architecture():
    if not FREEZE_FILE.is_file():
        raise FileNotFoundError(f"Missing architecture freeze: {FREEZE_FILE}")
    actual_hash = sha256_file(FREEZE_FILE)
    if actual_hash != EXPECTED_FREEZE_SHA256:
        raise ValueError(
            "Architecture freeze hash mismatch; do not start imbalance runs."
        )
    payload = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    if payload.get("selected_architecture_id") != "S1N":
        raise ValueError(
            "This imbalance script is frozen for S1N, but the selected "
            f"architecture is {payload.get('selected_architecture_id')!r}."
        )
    if payload.get("test_inputs_used") is not False:
        raise ValueError("Architecture selection must remain test-blind.")
    return payload, actual_hash


def run_name_for_seed(strategy: str, seed: int) -> str:
    base_name = BASE_RUN_NAMES[strategy]
    if seed == 42:
        return base_name
    return f"{base_name}_seed{seed}"


def build_strategy_run_paths(strategy: str, seed: int):
    return build_run_paths(
        run_name_for_seed(strategy, seed),
        experiment_dir=EXPERIMENT_DIR,
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calculate_class_counts(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if len(counts) != NUM_CLASSES or np.any(counts <= 0):
        raise ValueError(f"Every class must be present; counts={counts.tolist()}")
    return counts.astype(np.int64)


def inverse_frequency_weights(
    class_counts: np.ndarray,
) -> np.ndarray:
    class_counts = np.asarray(class_counts, dtype=np.float64)
    if class_counts.shape != (NUM_CLASSES,):
        raise ValueError(
            f"class_counts must have shape ({NUM_CLASSES},), "
            f"got {class_counts.shape}"
        )
    if np.any(class_counts <= 0):
        raise ValueError("Class counts must all be positive.")
    total = float(class_counts.sum())
    return total / (NUM_CLASSES * class_counts)


def build_dataloaders(strategy: str, seed: int):
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    train_dataset = WaferMapDataset(
        "train",
        transform=build_train_transform(),
    )
    val_dataset = WaferMapDataset("val")
    class_counts = calculate_class_counts(train_dataset.labels)
    class_weights = inverse_frequency_weights(class_counts)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    common_kwargs = {
        "batch_size": TRAIN_BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": False,
        "drop_last": False,
        "generator": loader_generator,
    }

    if strategy == "weighted_ce":
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs,
        )
    else:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(seed)
        sample_weights = torch.as_tensor(
            class_weights[train_dataset.labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampler_generator,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=False,
            **common_kwargs,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader, class_counts, class_weights


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
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_index, (images, labels) in enumerate(data_loader):
        if MAX_TRAIN_BATCHES is not None and batch_index >= MAX_TRAIN_BATCHES:
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
        all_predictions.extend(logits.detach().argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    if not all_labels:
        raise RuntimeError("No training batches were processed.")
    return calculate_metrics(
        total_loss,
        np.asarray(all_labels),
        np.asarray(all_predictions),
    )


@torch.inference_mode()
def evaluate(model, data_loader, criterion, device):
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
        all_predictions.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return calculate_metrics(
        total_loss,
        np.asarray(all_labels),
        np.asarray(all_predictions),
    )


def build_criteria(
    strategy: str,
    class_weights: np.ndarray,
    device: torch.device,
):
    """Return training and common unweighted validation criteria."""
    if strategy == "weighted_ce":
        train_criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )
        )
    elif strategy == "balanced_sampler":
        train_criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    validation_criterion = nn.CrossEntropyLoss()
    return train_criterion, validation_criterion


def save_checkpoint(
    path: Path,
    run_name: str,
    strategy: str,
    seed: int,
    epoch: int,
    model,
    optimizer,
    scheduler,
    val_metrics: dict,
    best_val_macro_f1: float,
    class_counts: np.ndarray,
    class_weights: np.ndarray,
    freeze_hash: str,
):
    checkpoint = {
        "run_name": run_name,
        "model_name": "ShuffleNetV2HighRes",
        "selected_architecture_id": "S1N",
        "loss_name": (
            "WeightedCrossEntropyLoss"
            if strategy == "weighted_ce"
            else "CrossEntropyLoss"
        ),
        "sampling_strategy": strategy,
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
        "train_class_counts": class_counts.tolist(),
        "inverse_frequency_class_weights": class_weights.tolist(),
        "architecture_freeze_sha256": freeze_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def initialize_metrics_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES).writeheader()


def append_metrics(path: Path, row: dict):
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=METRIC_FIELDNAMES).writerow(row)


def print_preflight(strategy: str, seed: int, paths, class_counts, weights):
    print("Imbalance training preflight passed")
    print(f"strategy={strategy}")
    print(f"seed={seed}")
    print("selected_architecture=S1N")
    print(f"train_class_counts={class_counts.tolist()}")
    print(
        "inverse_frequency_weight_range="
        f"{weights.min():.8f},{weights.max():.8f}"
    )
    print(f"checkpoint_dir={paths.checkpoint_dir}")
    print(f"metrics_file={paths.metrics_file}")
    print(f"log_file={paths.log_file}")


def run_experiment(strategy: str, seed: int, preflight: bool = False):
    freeze_payload, freeze_hash = validate_selected_architecture()
    run_name = run_name_for_seed(strategy, seed)
    paths = build_strategy_run_paths(strategy, seed)
    ensure_output_paths_are_available(paths)
    train_loader, val_loader, class_counts, class_weights = build_dataloaders(
        strategy,
        seed,
    )

    if preflight:
        print_preflight(strategy, seed, paths, class_counts, class_weights)
        return

    seed_everything(seed)
    device = select_device()
    print(f"Training device: {device}")
    model = ShuffleNetV2HighRes().to(device)
    criterion, validation_criterion = build_criteria(
        strategy,
        class_weights,
        device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    initialize_metrics_file(paths.metrics_file)

    log_lines = [
        "WM-811K HighRes ShuffleNetV2 class-imbalance training",
        f"run_name={run_name}",
        "model=ShuffleNetV2HighRes",
        "selected_architecture=S1N",
        f"strategy={strategy}",
        f"device={device}",
        f"seed={seed}",
        f"train_batch_size={TRAIN_BATCH_SIZE}",
        f"eval_batch_size={EVAL_BATCH_SIZE}",
        f"num_epochs={NUM_EPOCHS}",
        f"max_train_batches={MAX_TRAIN_BATCHES}",
        f"learning_rate={LEARNING_RATE}",
        f"weight_decay={WEIGHT_DECAY}",
        "validation_criterion=CrossEntropyLoss_unweighted",
        f"train_class_counts={class_counts.tolist()}",
        f"inverse_frequency_class_weights={class_weights.tolist()}",
        f"architecture_freeze_sha256={freeze_hash}",
        f"freeze_selected_architecture={freeze_payload['selected_architecture_id']}",
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
            validation_criterion,
            device,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "epoch_seconds": epoch_seconds,
        }
        append_metrics(paths.metrics_file, row)
        save_checkpoint(
            paths.last_checkpoint,
            run_name,
            strategy,
            seed,
            epoch,
            model,
            optimizer,
            scheduler,
            val_metrics,
            best_val_macro_f1,
            class_counts,
            class_weights,
            freeze_hash,
        )
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                paths.best_checkpoint,
                run_name,
                strategy,
                seed,
                epoch,
                model,
                optimizer,
                scheduler,
                val_metrics,
                best_val_macro_f1,
                class_counts,
                class_weights,
                freeze_hash,
            )
        scheduler.step()
        message = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_balanced_acc={val_metrics['balanced_accuracy']:.4f} "
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


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    run_experiment(args.strategy, args.seed, args.preflight)
