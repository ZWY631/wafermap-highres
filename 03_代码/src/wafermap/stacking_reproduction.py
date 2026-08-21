from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.transforms import InterpolationMode

from wafermap.constants import NUM_CLASSES


SUPPORTED_SEEDS = (42, 123, 2026)
NUM_OOF_FOLDS = 2
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MFE_BATCH_SIZE = 512
CNN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
META_RIDGE_ALPHA = 0.1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cnn_train_transform():
    return transforms.Compose(
        [
            transforms.RandomRotation(
                degrees=180,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if len(features) != len(labels):
            raise ValueError("Feature and label lengths differ.")
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class WaferBinaryDataset(Dataset):
    """Author-compatible CNN input: failed dies only, then center to +/-0.5."""

    def __init__(
        self,
        images: np.ndarray,
        positions: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        if len(positions) != len(labels):
            raise ValueError("Image-position and label lengths differ.")
        self.images = images
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = np.asarray(self.images[int(self.positions[index])])
        binary = torch.from_numpy((image == 2).astype(np.float32)).unsqueeze(0)
        if self.transform is not None:
            binary = self.transform(binary)
        return binary - 0.5, torch.tensor(self.labels[index], dtype=torch.long)


class HandcraftedFNN(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(59, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, features):
        return self.network(features)


class VGG16WaferCNN(nn.Module):
    """Author PyTorch architecture, trained from scratch for fair comparison."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        architecture = vgg16(weights=None)
        architecture.features[0] = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1
        )
        architecture.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        architecture.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, NUM_CLASSES),
        )
        self.network = architecture

    def forward(self, images):
        return self.network(images)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(NUM_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


@torch.inference_mode()
def predict_probabilities(model, data_loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    for inputs, batch_labels in data_loader:
        logits = model(inputs.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.vstack(probabilities), np.concatenate(labels)


@torch.inference_mode()
def evaluate_classifier(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    labels = []
    for inputs, batch_labels in data_loader:
        inputs = inputs.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, batch_labels)
        total_loss += float(loss.item()) * len(batch_labels)
        total_samples += len(batch_labels)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    return {
        "loss": total_loss / total_samples,
        **classification_metrics(y_true, y_pred),
    }


@dataclass
class TrainingResult:
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_metrics: dict[str, float]
    history: list[dict[str, float]]


def train_classifier(
    model,
    train_loader,
    validation_loader,
    device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> TrainingResult:
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-7
    )
    best_macro_f1 = float("-inf")
    best_epoch = 0
    best_state = None
    best_metrics = None
    history = []

    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_samples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(labels)
            train_samples += len(labels)
        validation = evaluate_classifier(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation["loss"])
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss / train_samples,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "val_balanced_accuracy": validation["balanced_accuracy"],
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if validation["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            best_metrics = copy.deepcopy(validation)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training produced no best checkpoint.")
    model.load_state_dict(best_state)
    return TrainingResult(best_state, best_epoch, best_metrics, history)


def make_lot_grouped_oof_folds(
    labels: Sequence[int],
    groups: Sequence[str],
    seed: int,
    n_splits: int = NUM_OOF_FOLDS,
):
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = []
    coverage = np.zeros(len(labels), dtype=np.int64)
    for fold_id, (train_indices, holdout_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        overlap = set(groups[train_indices]).intersection(groups[holdout_indices])
        if overlap:
            raise RuntimeError(f"Lot leakage in fold {fold_id}: {len(overlap)} lots")
        if set(labels[holdout_indices]) != set(range(NUM_CLASSES)):
            raise RuntimeError(f"Fold {fold_id} does not contain all nine classes.")
        coverage[holdout_indices] += 1
        folds.append((train_indices, holdout_indices))
    if not np.all(coverage == 1):
        raise RuntimeError("OOF coverage must equal one for every training sample.")
    return folds


@dataclass
class MetaRidgeModel:
    coefficient: np.ndarray
    intercept: np.ndarray
    alpha: float

    def decision_function(self, base_probabilities: np.ndarray) -> np.ndarray:
        return base_probabilities @ self.coefficient.T + self.intercept

    def predict(self, base_probabilities: np.ndarray) -> np.ndarray:
        return self.decision_function(base_probabilities).argmax(axis=1)

    def predict_probabilities(self, base_probabilities: np.ndarray) -> np.ndarray:
        return softmax(self.decision_function(base_probabilities), axis=1)


def fit_meta_ridge(
    base_probabilities: np.ndarray,
    labels: np.ndarray,
    alpha: float = META_RIDGE_ALPHA,
) -> MetaRidgeModel:
    base_probabilities = np.asarray(base_probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if base_probabilities.shape != (len(labels), NUM_CLASSES * 2):
        raise ValueError(
            f"Expected {(len(labels), NUM_CLASSES * 2)}, got {base_probabilities.shape}"
        )
    targets = np.eye(NUM_CLASSES, dtype=np.float64)[labels]
    estimator = Ridge(alpha=alpha, fit_intercept=True)
    estimator.fit(base_probabilities, targets)
    return MetaRidgeModel(
        coefficient=np.asarray(estimator.coef_, dtype=np.float64),
        intercept=np.asarray(estimator.intercept_, dtype=np.float64),
        alpha=float(alpha),
    )


def fit_feature_scaler(features: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.asarray(features, dtype=np.float32))
    return scaler
