#!/usr/bin/env python3
"""Evaluate nine frozen S1N checkpoints on the WM-811K validation split only."""

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

from wafermap.constants import IMAGE_SIZE, NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.dataset import WaferMapDataset
from wafermap.models_improved import ShuffleNetV2HighRes
from wafermap.paths import EXPERIMENT_DIR, RESULT_DIR


EVALUATION_ID = "20260807_imbalance_validation_evaluation"
PROTOCOL_FILE = (
    PROJECT_ROOT / "00_项目管理" / "20260807_类别不平衡验证集评估协议.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "bcb7375c932249a9feb23d44e3ae8b9bb192ab4a24d1fc5460821900d514cc33"
)
TRAINING_PROTOCOL_FILE = (
    PROJECT_ROOT / "00_项目管理" / "20260805_类别不平衡消融冻结协议.md"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "d48028dda2f3ef7c5bd284e67053121ca972363513402130001cf6b3ad0e594f"
)
ARCHITECTURE_FREEZE_FILE = (
    PROJECT_ROOT / "00_项目管理" / "20260805_stem_2x2_架构选择冻结清单.json"
)
EXPECTED_ARCHITECTURE_FREEZE_SHA256 = (
    "26e9f93dfdae615fbdf91da612cb877637cb13b18e6572b934b9d3690d705ead"
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

OUTPUT_DIR = EXPERIMENT_DIR / "metrics" / EVALUATION_ID
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PAPER_TABLE_DIR = RESULT_DIR / "tables"
PAPER_FIGURE_DIR = RESULT_DIR / "figures" / "model_results"
PAPER_PER_SEED_TABLE = PAPER_TABLE_DIR / "table_imbalance_validation_per_seed.csv"
PAPER_STRATEGY_TABLE = PAPER_TABLE_DIR / "table_imbalance_validation_summary.csv"
PAPER_PER_CLASS_TABLE = PAPER_TABLE_DIR / "table_imbalance_validation_per_class.csv"
PAPER_DELTA_TABLE = PAPER_TABLE_DIR / "table_imbalance_validation_delta_vs_ordinary_ce.csv"
PAPER_ERROR_FLOW_TABLE = PAPER_TABLE_DIR / "table_imbalance_validation_error_flows.csv"
PAPER_FIGURE_PNG = PAPER_FIGURE_DIR / "imbalance_validation_confusion_matrices.png"
PAPER_FIGURE_PDF = PAPER_FIGURE_DIR / "imbalance_validation_confusion_matrices.pdf"
AUDIT_RECORD = (
    PROJECT_ROOT / "00_项目管理" / "20260807_类别不平衡验证集评估运行审计.json"
)
LOCK_FILE = PROJECT_ROOT / "tmp" / f"{EVALUATION_ID}.lock"

BATCH_SIZE = 256
EXPECTED_VALIDATION_SAMPLES = 25938
EXPECTED_VALIDATION_COUNTS = np.asarray(
    [644, 83, 778, 1452, 539, 23, 130, 179, 22110], dtype=np.int64
)
SEEDS = (42, 123, 2026)
METRICS = (
    "validation_loss",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "elapsed_seconds",
    "samples_per_second",
)

INPUTS = (
    {
        "strategy_id": "ordinary_ce",
        "strategy": "Ordinary CE",
        "seed": 42,
        "run_name": "shufflenet_v2_highres_ce_full",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_ce_full/best.pt",
        "checkpoint_sha256": "cafebb6a76ef566ea05fa47e7745677c35cb3abbec87b1de468c0898b02c67bc",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_ce_full_history.csv",
        "history_sha256": "bca243d438d883a4302b36ae64ea637a2a0c54e29b0b5053daf018ab58002143",
        "best_epoch": 28,
        "best_val_macro_f1": 0.9053773248524237,
        "loss_name": "CrossEntropyLoss",
        "run_manifest_path": None,
        "run_manifest_sha256": None,
    },
    {
        "strategy_id": "ordinary_ce",
        "strategy": "Ordinary CE",
        "seed": 123,
        "run_name": "shufflenet_v2_highres_ce_full_seed123",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_ce_full_seed123/best.pt",
        "checkpoint_sha256": "189785ae00a95ee3c342b5f29e5b8306bf864256016259b19e80be333226929f",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_ce_full_seed123_history.csv",
        "history_sha256": "731ab1dea144e6175665755d593945aacf24c0703c311414d4e4c5a406bdfc13",
        "best_epoch": 27,
        "best_val_macro_f1": 0.9072335751434787,
        "loss_name": "CrossEntropyLoss",
        "run_manifest_path": None,
        "run_manifest_sha256": None,
    },
    {
        "strategy_id": "ordinary_ce",
        "strategy": "Ordinary CE",
        "seed": 2026,
        "run_name": "shufflenet_v2_highres_ce_full_seed2026",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_ce_full_seed2026/best.pt",
        "checkpoint_sha256": "15791aaddb863abc45f1de55c0c1e8781420238ae7b97c0b6a84740d155c0c7b",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_ce_full_seed2026_history.csv",
        "history_sha256": "f9d7bec573df034438cac3a59713c972d3016eb9bd7c999fadd68eca170332e8",
        "best_epoch": 26,
        "best_val_macro_f1": 0.9020327874707613,
        "loss_name": "CrossEntropyLoss",
        "run_manifest_path": None,
        "run_manifest_sha256": None,
    },
    {
        "strategy_id": "weighted_ce",
        "strategy": "Weighted CE",
        "seed": 42,
        "run_name": "shufflenet_v2_highres_weighted_ce_full",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_weighted_ce_full/best.pt",
        "checkpoint_sha256": "c964f651952a49f6f6a757cc960ce27e3e4a272b6c541133187d32c1359bdfda",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_weighted_ce_full_history.csv",
        "history_sha256": "8a780748cdc59ed151d0d6ea1c59092a001735933bd6bae0eaf5df63cce949ef",
        "best_epoch": 30,
        "best_val_macro_f1": 0.8216054223304172,
        "loss_name": "WeightedCrossEntropyLoss",
        "run_manifest_path": "04_实验/checkpoints/shufflenet_v2_highres_weighted_ce_full/run_manifest.json",
        "run_manifest_sha256": "b0aee3c69930a13b8ff24cd2554bdb50493e6337b59027efb4573adaabecb131",
    },
    {
        "strategy_id": "weighted_ce",
        "strategy": "Weighted CE",
        "seed": 123,
        "run_name": "shufflenet_v2_highres_weighted_ce_full_seed123",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_weighted_ce_full_seed123/best.pt",
        "checkpoint_sha256": "28ae79372c7361a71e272b5d5ae823df5ac012b36ba1f3e45a1abee480e38699",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_weighted_ce_full_seed123_history.csv",
        "history_sha256": "e1bac1aec0c0228ad402147cfc162e36ee1087a842a8bd31a86174bc5be4da6d",
        "best_epoch": 26,
        "best_val_macro_f1": 0.8255535085668846,
        "loss_name": "WeightedCrossEntropyLoss",
        "run_manifest_path": "04_实验/checkpoints/shufflenet_v2_highres_weighted_ce_full_seed123/run_manifest.json",
        "run_manifest_sha256": "61c2c92d66d57cc707aeaac234ea8e6f3531de47a343d95934f21733bd34cb74",
    },
    {
        "strategy_id": "weighted_ce",
        "strategy": "Weighted CE",
        "seed": 2026,
        "run_name": "shufflenet_v2_highres_weighted_ce_full_seed2026",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_weighted_ce_full_seed2026/best.pt",
        "checkpoint_sha256": "2b7627f6892a5b2c7f948530da24f71ced74862f8daf67d1d0e23d9e6ed53744",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_weighted_ce_full_seed2026_history.csv",
        "history_sha256": "13bb9cb57dae78fd65fe7dbe99dfb41a92bd852c599b63f4fe6f4bd6c0f3d7b5",
        "best_epoch": 30,
        "best_val_macro_f1": 0.8287354700722174,
        "loss_name": "WeightedCrossEntropyLoss",
        "run_manifest_path": "04_实验/checkpoints/shufflenet_v2_highres_weighted_ce_full_seed2026/run_manifest.json",
        "run_manifest_sha256": "ef16690040c0eb69cc0c964d834a2f58ca51ae6f91cd60ae7aa2ea793c420056",
    },
    {
        "strategy_id": "balanced_sampler",
        "strategy": "Balanced sampler",
        "seed": 42,
        "run_name": "shufflenet_v2_highres_balanced_sampler_full",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_balanced_sampler_full/best.pt",
        "checkpoint_sha256": "43bdbf3102823f4a70714d6621349702eeb779160f7841d24b8cdc6f2220db93",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_balanced_sampler_full_history.csv",
        "history_sha256": "754952057006f67e7b14e0ffdc2538f49aade504decdcae3f235184a2e5df086",
        "best_epoch": 22,
        "best_val_macro_f1": 0.8505548881374216,
        "loss_name": "CrossEntropyLoss",
        "run_manifest_path": "04_实验/checkpoints/shufflenet_v2_highres_balanced_sampler_full/run_manifest.json",
        "run_manifest_sha256": "024fdfa3dff571777db6ae06339fc10f8c38cb2dd7c6f6a986f376c9df253b35",
    },
    {
        "strategy_id": "balanced_sampler",
        "strategy": "Balanced sampler",
        "seed": 123,
        "run_name": "shufflenet_v2_highres_balanced_sampler_full_seed123",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_balanced_sampler_full_seed123/best.pt",
        "checkpoint_sha256": "6deafd3ec7c0e0647c2ad33677c46dd940b5366fb16a83ee1f0d78b12877b971",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_balanced_sampler_full_seed123_history.csv",
        "history_sha256": "0ebcb02c0fbd5cc0e98f3f44b275e4931baedbca06379e4d0e6d326d8d998036",
        "best_epoch": 21,
        "best_val_macro_f1": 0.8322596265761477,
        "loss_name": "CrossEntropyLoss",
        "run_manifest_path": "04_实验/checkpoints/shufflenet_v2_highres_balanced_sampler_full_seed123/run_manifest.json",
        "run_manifest_sha256": "92fd0ec7efba2cb4c0e09fca902ed737ce31d91664ceacc75766b31b8a46bf67",
    },
    {
        "strategy_id": "balanced_sampler",
        "strategy": "Balanced sampler",
        "seed": 2026,
        "run_name": "shufflenet_v2_highres_balanced_sampler_full_seed2026",
        "checkpoint_path": "04_实验/checkpoints/shufflenet_v2_highres_balanced_sampler_full_seed2026/best.pt",
        "checkpoint_sha256": "6988d556634bb57b49f7c76e4758553598fa9dc8c2840b143b24870d34b636e5",
        "history_path": "04_实验/metrics/shufflenet_v2_highres_balanced_sampler_full_seed2026_history.csv",
        "history_sha256": "2e3d3516b5d219cad3395565af98a6b1eff0b76eccdf40393d577774c6201ccb",
        "best_epoch": 30,
        "best_val_macro_f1": 0.8279704244531133,
        "loss_name": "CrossEntropyLoss",
        "run_manifest_path": "04_实验/checkpoints/shufflenet_v2_highres_balanced_sampler_full_seed2026/run_manifest.json",
        "run_manifest_sha256": "fdca53f29f210fce6456366f7c2a53b8931e94afd5cc0ed1486e2e1e2c07786e",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen WM-811K imbalance checkpoints on validation only."
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
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


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


def ensure_output_paths_are_available():
    existing = [path for path in output_paths() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite validation-evaluation artifacts:\n"
            + "\n".join(f"- {path}" for path in existing)
        )
    if LOCK_FILE.exists():
        raise FileExistsError(f"Evaluation lock exists: {LOCK_FILE}")


def validate_protocol_and_data():
    require_file_hash(PROTOCOL_FILE, EXPECTED_PROTOCOL_SHA256, "evaluation protocol")
    require_file_hash(
        TRAINING_PROTOCOL_FILE,
        EXPECTED_TRAINING_PROTOCOL_SHA256,
        "training protocol",
    )
    require_file_hash(
        ARCHITECTURE_FREEZE_FILE,
        EXPECTED_ARCHITECTURE_FREEZE_SHA256,
        "architecture freeze",
    )
    require_file_hash(PROCESSED_IMAGES, EXPECTED_IMAGES_SHA256, "processed images")
    require_file_hash(METADATA_FILE, EXPECTED_METADATA_SHA256, "metadata")


def validate_history(entry: dict):
    history_path = PROJECT_ROOT / entry["history_path"]
    require_file_hash(history_path, entry["history_sha256"], "training history")
    history = pd.read_csv(history_path)
    expected_columns = {
        "epoch",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
    }
    if not expected_columns.issubset(history.columns):
        raise ValueError(f"Unexpected history columns: {history_path}")
    if tuple(history["epoch"].astype(int)) != tuple(range(1, 31)):
        raise ValueError(f"History is not a full 30-epoch run: {history_path}")
    best_index = history["val_macro_f1"].astype(float).idxmax()
    best_row = history.loc[best_index]
    if int(best_row["epoch"]) != entry["best_epoch"]:
        raise ValueError(f"Best epoch mismatch: {history_path}")
    if not np.isclose(
        float(best_row["val_macro_f1"]), entry["best_val_macro_f1"], atol=1e-12
    ):
        raise ValueError(f"Best validation Macro-F1 mismatch: {history_path}")


def validate_run_manifest(entry: dict):
    manifest_path_string = entry["run_manifest_path"]
    if manifest_path_string is None:
        return
    manifest_path = PROJECT_ROOT / manifest_path_string
    require_file_hash(
        manifest_path,
        entry["run_manifest_sha256"],
        "imbalance run manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": "20260805_class_imbalance_ablation",
        "strategy": entry["strategy_id"],
        "run_name": entry["run_name"],
        "seed": entry["seed"],
        "selected_architecture": "S1N",
        "architecture_freeze_sha256": EXPECTED_ARCHITECTURE_FREEZE_SHA256,
        "protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Run manifest mismatch for {key}: {manifest_path}")
    artifact_hashes = manifest.get("artifact_sha256", {})
    expected_artifact_paths = {
        entry["checkpoint_path"],
        entry["history_path"],
    }
    if len(artifact_hashes) != 4 or not expected_artifact_paths.issubset(artifact_hashes):
        raise ValueError(f"Incomplete artifact list: {manifest_path}")
    for relative_path, expected_hash in artifact_hashes.items():
        require_file_hash(PROJECT_ROOT / relative_path, expected_hash, "run artifact")
    for relative_path, expected_hash in (
        (entry["checkpoint_path"], entry["checkpoint_sha256"]),
        (entry["history_path"], entry["history_sha256"]),
    ):
        if artifact_hashes.get(relative_path) != expected_hash:
            raise ValueError(f"Run manifest artifact mismatch: {manifest_path}")


def load_and_validate_checkpoint(entry: dict):
    checkpoint_path = PROJECT_ROOT / entry["checkpoint_path"]
    require_file_hash(checkpoint_path, entry["checkpoint_sha256"], "checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "run_name": entry["run_name"],
        "model_name": "ShuffleNetV2HighRes",
        "image_size": IMAGE_SIZE,
        "parameter_count": 1262397,
        "epoch": entry["best_epoch"],
        "random_seed": entry["seed"],
        "loss_name": entry["loss_name"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint mismatch for {key}: {checkpoint_path}")
    if checkpoint.get("class_names") != list(WM811K_CLASS_NAMES):
        raise ValueError(f"Class order mismatch: {checkpoint_path}")
    if not np.isclose(
        float(checkpoint.get("best_val_macro_f1")),
        entry["best_val_macro_f1"],
        atol=1e-12,
    ):
        raise ValueError(f"Checkpoint Macro-F1 mismatch: {checkpoint_path}")
    if entry["strategy_id"] == "ordinary_ce":
        if checkpoint.get("sampling_strategy") is not None:
            raise ValueError(f"Unexpected sampling strategy: {checkpoint_path}")
    else:
        if checkpoint.get("selected_architecture_id") != "S1N":
            raise ValueError(f"Architecture ID mismatch: {checkpoint_path}")
        if checkpoint.get("sampling_strategy") != entry["strategy_id"]:
            raise ValueError(f"Sampling strategy mismatch: {checkpoint_path}")
        if checkpoint.get("architecture_freeze_sha256") != EXPECTED_ARCHITECTURE_FREEZE_SHA256:
            raise ValueError(f"Architecture hash mismatch: {checkpoint_path}")
    model = ShuffleNetV2HighRes()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 1262397:
        raise ValueError(f"Parameter count mismatch after loading: {checkpoint_path}")
    return model, checkpoint


def validate_inputs():
    if len(INPUTS) != 9:
        raise ValueError("Expected exactly nine frozen checkpoints.")
    combinations = {(item["strategy_id"], item["seed"]) for item in INPUTS}
    expected_combinations = {
        (strategy, seed)
        for strategy in ("ordinary_ce", "weighted_ce", "balanced_sampler")
        for seed in SEEDS
    }
    if combinations != expected_combinations:
        raise ValueError("Unexpected strategy/seed combinations.")
    for entry in INPUTS:
        validate_history(entry)
        validate_run_manifest(entry)
        model, _ = load_and_validate_checkpoint(entry)
        del model


def build_validation_dataset() -> WaferMapDataset:
    dataset = WaferMapDataset("val")
    if dataset.transform is not None:
        raise ValueError("Validation dataset must not use random augmentation.")
    if len(dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("Validation sample count changed.")
    counts = np.bincount(dataset.labels, minlength=NUM_CLASSES)
    if not np.array_equal(counts, EXPECTED_VALIDATION_COUNTS):
        raise ValueError("Validation class counts changed.")
    return dataset


def infer(model, dataset: WaferMapDataset, device: torch.device):
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
    predictions = []
    confidences = []
    labels = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, target).item())
            probabilities = torch.softmax(logits, dim=1)
            predictions.append(probabilities.argmax(dim=1).cpu().numpy())
            confidences.append(probabilities.max(dim=1).values.cpu().numpy())
            labels.append(target.cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    y_true = np.concatenate(labels).astype(np.int64)
    y_pred = np.concatenate(predictions).astype(np.int64)
    confidence = np.concatenate(confidences).astype(np.float32)
    if not np.array_equal(y_true, dataset.labels):
        raise RuntimeError("Validation sample order changed during inference.")
    if y_pred.shape != (EXPECTED_VALIDATION_SAMPLES,):
        raise RuntimeError("Unexpected validation prediction shape.")
    return y_true, y_pred, confidence, total_loss / len(y_true), elapsed


def prediction_frame(dataset, y_true, y_pred, confidence) -> pd.DataFrame:
    columns = ["source_index", "lotName", "waferIndex", "label", "label_id"]
    frame = dataset.metadata[columns].copy()
    frame = frame.rename(columns={"label_id": "true_label_id", "label": "true_label"})
    frame["predicted_label_id"] = y_pred
    frame["predicted_label"] = [WM811K_CLASS_NAMES[index] for index in y_pred]
    frame["confidence"] = confidence
    frame["correct"] = y_pred == y_true
    return frame


def metric_rows(entry: dict, y_true, y_pred, validation_loss, elapsed):
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
        "validation_samples": len(y_true),
        "validation_loss": float(validation_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(len(y_true) / elapsed),
    }
    per_class = []
    for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
        per_class.append(
            {
                "strategy_id": entry["strategy_id"],
                "strategy": entry["strategy"],
                "seed": entry["seed"],
                "run_name": entry["run_name"],
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1_score": float(class_f1[class_id]),
                "support": int(support[class_id]),
                "predicted_count": int((y_pred == class_id).sum()),
                "true_positive": int(
                    ((y_true == class_id) & (y_pred == class_id)).sum()
                ),
                "false_positive": int(
                    ((y_true != class_id) & (y_pred == class_id)).sum()
                ),
                "false_negative": int(
                    ((y_true == class_id) & (y_pred != class_id)).sum()
                ),
                "high_uncertainty": class_name == "Near-full",
            }
        )
    return summary, per_class, confusion_matrix(y_true, y_pred, labels=labels)


def aggregate_strategy_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy_id, strategy), group in per_seed.groupby(
        ["strategy_id", "strategy"], sort=False
    ):
        if set(group["seed"]) != set(SEEDS):
            raise ValueError(f"Missing seed in {strategy_id}.")
        row = {
            "strategy_id": strategy_id,
            "strategy": strategy,
            "training_runs": len(group),
            "validation_samples_per_seed": int(group["validation_samples"].iloc[0]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def aggregate_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in per_class.groupby(
        ["strategy_id", "strategy", "class_id", "class_name"], sort=False
    ):
        strategy_id, strategy, class_id, class_name = keys
        row = {
            "strategy_id": strategy_id,
            "strategy": strategy,
            "class_id": int(class_id),
            "class_name": class_name,
            "seeds": len(group),
            "support_per_seed": int(group["support"].iloc[0]),
            "high_uncertainty": bool(group["high_uncertainty"].iloc[0]),
        }
        for metric in (
            "precision",
            "recall",
            "f1_score",
            "predicted_count",
            "true_positive",
            "false_positive",
            "false_negative",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        row["f1_score_minimum"] = float(group["f1_score"].min())
        row["f1_score_maximum"] = float(group["f1_score"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["strategy_id", "class_id"]
    ).reset_index(drop=True)


def delta_vs_ordinary_ce(per_class_summary: pd.DataFrame) -> pd.DataFrame:
    ordinary = per_class_summary.loc[
        per_class_summary["strategy_id"] == "ordinary_ce"
    ].set_index("class_id")
    rows = []
    for _, row in per_class_summary.iterrows():
        if row["strategy_id"] == "ordinary_ce":
            continue
        baseline = ordinary.loc[row["class_id"]]
        rows.append(
            {
                "strategy_id": row["strategy_id"],
                "strategy": row["strategy"],
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "support_per_seed": int(row["support_per_seed"]),
                "high_uncertainty": bool(row["high_uncertainty"]),
                "delta_precision_mean": float(
                    row["precision_mean"] - baseline["precision_mean"]
                ),
                "delta_recall_mean": float(row["recall_mean"] - baseline["recall_mean"]),
                "delta_f1_mean": float(row["f1_score_mean"] - baseline["f1_score_mean"]),
                "delta_predicted_count_mean": float(
                    row["predicted_count_mean"] - baseline["predicted_count_mean"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy_id", "class_id"]).reset_index(
        drop=True
    )


def error_flow_row(entry: dict, y_true, y_pred) -> dict:
    none_class_id = WM811K_CLASS_NAMES.index("none")
    none_mask = y_true == none_class_id
    defect_mask = ~none_mask
    none_to_defect = int((none_mask & (y_pred != none_class_id)).sum())
    defect_to_none = int((defect_mask & (y_pred == none_class_id)).sum())
    predicted_defect = int((y_pred != none_class_id).sum())
    return {
        "strategy_id": entry["strategy_id"],
        "strategy": entry["strategy"],
        "seed": entry["seed"],
        "none_support": int(none_mask.sum()),
        "defect_support": int(defect_mask.sum()),
        "none_to_defect_count": none_to_defect,
        "none_to_defect_rate": float(none_to_defect / none_mask.sum()),
        "defect_to_none_count": defect_to_none,
        "defect_to_none_rate": float(defect_to_none / defect_mask.sum()),
        "predicted_defect_count": predicted_defect,
        "predicted_defect_to_true_defect_ratio": float(
            predicted_defect / defect_mask.sum()
        ),
    }


def aggregate_error_flows(error_flows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy_id, strategy), group in error_flows.groupby(
        ["strategy_id", "strategy"], sort=False
    ):
        row = {
            "strategy_id": strategy_id,
            "strategy": strategy,
            "seeds": len(group),
            "none_support_per_seed": int(group["none_support"].iloc[0]),
            "defect_support_per_seed": int(group["defect_support"].iloc[0]),
        }
        for metric in (
            "none_to_defect_count",
            "none_to_defect_rate",
            "defect_to_none_count",
            "defect_to_none_rate",
            "predicted_defect_count",
            "predicted_defect_to_true_defect_ratio",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy_id").reset_index(drop=True)


def write_confusion_figure(mean_matrices: dict, destination: Path):
    strategy_order = ("ordinary_ce", "weighted_ce", "balanced_sampler")
    titles = {
        "ordinary_ce": "Ordinary CE",
        "weighted_ce": "Weighted CE",
        "balanced_sampler": "Balanced sampler",
    }
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for axis, strategy_id in zip(axes, strategy_order):
        matrix = np.asarray(mean_matrices[strategy_id], dtype=np.float64)
        normalized = np.divide(
            matrix,
            EXPECTED_VALIDATION_COUNTS[:, None],
            out=np.zeros_like(matrix),
            where=EXPECTED_VALIDATION_COUNTS[:, None] != 0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(titles[strategy_id])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(np.arange(NUM_CLASSES), WM811K_CLASS_NAMES, rotation=50, ha="right")
        axis.set_yticks(np.arange(NUM_CLASSES), WM811K_CLASS_NAMES)
        for row_index in range(NUM_CLASSES):
            for column_index in range(NUM_CLASSES):
                value = normalized[row_index, column_index]
                color = "white" if value > 0.50 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized count")
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_audit(status: str, **extra):
    record = {
        "evaluation_id": EVALUATION_ID,
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
        "selection_split": "validation",
        "test_split_accessed": False,
        "evaluation_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "architecture_freeze_sha256": EXPECTED_ARCHITECTURE_FREEZE_SHA256,
        "evaluator_sha256": sha256_file(Path(__file__)),
        **extra,
    }
    AUDIT_RECORD.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_run_manifest(staging: Path, device: torch.device, durations: dict):
    artifacts = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    inputs = [
        {
            "strategy_id": entry["strategy_id"],
            "seed": entry["seed"],
            "checkpoint_path": entry["checkpoint_path"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "history_path": entry["history_path"],
            "history_sha256": entry["history_sha256"],
            "run_manifest_path": entry["run_manifest_path"],
            "run_manifest_sha256": entry["run_manifest_sha256"],
        }
        for entry in INPUTS
    ]
    manifest = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "selection_split": "validation",
        "test_split_accessed": False,
        "checkpoint_count": len(INPUTS),
        "validation_samples": EXPECTED_VALIDATION_SAMPLES,
        "device": str(device),
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
        "inputs": inputs,
        "artifact_sha256": artifacts,
    }
    (staging / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_artifact_manifest(staging: Path):
    artifact_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    payload = {
        "evaluation_id": EVALUATION_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "artifact_sha256": artifact_hashes,
    }
    (staging / "artifact_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def publish_paper_artifacts():
    artifacts = (
        (OUTPUT_DIR / "per_seed_metrics.csv", PAPER_PER_SEED_TABLE),
        (OUTPUT_DIR / "strategy_summary.csv", PAPER_STRATEGY_TABLE),
        (OUTPUT_DIR / "per_class_strategy_summary.csv", PAPER_PER_CLASS_TABLE),
        (OUTPUT_DIR / "per_class_delta_vs_ordinary_ce.csv", PAPER_DELTA_TABLE),
        (OUTPUT_DIR / "error_flow_summary.csv", PAPER_ERROR_FLOW_TABLE),
        (OUTPUT_DIR / "mean_confusion_matrices.png", PAPER_FIGURE_PNG),
        (OUTPUT_DIR / "mean_confusion_matrices.pdf", PAPER_FIGURE_PDF),
    )
    for source, target in artifacts:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run_once(dataset: WaferMapDataset):
    device = select_device()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    write_audit("started", device=str(device))
    (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{EVALUATION_ID}_", dir=PROJECT_ROOT / "tmp"))
    prediction_directory = staging / "predictions"
    prediction_directory.mkdir(parents=True)
    summaries = []
    class_rows = []
    error_flow_rows = []
    matrices = {}
    mean_matrices = {strategy: [] for strategy in ("ordinary_ce", "weighted_ce", "balanced_sampler")}
    durations = {}

    try:
        for entry in INPUTS:
            key = f"{entry['strategy_id']}_seed{entry['seed']}"
            model, checkpoint = load_and_validate_checkpoint(entry)
            y_true, y_pred, confidence, validation_loss, elapsed = infer(model, dataset, device)
            summary, rows, matrix = metric_rows(
                entry, y_true, y_pred, validation_loss, elapsed
            )
            recorded = checkpoint["val_metrics"]
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                difference = summary[metric] - float(recorded[metric])
                summary[f"recorded_{metric}"] = float(recorded[metric])
                summary[f"recomputed_minus_recorded_{metric}"] = float(difference)
                if not np.isclose(summary[metric], float(recorded[metric]), atol=1e-6):
                    raise RuntimeError(
                        f"Validation reproduction mismatch for {key} {metric}: "
                        f"{summary[metric]} != {recorded[metric]}"
                    )
            prediction_frame(dataset, y_true, y_pred, confidence).to_csv(
                prediction_directory / f"{key}.csv", index=False
            )
            summaries.append(summary)
            class_rows.extend(rows)
            error_flow_rows.append(error_flow_row(entry, y_true, y_pred))
            matrices[key] = matrix.tolist()
            mean_matrices[entry["strategy_id"]].append(matrix.astype(np.float64))
            durations[key] = elapsed
            print(f"VALIDATION {key}: {elapsed:.2f}s", flush=True)
            del model, checkpoint, y_true, y_pred, confidence
            if device.type == "mps":
                torch.mps.empty_cache()

        per_seed = pd.DataFrame(summaries)
        per_class = pd.DataFrame(class_rows)
        error_flows = pd.DataFrame(error_flow_rows)
        if len(per_seed) != 9 or len(per_class) != 81 or len(error_flows) != 9:
            raise RuntimeError("Unexpected number of metric rows.")
        strategy_summary = aggregate_strategy_metrics(per_seed)
        per_class_summary = aggregate_per_class(per_class)
        delta_table = delta_vs_ordinary_ce(per_class_summary)
        error_flow_summary = aggregate_error_flows(error_flows)
        mean_matrices = {
            strategy: np.mean(np.stack(matrices), axis=0)
            for strategy, matrices in mean_matrices.items()
        }
        mean_matrix_rows = []
        for strategy_id, matrix in mean_matrices.items():
            for true_class in range(NUM_CLASSES):
                for predicted_class in range(NUM_CLASSES):
                    mean_matrix_rows.append(
                        {
                            "strategy_id": strategy_id,
                            "true_class_id": true_class,
                            "true_class": WM811K_CLASS_NAMES[true_class],
                            "predicted_class_id": predicted_class,
                            "predicted_class": WM811K_CLASS_NAMES[predicted_class],
                            "mean_count_per_seed": float(matrix[true_class, predicted_class]),
                            "row_normalized_mean_count": float(
                                matrix[true_class, predicted_class]
                                / EXPECTED_VALIDATION_COUNTS[true_class]
                            ),
                        }
                    )
        per_seed.to_csv(staging / "per_seed_metrics.csv", index=False, float_format="%.10f")
        strategy_summary.to_csv(
            staging / "strategy_summary.csv", index=False, float_format="%.10f"
        )
        per_class.to_csv(staging / "per_class_metrics.csv", index=False, float_format="%.10f")
        per_class_summary.to_csv(
            staging / "per_class_strategy_summary.csv", index=False, float_format="%.10f"
        )
        delta_table.to_csv(
            staging / "per_class_delta_vs_ordinary_ce.csv", index=False, float_format="%.10f"
        )
        error_flows.to_csv(
            staging / "error_flows_per_seed.csv", index=False, float_format="%.10f"
        )
        error_flow_summary.to_csv(
            staging / "error_flow_summary.csv", index=False, float_format="%.10f"
        )
        pd.DataFrame(mean_matrix_rows).to_csv(
            staging / "mean_confusion_matrices.csv", index=False, float_format="%.10f"
        )
        (staging / "confusion_matrices.json").write_text(
            json.dumps(matrices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_confusion_figure(mean_matrices, staging / "mean_confusion_matrices.png")
        write_confusion_figure(mean_matrices, staging / "mean_confusion_matrices.pdf")
        create_run_manifest(staging, device, durations)
        create_artifact_manifest(staging)

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
        print("Validation-only imbalance evaluation completed.")
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
    ensure_output_paths_are_available()
    validate_protocol_and_data()
    validate_inputs()
    dataset = build_validation_dataset()
    if args.preflight:
        print("Validation-only imbalance evaluation preflight: PASS")
        print("split=val checkpoints=9 strategies=3 test_split_accessed=false")
        return
    run_once(dataset)


if __name__ == "__main__":
    main()
