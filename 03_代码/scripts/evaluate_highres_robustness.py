#!/usr/bin/env python3
"""Evaluate frozen HighRes checkpoints under deterministic test perturbations."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.models_improved import ShuffleNetV2HighRes
from wafermap.paths import PROCESSED_DATA_DIR
from wafermap.robustness import LEVEL_NAMES, ROBUSTNESS_LEVELS, apply_corruption


EVALUATION_ID = "20260801_highres_robustness"
OUTPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / EVALUATION_ID
RESULT_TABLE = PROJECT_ROOT / "05_结果" / "tables" / "table_highres_robustness.csv"
FIGURE_PATH = PROJECT_ROOT / "05_结果" / "figures" / "model_results" / "highres_robustness_curves.png"
FREEZE_MANIFEST = PROJECT_ROOT / "00_项目管理" / "20260801_五模型统一评估冻结清单.json"
EXPECTED_MANIFEST_SHA256 = "2a97b07434239589da586aaa5b836d3b1aaba8d957c40cc8f5d5e54fd1139d5f"
BATCH_SIZE = 256
SEEDS = (42, 123, 2026)

IMAGE_FILE = PROCESSED_DATA_DIR / "wm811k_labeled_64x64" / "images_uint8.npy"
METADATA_FILE = PROCESSED_DATA_DIR / "wm811k_labeled_64x64" / "metadata.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RobustWaferDataset(Dataset):
    def __init__(self, images, metadata, positions, corruption=None, level_index=None):
        self.images = images
        self.metadata = metadata.iloc[positions].reset_index(drop=True)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.corruption = corruption
        self.level_index = level_index
        self.labels = self.metadata["label_id"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        source_index = int(self.metadata.iloc[index]["source_index"])
        image = self.images[int(self.positions[index])].copy()
        if self.corruption is not None:
            image = apply_corruption(
                image, source_index, self.corruption, int(self.level_index)
            )
        tensor = torch.from_numpy(image).to(torch.float32).unsqueeze(0) / 2.0
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def load_manifest():
    if sha256_file(FREEZE_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Unified freeze manifest hash mismatch.")
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    highres = next(item for item in manifest["models"] if item["model_id"] == "highres_shufflenet_v2")
    return highres


def load_data():
    images = np.load(IMAGE_FILE, mmap_mode="r")
    metadata = pd.read_csv(METADATA_FILE)
    mask = metadata["split"].eq("test").to_numpy()
    positions = np.flatnonzero(mask)
    if len(positions) != 25943:
        raise ValueError("Unexpected test sample count.")
    return images, metadata, positions


def select_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    predictions = []
    labels = []
    synchronize(device)
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            predictions.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
            labels.append(targets.numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "elapsed_seconds": elapsed,
    }


def main():
    if OUTPUT_DIR.exists() or RESULT_TABLE.exists() or FIGURE_PATH.exists():
        raise FileExistsError("Robustness outputs already exist; refusing to overwrite.")
    highres = load_manifest()
    images, metadata, positions = load_data()
    device = select_device()
    rows = []
    for checkpoint_info in highres["checkpoints"]:
        seed = int(checkpoint_info["seed"])
        checkpoint_file = PROJECT_ROOT / checkpoint_info["checkpoint_path"]
        if sha256_file(checkpoint_file) != checkpoint_info["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model = ShuffleNetV2HighRes(num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        clean_dataset = RobustWaferDataset(images, metadata, positions)
        clean_metrics = evaluate(model, clean_dataset, device)
        for metric_name, metric_value in clean_metrics.items():
            if metric_name != "elapsed_seconds":
                rows.append({"seed": seed, "corruption": "clean", "level": "clean", "level_index": -1, "metric": metric_name, "value": metric_value, "delta_from_clean_pp": 0.0})
        for corruption in ROBUSTNESS_LEVELS:
            for level_index, level_name in enumerate(LEVEL_NAMES):
                dataset = RobustWaferDataset(images, metadata, positions, corruption, level_index)
                metrics = evaluate(model, dataset, device)
                for metric_name in ("accuracy", "macro_f1", "balanced_accuracy"):
                    clean_value = clean_metrics[metric_name]
                    value = metrics[metric_name]
                    rows.append({"seed": seed, "corruption": corruption, "level": level_name, "level_index": level_index, "metric": metric_name, "value": value, "delta_from_clean_pp": (value - clean_value) * 100.0})
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"seed={seed} complete", flush=True)

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True)
    result.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)
    aggregate = result.groupby(["corruption", "level", "level_index", "metric"], dropna=False).agg(value_mean=("value", "mean"), value_sample_std=("value", lambda values: values.std(ddof=1)), delta_from_clean_pp_mean=("delta_from_clean_pp", "mean"), delta_from_clean_pp_sample_std=("delta_from_clean_pp", lambda values: values.std(ddof=1))).reset_index()
    aggregate.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    RESULT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(RESULT_TABLE, index=False)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, ("accuracy", "macro_f1", "balanced_accuracy")):
        subset = aggregate[(aggregate["metric"] == metric) & (aggregate["corruption"] != "clean")]
        for corruption in ROBUSTNESS_LEVELS:
            curve = subset[subset["corruption"] == corruption].sort_values("level_index")
            axis.plot(curve["level_index"], curve["delta_from_clean_pp_mean"], marker="o", label=corruption)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks([0, 1, 2], LEVEL_NAMES)
        axis.set_ylabel("Change from clean (percentage points)")
        axis.set_title(metric)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    figure.savefig(FIGURE_PATH, dpi=300)
    figure.savefig(FIGURE_PATH.with_suffix(".pdf"))
    plt.close(figure)
    manifest = {"evaluation_id": EVALUATION_ID, "completed_at": datetime.now().astimezone().isoformat(), "device": str(device), "test_samples": len(positions), "seeds": list(SEEDS), "protocol": {"corruptions": ROBUSTNESS_LEVELS, "level_names": LEVEL_NAMES, "batch_size": BATCH_SIZE}, "source_checkpoint_hashes": {str(seed): next(item["checkpoint_sha256"] for item in highres["checkpoints"] if int(item["seed"]) == seed) for seed in SEEDS}, "script_sha256": sha256_file(Path(__file__))}
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Robustness outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
