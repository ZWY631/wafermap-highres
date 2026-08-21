#!/usr/bin/env python3
"""Compare frozen HighRes, standard ShuffleNetV2, and ResNet18 robustness."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES
from wafermap.models import ResNet18Baseline, ShuffleNetV2Baseline
from wafermap.robustness import LEVEL_NAMES, ROBUSTNESS_LEVELS, apply_corruption


EVALUATION_ID = "20260807_three_model_robustness_comparison"
PROTOCOL_FILE = (
    PROJECT_ROOT / "00_项目管理" / "20260807_三模型鲁棒性横向对比冻结协议.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "c28e1a55f9a7e540107c5228a41e82c8f7273e0f6c9c3e497eac1678f6281726"
)
ORIGINAL_PROTOCOL = PROJECT_ROOT / "00_项目管理" / "20260801_鲁棒性实验冻结协议.md"
EXPECTED_ORIGINAL_PROTOCOL_SHA256 = (
    "b7b30f449ea5d0e75b8ad42babcccd1772936d8f4fe9c24ecbd035654f76ba9c"
)
FREEZE_MANIFEST = PROJECT_ROOT / "00_项目管理" / "20260801_五模型统一评估冻结清单.json"
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "2a97b07434239589da586aaa5b836d3b1aaba8d957c40cc8f5d5e54fd1139d5f"
)
UNIFIED_PER_SEED = (
    PROJECT_ROOT / "04_实验" / "metrics" / "20260801_five_model_multiseed_unified_test" / "per_seed_metrics.csv"
)
EXPECTED_UNIFIED_PER_SEED_SHA256 = (
    "6411f0f83dde805cf9b9b105f8a8d94a654bb52524d99b6def13f1274c28c9fc"
)
HIGHRES_ROBUSTNESS_DIR = (
    PROJECT_ROOT / "04_实验" / "metrics" / "20260801_highres_robustness"
)
HIGHRES_PER_SEED = HIGHRES_ROBUSTNESS_DIR / "per_seed_metrics.csv"
HIGHRES_AGGREGATE = HIGHRES_ROBUSTNESS_DIR / "aggregate_metrics.csv"
HIGHRES_RUN_MANIFEST = HIGHRES_ROBUSTNESS_DIR / "run_manifest.json"
EXPECTED_HIGHRES_PER_SEED_SHA256 = (
    "c5a2ff758e14fbc621846cde9690d293c3462b2b840a458c3a612bd8eaa2c282"
)
EXPECTED_HIGHRES_AGGREGATE_SHA256 = (
    "11f6cec296846944b13ecc9e47f45f53fae5053b226551d65b60a7d40267f52d"
)
EXPECTED_HIGHRES_RUN_MANIFEST_SHA256 = (
    "471b371a6f3cc2e6313bde2fc375978f0eb541ee4877acb7f637f97b7a22aa0d"
)

FROZEN_CODE = {
    "03_代码/scripts/evaluate_highres_robustness.py": "d637c2aead6e45b44586f3ed0e2f24c5e05f1267bbb482469c849fe7083a63c2",
    "03_代码/src/wafermap/robustness.py": "d8b492439e90c9b55224537588619365a06041d34f5d40e08f2f5fbb687a8a67",
    "03_代码/src/wafermap/models.py": "01d1af8ebaa12b5b390742a9e56bd81902b55ba2653a4f80f1f586b63c22ba35",
    "03_代码/src/wafermap/models_improved.py": "9f42de30839646d2ae7c5b82190d97b29dd6fb54cd47f8c574c4bfd1fadde392",
}
PROCESSED_DIR = PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64"
IMAGE_FILE = PROCESSED_DIR / "images_uint8.npy"
METADATA_FILE = PROCESSED_DIR / "metadata.csv"
EXPECTED_IMAGE_SHA256 = (
    "ec496dd8a5cdb2f83c54a048497d1f6d77e64ae74efd495c36456629c9cf9aa3"
)
EXPECTED_METADATA_SHA256 = (
    "b6b1e0c2d819c60aa879e50015408161324616cbe5d928f4001d6609a7c9e681"
)

SEEDS = (42, 123, 2026)
BASELINE_MODEL_IDS = ("standard_shufflenet_v2", "resnet18")
MODEL_IDS = ("highres_shufflenet_v2",) + BASELINE_MODEL_IDS
MODEL_DISPLAY_NAMES = {
    "highres_shufflenet_v2": "HighRes ShuffleNetV2",
    "standard_shufflenet_v2": "Standard ShuffleNetV2",
    "resnet18": "ResNet18",
}
MODEL_FACTORIES = {
    "standard_shufflenet_v2": ShuffleNetV2Baseline,
    "resnet18": ResNet18Baseline,
}
METRICS = ("accuracy", "macro_f1", "balanced_accuracy")
EXPECTED_TEST_SAMPLES = 25943
EXPECTED_TEST_COUNTS = np.asarray([644, 83, 778, 1452, 539, 21, 130, 179, 22117])
BATCH_SIZE = 256

OUTPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / EVALUATION_ID
TABLE_DIR = PROJECT_ROOT / "05_结果" / "tables"
FIGURE_DIR = PROJECT_ROOT / "05_结果" / "figures" / "model_results"
PUBLISHED = {
    "aggregate_metrics.csv": TABLE_DIR / "table_three_model_robustness.csv",
    "comparison_vs_highres.csv": TABLE_DIR / "table_three_model_robustness_vs_highres.csv",
    "descriptive_summary.csv": TABLE_DIR / "table_three_model_robustness_summary.csv",
    "three_model_robustness_curves.png": FIGURE_DIR / "three_model_robustness_curves.png",
    "three_model_robustness_curves.pdf": FIGURE_DIR / "three_model_robustness_curves.pdf",
    "three_model_robustness_summary.png": FIGURE_DIR / "three_model_robustness_summary.png",
    "three_model_robustness_summary.pdf": FIGURE_DIR / "three_model_robustness_summary.pdf",
}
LOCK_FILE = PROJECT_ROOT / "tmp" / f"{EVALUATION_ID}.lock"


class ArrayWaferDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = np.asarray(self.images[index]).copy()
        tensor = torch.from_numpy(image).to(torch.float32).unsqueeze(0) / 2.0
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen three-model robustness comparison.")
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


def require_hash(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}: {path}")


def ensure_outputs_absent():
    collisions = []
    if OUTPUT_DIR.exists():
        collisions.append(OUTPUT_DIR)
    if LOCK_FILE.exists():
        collisions.append(LOCK_FILE)
    collisions.extend(path for path in PUBLISHED.values() if path.exists())
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing robustness outputs:\n"
            + "\n".join(str(path) for path in collisions)
        )


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_freeze_manifest() -> dict:
    require_hash(FREEZE_MANIFEST, EXPECTED_FREEZE_MANIFEST_SHA256, "freeze manifest")
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_unified_test":
        raise ValueError("Unexpected freeze-manifest status.")
    if manifest.get("seeds") != list(SEEDS) or manifest.get("test_tuning_allowed") is not False:
        raise ValueError("Freeze-manifest seed or tuning policy mismatch.")
    models = {item["model_id"]: item for item in manifest["models"]}
    if not set(MODEL_IDS).issubset(models):
        raise ValueError("Freeze manifest does not contain all three required models.")
    return manifest


def validate_static_inputs():
    require_hash(PROTOCOL_FILE, EXPECTED_PROTOCOL_SHA256, "comparison protocol")
    require_hash(ORIGINAL_PROTOCOL, EXPECTED_ORIGINAL_PROTOCOL_SHA256, "original robustness protocol")
    require_hash(UNIFIED_PER_SEED, EXPECTED_UNIFIED_PER_SEED_SHA256, "unified per-seed metrics")
    require_hash(HIGHRES_PER_SEED, EXPECTED_HIGHRES_PER_SEED_SHA256, "HighRes robustness metrics")
    require_hash(HIGHRES_AGGREGATE, EXPECTED_HIGHRES_AGGREGATE_SHA256, "HighRes aggregate metrics")
    require_hash(HIGHRES_RUN_MANIFEST, EXPECTED_HIGHRES_RUN_MANIFEST_SHA256, "HighRes run manifest")
    require_hash(IMAGE_FILE, EXPECTED_IMAGE_SHA256, "processed images")
    require_hash(METADATA_FILE, EXPECTED_METADATA_SHA256, "processed metadata")
    for relative_path, expected in FROZEN_CODE.items():
        require_hash(PROJECT_ROOT / relative_path, expected, f"frozen code {relative_path}")


def validate_checkpoints(manifest: dict, keep_models: bool = False):
    model_info = {item["model_id"]: item for item in manifest["models"]}
    loaded = []
    for model_id in BASELINE_MODEL_IDS:
        info = model_info[model_id]
        if int(info["parameter_count"]) <= 0 or len(info["checkpoints"]) != len(SEEDS):
            raise ValueError(f"Invalid checkpoint metadata for {model_id}")
        for checkpoint_info in info["checkpoints"]:
            seed = int(checkpoint_info["seed"])
            path = PROJECT_ROOT / checkpoint_info["checkpoint_path"]
            require_hash(path, checkpoint_info["checkpoint_sha256"], f"{model_id} seed {seed} checkpoint")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = MODEL_FACTORIES[model_id]()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if sum(parameter.numel() for parameter in model.parameters()) != int(info["parameter_count"]):
                raise ValueError(f"Parameter count mismatch: {path}")
            if int(checkpoint["epoch"]) != int(checkpoint_info["best_epoch"]):
                raise ValueError(f"Best epoch mismatch: {path}")
            if int(checkpoint["random_seed"]) != seed:
                raise ValueError(f"Random seed mismatch: {path}")
            if keep_models:
                loaded.append(
                    {
                        "model_id": model_id,
                        "model_name": MODEL_DISPLAY_NAMES[model_id],
                        "seed": seed,
                        "model": model.eval(),
                        "checkpoint_path": checkpoint_info["checkpoint_path"],
                        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
                    }
                )
            else:
                del model, checkpoint
    if len(loaded) not in (0, 6):
        raise RuntimeError("Expected zero or six loaded baseline models.")
    return loaded


def load_data():
    images = np.load(IMAGE_FILE, mmap_mode="r")
    metadata = pd.read_csv(METADATA_FILE)
    if images.shape != (len(metadata), 64, 64) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected processed data: {images.shape}, {images.dtype}")
    positions = np.flatnonzero(metadata["split"].eq("test").to_numpy())
    test_metadata = metadata.iloc[positions].reset_index(drop=True)
    labels = test_metadata["label_id"].to_numpy(dtype=np.int64)
    if len(labels) != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"Unexpected test size: {len(labels)}")
    if not np.array_equal(np.bincount(labels, minlength=NUM_CLASSES), EXPECTED_TEST_COUNTS):
        raise ValueError("Test class counts have changed.")
    source_indexes = test_metadata["source_index"].to_numpy(dtype=np.int64)
    return images, positions, labels, source_indexes


def severity_value(corruption: str, level_index: int):
    if corruption == "clean":
        return 0.0
    return float(ROBUSTNESS_LEVELS[corruption][level_index])


def validate_and_load_reused_rows():
    unified = pd.read_csv(UNIFIED_PER_SEED)
    highres = pd.read_csv(HIGHRES_PER_SEED)
    expected_highres_rows = len(SEEDS) * (1 + len(ROBUSTNESS_LEVELS) * len(LEVEL_NAMES)) * len(METRICS)
    if len(highres) != expected_highres_rows:
        raise ValueError(f"Unexpected HighRes robustness row count: {len(highres)}")
    if highres.duplicated(["seed", "corruption", "level", "metric"]).any():
        raise ValueError("HighRes robustness rows are not unique.")
    if set(highres["metric"]) != set(METRICS) or set(highres["seed"]) != set(SEEDS):
        raise ValueError("Unexpected HighRes robustness metrics or seeds.")

    rows = []
    highres_clean = highres.loc[highres["corruption"] == "clean"].set_index(["seed", "metric"])
    unified_highres = unified.loc[unified["model_id"] == "highres_shufflenet_v2"]
    for _, row in unified_highres.iterrows():
        for metric in METRICS:
            old_value = float(highres_clean.loc[(int(row["seed"]), metric), "value"])
            if old_value != float(row[metric]):
                raise ValueError(f"HighRes clean metric mismatch for seed={row['seed']} metric={metric}")

    for _, row in highres.iterrows():
        rows.append(
            {
                "model_id": "highres_shufflenet_v2",
                "model": MODEL_DISPLAY_NAMES["highres_shufflenet_v2"],
                "seed": int(row["seed"]),
                "corruption": row["corruption"],
                "level": row["level"],
                "level_index": int(row["level_index"]),
                "severity": severity_value(row["corruption"], int(row["level_index"])),
                "metric": row["metric"],
                "value": float(row["value"]),
                "clean_value": float(highres_clean.loc[(int(row["seed"]), row["metric"]), "value"]),
                "degradation_from_clean_pp": -float(row["delta_from_clean_pp"]),
                "source": "reused_highres_robustness",
            }
        )

    clean_values = {}
    for model_id in BASELINE_MODEL_IDS:
        subset = unified.loc[unified["model_id"] == model_id]
        if set(subset["seed"].astype(int)) != set(SEEDS) or len(subset) != len(SEEDS):
            raise ValueError(f"Unified clean metrics missing for {model_id}")
        for _, row in subset.iterrows():
            seed = int(row["seed"])
            clean_values[(model_id, seed)] = {metric: float(row[metric]) for metric in METRICS}
            for metric in METRICS:
                rows.append(
                    {
                        "model_id": model_id,
                        "model": MODEL_DISPLAY_NAMES[model_id],
                        "seed": seed,
                        "corruption": "clean",
                        "level": "clean",
                        "level_index": -1,
                        "severity": 0.0,
                        "metric": metric,
                        "value": float(row[metric]),
                        "clean_value": float(row[metric]),
                        "degradation_from_clean_pp": 0.0,
                        "source": "reused_unified_clean",
                    }
                )
    return rows, clean_values


def generate_corrupted_images(images, positions, source_indexes, corruption, level_index):
    output = np.empty((len(positions), 64, 64), dtype=np.uint8)
    for index, (position, source_index) in enumerate(zip(positions, source_indexes)):
        output[index] = apply_corruption(
            np.asarray(images[int(position)]).copy(),
            int(source_index),
            corruption,
            int(level_index),
        )
    return output


def evaluate(model, dataset, labels, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    predictions = []
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_images, _ in loader:
            predictions.append(model(batch_images.to(device)).argmax(dim=1).cpu().numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    predicted = np.concatenate(predictions).astype(np.uint8)
    if len(predicted) != len(labels):
        raise RuntimeError("Inference prediction count mismatch.")
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(labels, predicted, labels=np.arange(NUM_CLASSES), average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }
    matrix = confusion_matrix(labels, predicted, labels=np.arange(NUM_CLASSES))
    return predicted, metrics, matrix, elapsed


def aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_id", "model", "corruption", "level", "level_index", "severity", "metric"]
    return (
        per_seed.groupby(keys, sort=False, dropna=False)
        .agg(
            value_mean=("value", "mean"),
            value_sample_std=("value", lambda values: values.std(ddof=1)),
            degradation_from_clean_pp_mean=("degradation_from_clean_pp", "mean"),
            degradation_from_clean_pp_sample_std=(
                "degradation_from_clean_pp", lambda values: values.std(ddof=1)
            ),
            seeds=("seed", "count"),
        )
        .reset_index()
    )


def comparison_vs_highres(aggregate: pd.DataFrame) -> pd.DataFrame:
    join_keys = ["corruption", "level", "level_index", "severity", "metric"]
    highres = aggregate.loc[aggregate["model_id"] == "highres_shufflenet_v2", join_keys + [
        "value_mean", "degradation_from_clean_pp_mean"
    ]].rename(
        columns={
            "value_mean": "highres_value_mean",
            "degradation_from_clean_pp_mean": "highres_degradation_from_clean_pp_mean",
        }
    )
    baselines = aggregate.loc[aggregate["model_id"].isin(BASELINE_MODEL_IDS)].copy()
    compared = baselines.merge(highres, on=join_keys, how="left", validate="many_to_one")
    if compared[["highres_value_mean", "highres_degradation_from_clean_pp_mean"]].isna().any().any():
        raise ValueError("HighRes comparison rows are incomplete.")
    compared["value_delta_vs_highres_pp"] = (
        compared["value_mean"] - compared["highres_value_mean"]
    ) * 100.0
    compared["additional_degradation_vs_highres_pp"] = (
        compared["degradation_from_clean_pp_mean"]
        - compared["highres_degradation_from_clean_pp_mean"]
    )
    return compared


def descriptive_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    perturbed = aggregate.loc[aggregate["corruption"] != "clean"].copy()
    rows = []
    scopes = [("all_12_conditions", None)] + [
        (corruption, corruption) for corruption in ROBUSTNESS_LEVELS
    ]
    for (model_id, model, metric), group in perturbed.groupby(
        ["model_id", "model", "metric"], sort=False
    ):
        for scope_name, corruption in scopes:
            scoped = group if corruption is None else group.loc[group["corruption"] == corruption]
            rows.append(
                {
                    "model_id": model_id,
                    "model": model,
                    "metric": metric,
                    "scope": scope_name,
                    "conditions": len(scoped),
                    "mean_perturbed_value": float(scoped["value_mean"].mean()),
                    "mean_degradation_from_clean_pp": float(
                        scoped["degradation_from_clean_pp_mean"].mean()
                    ),
                    "condition_std_degradation_pp": float(
                        scoped["degradation_from_clean_pp_mean"].std(ddof=1)
                    ),
                    "worst_degradation_from_clean_pp": float(
                        scoped["degradation_from_clean_pp_mean"].max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_curves(aggregate: pd.DataFrame, staging: Path):
    colors = {
        "highres_shufflenet_v2": "#0072B2",
        "standard_shufflenet_v2": "#D55E00",
        "resnet18": "#009E73",
    }
    figure, axes = plt.subplots(4, 3, figsize=(15, 15), constrained_layout=True)
    for row_index, corruption in enumerate(ROBUSTNESS_LEVELS):
        for column_index, metric in enumerate(METRICS):
            axis = axes[row_index, column_index]
            subset = aggregate.loc[
                (aggregate["corruption"] == corruption) & (aggregate["metric"] == metric)
            ]
            for model_id in MODEL_IDS:
                curve = subset.loc[subset["model_id"] == model_id].sort_values("level_index")
                axis.errorbar(
                    curve["level_index"],
                    curve["degradation_from_clean_pp_mean"],
                    yerr=curve["degradation_from_clean_pp_sample_std"],
                    marker="o",
                    capsize=2,
                    linewidth=1.5,
                    color=colors[model_id],
                    label=MODEL_DISPLAY_NAMES[model_id],
                )
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_xticks(range(len(LEVEL_NAMES)), [name.title() for name in LEVEL_NAMES])
            axis.set_ylabel("Degradation from clean (pp)")
            axis.set_title(f"{corruption.replace('_', ' ').title()} - {metric}", fontsize=10)
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Robustness degradation under identical frozen perturbations", fontsize=15)
    for suffix in ("png", "pdf"):
        figure.savefig(
            staging / f"three_model_robustness_curves.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_summary(summary: pd.DataFrame, staging: Path):
    subset = summary.loc[summary["scope"] == "all_12_conditions"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for axis, metric in zip(axes, METRICS):
        current = subset.loc[subset["metric"] == metric].set_index("model_id").loc[list(MODEL_IDS)]
        axis.bar(
            range(len(MODEL_IDS)),
            current["mean_degradation_from_clean_pp"],
            yerr=current["condition_std_degradation_pp"],
            capsize=3,
            color=colors,
        )
        axis.set_xticks(range(len(MODEL_IDS)), ["HighRes", "Standard", "ResNet18"], rotation=15)
        axis.set_ylabel("Mean degradation across 12 conditions (pp)")
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Descriptive mean degradation across frozen perturbation conditions")
    for suffix in ("png", "pdf"):
        figure.savefig(
            staging / f"three_model_robustness_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, float_format="%.10g")


def create_manifests(staging: Path, device, durations, loaded_models):
    artifact_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in {"run_manifest.json", "artifact_manifest.json"}
    }
    run_manifest = {
        "schema_version": 1,
        "evaluation_id": EVALUATION_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "test_samples": EXPECTED_TEST_SAMPLES,
        "seeds": list(SEEDS),
        "models": list(MODEL_IDS),
        "new_inference_runs": 72,
        "reused_highres_metric_rows": 117,
        "reused_baseline_clean_metric_rows": 18,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "durations_seconds": durations,
        "checkpoint_sha256": {
            f"{entry['model_id']}_seed{entry['seed']}": entry["checkpoint_sha256"]
            for entry in loaded_models
        },
        "frozen_perturbations": {
            "levels": ROBUSTNESS_LEVELS,
            "level_names": LEVEL_NAMES,
            "batch_size": BATCH_SIZE,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifact_sha256": artifact_hashes,
    }
    (staging / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    all_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    (staging / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_id": EVALUATION_ID,
                "created_at": datetime.now().astimezone().isoformat(),
                "artifact_sha256": all_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def publish_artifacts():
    for source_name, target in PUBLISHED.items():
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite published artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(OUTPUT_DIR / source_name, temporary)
        temporary.replace(target)


def run_once(manifest, reused_rows, clean_values, images, positions, labels, source_indexes):
    ensure_outputs_absent()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.close(descriptor)
    staging = Path(tempfile.mkdtemp(prefix=f"{EVALUATION_ID}_", dir=PROJECT_ROOT / "tmp"))
    device = select_device()
    loaded_models = validate_checkpoints(manifest, keep_models=True)
    for entry in loaded_models:
        entry["model"].to(device).eval()

    rows = list(reused_rows)
    predictions = {}
    matrices = {}
    durations = {}
    started = time.perf_counter()
    try:
        condition_number = 0
        for corruption in ROBUSTNESS_LEVELS:
            for level_index, level_name in enumerate(LEVEL_NAMES):
                condition_number += 1
                phase = time.perf_counter()
                corrupted = generate_corrupted_images(
                    images, positions, source_indexes, corruption, level_index
                )
                durations[f"generate_{corruption}_{level_name}"] = time.perf_counter() - phase
                dataset = ArrayWaferDataset(corrupted, labels)
                print(
                    f"CONDITION_READY {condition_number}/12 {corruption} {level_name}",
                    flush=True,
                )
                for entry in loaded_models:
                    key = f"{entry['model_id']}_seed{entry['seed']}_{corruption}_{level_name}"
                    predicted, metric_values, matrix, elapsed = evaluate(
                        entry["model"], dataset, labels, device
                    )
                    predictions[key] = predicted
                    matrices[key] = matrix.tolist()
                    durations[f"infer_{key}"] = elapsed
                    for metric, value in metric_values.items():
                        clean_value = clean_values[(entry["model_id"], entry["seed"])][metric]
                        rows.append(
                            {
                                "model_id": entry["model_id"],
                                "model": entry["model_name"],
                                "seed": entry["seed"],
                                "corruption": corruption,
                                "level": level_name,
                                "level_index": level_index,
                                "severity": severity_value(corruption, level_index),
                                "metric": metric,
                                "value": value,
                                "clean_value": clean_value,
                                "degradation_from_clean_pp": (clean_value - value) * 100.0,
                                "source": "new_inference",
                            }
                        )
                    print(
                        f"INFER_COMPLETE {entry['model_id']} seed={entry['seed']} "
                        f"condition={corruption}/{level_name} seconds={elapsed:.2f}",
                        flush=True,
                    )
                del dataset, corrupted

        per_seed = pd.DataFrame(rows)
        expected_rows = len(MODEL_IDS) * len(SEEDS) * (
            1 + len(ROBUSTNESS_LEVELS) * len(LEVEL_NAMES)
        ) * len(METRICS)
        if len(per_seed) != expected_rows:
            raise RuntimeError(f"Unexpected final metric row count: {len(per_seed)}")
        aggregate = aggregate_metrics(per_seed)
        compared = comparison_vs_highres(aggregate)
        summary = descriptive_summary(aggregate)

        write_csv(per_seed, staging / "per_seed_metrics.csv")
        write_csv(aggregate, staging / "aggregate_metrics.csv")
        write_csv(compared, staging / "comparison_vs_highres.csv")
        write_csv(summary, staging / "descriptive_summary.csv")
        np.savez_compressed(staging / "baseline_perturbed_predictions.npz", **predictions)
        (staging / "baseline_perturbed_confusion_matrices.json").write_text(
            json.dumps(matrices, indent=2) + "\n", encoding="utf-8"
        )
        plot_curves(aggregate, staging)
        plot_summary(summary, staging)
        durations["total_before_manifest"] = time.perf_counter() - started
        create_manifests(staging, device, durations, loaded_models)
        staging.replace(OUTPUT_DIR)
        publish_artifacts()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for entry in loaded_models:
            entry["model"].to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()
        LOCK_FILE.unlink(missing_ok=True)

    print(f"ROBUSTNESS_OUTPUT {OUTPUT_DIR}")
    print("Three-model robustness comparison completed.")


def main():
    args = parse_args()
    ensure_outputs_absent()
    validate_static_inputs()
    manifest = load_freeze_manifest()
    validate_checkpoints(manifest, keep_models=False)
    images, positions, labels, source_indexes = load_data()
    reused_rows, clean_values = validate_and_load_reused_rows()
    print("PREFLIGHT_MODELS 3")
    print("PREFLIGHT_BASELINE_CHECKPOINTS 6")
    print("PREFLIGHT_REUSED_HIGHRES_METRIC_ROWS 117")
    print("PREFLIGHT_NEW_INFERENCE_RUNS 72")
    print(f"PREFLIGHT_TEST_SAMPLES {len(labels)}")
    print("PREFLIGHT_INPUT_HASHES_OK")
    if args.preflight:
        print("Preflight completed; no robustness outputs were written.")
        return
    run_once(
        manifest,
        reused_rows,
        clean_values,
        images,
        positions,
        labels,
        source_indexes,
    )


if __name__ == "__main__":
    main()
