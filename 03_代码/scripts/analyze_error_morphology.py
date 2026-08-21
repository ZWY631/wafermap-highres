#!/usr/bin/env python3
"""Quantify associations between wafer-map morphology and frozen test errors."""

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
import scipy
from scipy import ndimage
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import WM811K_CLASS_NAMES


ANALYSIS_ID = "20260807_error_morphology_quantification"
PROTOCOL_FILE = (
    PROJECT_ROOT / "00_项目管理" / "20260807_错误形态量化分析冻结协议.md"
)
EXPECTED_PROTOCOL_SHA256 = (
    "5f82c6d72e3aa8feb6d2bb1f8499186dc75ac28f60d410b0d191976db053255e"
)
IMBALANCE_CONCLUSION = (
    PROJECT_ROOT / "00_项目管理" / "20260807_类别不平衡测试结论冻结清单.json"
)
EXPECTED_IMBALANCE_CONCLUSION_SHA256 = (
    "d70e6c0ead89f9568246f28129b59a7c48432c4cefdcfbeecb726792b237c9af"
)
PROCESSED_DIR = (
    PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64"
)
IMAGE_FILE = PROCESSED_DIR / "images_uint8.npy"
METADATA_FILE = PROCESSED_DIR / "metadata.csv"
EXPECTED_IMAGE_SHA256 = (
    "ec496dd8a5cdb2f83c54a048497d1f6d77e64ae74efd495c36456629c9cf9aa3"
)
EXPECTED_METADATA_SHA256 = (
    "b6b1e0c2d819c60aa879e50015408161324616cbe5d928f4001d6609a7c9e681"
)

SEEDS = (42, 123, 2026)
PREDICTION_FILES = {
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

EXPECTED_TEST_SAMPLES = 25943
EXPECTED_TEST_COUNTS = np.asarray(
    [644, 83, 778, 1452, 539, 21, 130, 179, 22117], dtype=np.int64
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807
NONE_CLASS_ID = 8

FEATURES = (
    "wafer_die_count",
    "fail_die_count",
    "fail_die_fraction",
    "component_count",
    "largest_component_fraction",
    "singleton_fail_fraction",
    "component_size_cv",
    "centroid_offset_normalized",
    "radial_mean_normalized",
    "radial_std_normalized",
    "edge_fail_fraction",
    "spatial_dispersion_normalized",
)
OUTCOMES = ("error_rate_across_seeds", "predicted_none_rate_across_seeds")
KEY_COLUMNS = (
    "source_index",
    "lotName",
    "waferIndex",
    "label",
    "split",
    "label_id",
)

OUTPUT_DIR = (
    PROJECT_ROOT / "04_实验" / "metrics" / ANALYSIS_ID
)
TABLE_DIR = PROJECT_ROOT / "05_结果" / "tables"
FIGURE_DIR = PROJECT_ROOT / "05_结果" / "figures" / "model_results"
PUBLISHED_ARTIFACTS = {
    "sample_level_correlations.csv": TABLE_DIR / "table_error_morphology_correlations.csv",
    "per_class_morphology_error_summary.csv": TABLE_DIR / "table_error_morphology_by_class.csv",
    "error_group_morphology_summary.csv": TABLE_DIR / "table_error_morphology_by_error_group.csv",
    "error_group_morphology_distributions.png": FIGURE_DIR / "error_morphology_distributions.png",
    "error_group_morphology_distributions.pdf": FIGURE_DIR / "error_morphology_distributions.pdf",
    "class_feature_error_scatter.png": FIGURE_DIR / "error_morphology_class_scatter.png",
    "class_feature_error_scatter.pdf": FIGURE_DIR / "error_morphology_class_scatter.pdf",
}
LOCK_FILE = PROJECT_ROOT / "tmp" / f"{ANALYSIS_ID}.lock"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen WM-811K morphology/error association analysis."
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


def require_hash(path: Path, expected: str, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}: {path}"
        )


def ensure_outputs_absent():
    collisions = []
    if OUTPUT_DIR.exists():
        collisions.append(OUTPUT_DIR)
    collisions.extend(path for path in PUBLISHED_ARTIFACTS.values() if path.exists())
    if collisions:
        joined = "\n".join(str(path) for path in collisions)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{joined}")


def normalize_correct(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false")).all():
        raise ValueError("Prediction correct column contains invalid values.")
    return normalized.eq("true").to_numpy(dtype=bool)


def load_and_validate_inputs():
    require_hash(PROTOCOL_FILE, EXPECTED_PROTOCOL_SHA256, "frozen protocol")
    require_hash(
        IMBALANCE_CONCLUSION,
        EXPECTED_IMBALANCE_CONCLUSION_SHA256,
        "imbalance conclusion",
    )
    require_hash(IMAGE_FILE, EXPECTED_IMAGE_SHA256, "processed image array")
    require_hash(METADATA_FILE, EXPECTED_METADATA_SHA256, "processed metadata")

    images = np.load(IMAGE_FILE, mmap_mode="r")
    metadata = pd.read_csv(METADATA_FILE)
    if images.shape != (len(metadata), 64, 64) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected processed data: shape={images.shape}, dtype={images.dtype}")
    values = np.unique(images)
    if not np.array_equal(values, np.asarray([0, 1, 2], dtype=np.uint8)):
        raise ValueError(f"Unexpected wafer-map pixel values: {values.tolist()}")
    if metadata["source_index"].duplicated().any():
        raise ValueError("metadata.csv contains duplicate source_index values.")

    test_metadata = metadata.loc[metadata["split"] == "test"].copy()
    test_metadata = test_metadata.reset_index(names="image_row")
    if len(test_metadata) != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"Unexpected test size: {len(test_metadata)}")
    counts = np.bincount(
        test_metadata["label_id"].to_numpy(dtype=np.int64),
        minlength=len(WM811K_CLASS_NAMES),
    )
    if not np.array_equal(counts, EXPECTED_TEST_COUNTS):
        raise ValueError(f"Unexpected test class counts: {counts.tolist()}")

    predictions = {}
    reference_keys = None
    for seed in SEEDS:
        entry = PREDICTION_FILES[seed]
        path = PROJECT_ROOT / entry["path"]
        require_hash(path, entry["sha256"], f"seed {seed} frozen predictions")
        frame = pd.read_csv(path)
        required = set(KEY_COLUMNS) | {
            "true_label_id", "predicted_label_id", "predicted_label", "correct"
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")
        if len(frame) != EXPECTED_TEST_SAMPLES:
            raise ValueError(f"Unexpected prediction rows in {path}: {len(frame)}")
        ids = frame["predicted_label_id"].to_numpy(dtype=np.int64)
        true_ids = frame["true_label_id"].to_numpy(dtype=np.int64)
        label_ids = frame["label_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(true_ids, label_ids):
            raise ValueError(f"True-label columns disagree in {path}")
        if not np.array_equal(ids == true_ids, normalize_correct(frame["correct"])):
            raise ValueError(f"Correct flags disagree with predictions in {path}")
        names = np.asarray(WM811K_CLASS_NAMES, dtype=object)[ids]
        if not np.array_equal(names, frame["predicted_label"].to_numpy(dtype=object)):
            raise ValueError(f"Predicted class names disagree with IDs in {path}")
        current_keys = frame[list(KEY_COLUMNS)].reset_index(drop=True)
        if reference_keys is None:
            reference_keys = current_keys
        else:
            pd.testing.assert_frame_equal(
                reference_keys, current_keys, check_dtype=False, check_exact=True
            )
        predictions[seed] = frame

    metadata_keys = test_metadata[list(KEY_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        metadata_keys, reference_keys, check_dtype=False, check_exact=True
    )
    return images, test_metadata, predictions


def extract_morphology(image: np.ndarray) -> dict[str, float]:
    wafer_mask = image > 0
    fail_mask = image == 2
    wafer_yx = np.argwhere(wafer_mask)
    fail_yx = np.argwhere(fail_mask)
    wafer_count = int(len(wafer_yx))
    fail_count = int(len(fail_yx))
    if wafer_count == 0:
        raise ValueError("Encountered a wafer map without valid die cells.")

    labeled, component_count = ndimage.label(
        fail_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if component_count:
        sizes = np.bincount(labeled.ravel())[1:].astype(np.float64)
        largest_fraction = float(sizes.max() / fail_count)
        singleton_fraction = float(sizes[sizes == 1].sum() / fail_count)
        component_cv = (
            float(sizes.std(ddof=1) / sizes.mean()) if component_count > 1 else 0.0
        )
    else:
        sizes = np.empty(0, dtype=np.float64)
        largest_fraction = 0.0
        singleton_fraction = 0.0
        component_cv = 0.0

    result = {
        "wafer_die_count": float(wafer_count),
        "fail_die_count": float(fail_count),
        "fail_die_fraction": float(fail_count / wafer_count),
        "component_count": float(component_count),
        "largest_component_fraction": largest_fraction,
        "singleton_fail_fraction": singleton_fraction,
        "component_size_cv": component_cv,
        "centroid_offset_normalized": np.nan,
        "radial_mean_normalized": np.nan,
        "radial_std_normalized": np.nan,
        "edge_fail_fraction": 0.0,
        "spatial_dispersion_normalized": np.nan,
    }
    if fail_count == 0:
        return result

    wafer_centroid = wafer_yx.mean(axis=0)
    wafer_distances = np.linalg.norm(wafer_yx - wafer_centroid, axis=1)
    wafer_radius = float(wafer_distances.max())
    if wafer_radius <= 0:
        raise ValueError("Encountered a wafer map with zero geometric radius.")
    fail_centroid = fail_yx.mean(axis=0)
    fail_radii = np.linalg.norm(fail_yx - wafer_centroid, axis=1) / wafer_radius
    self_distances = np.linalg.norm(fail_yx - fail_centroid, axis=1)
    result.update(
        {
            "centroid_offset_normalized": float(
                np.linalg.norm(fail_centroid - wafer_centroid) / wafer_radius
            ),
            "radial_mean_normalized": float(fail_radii.mean()),
            "radial_std_normalized": (
                float(fail_radii.std(ddof=1)) if fail_count > 1 else 0.0
            ),
            "edge_fail_fraction": float(np.mean(fail_radii >= 0.8)),
            "spatial_dispersion_normalized": float(
                np.sqrt(np.mean(self_distances**2)) / wafer_radius
            ),
        }
    )
    return result


def build_sample_table(images, test_metadata, predictions):
    rows = []
    image_rows = test_metadata["image_row"].to_numpy(dtype=np.int64)
    for position, image_row in enumerate(image_rows):
        if position and position % 5000 == 0:
            print(f"MORPHOLOGY_PROGRESS {position}/{EXPECTED_TEST_SAMPLES}", flush=True)
        rows.append(extract_morphology(np.asarray(images[image_row])))
    morphology = pd.DataFrame(rows, columns=FEATURES)

    sample = test_metadata[list(KEY_COLUMNS)].reset_index(drop=True).copy()
    sample = pd.concat([sample, morphology], axis=1)
    true_ids = sample["label_id"].to_numpy(dtype=np.int64)
    predicted = np.column_stack(
        [predictions[seed]["predicted_label_id"].to_numpy(dtype=np.int64) for seed in SEEDS]
    )
    errors = predicted != true_ids[:, None]
    predicted_none = predicted == NONE_CLASS_ID
    for column, seed in enumerate(SEEDS):
        sample[f"predicted_label_id_seed{seed}"] = predicted[:, column]
        sample[f"correct_seed{seed}"] = ~errors[:, column]
    sample["error_count_across_seeds"] = errors.sum(axis=1)
    sample["error_rate_across_seeds"] = errors.mean(axis=1)
    sample["predicted_none_count_across_seeds"] = predicted_none.sum(axis=1)
    sample["predicted_none_rate_across_seeds"] = np.where(
        true_ids != NONE_CLASS_ID, predicted_none.mean(axis=1), np.nan
    )
    sample["error_group"] = np.select(
        [sample["error_count_across_seeds"] == 0, sample["error_count_across_seeds"] == 3],
        ["all_correct", "all_wrong"],
        default="partially_wrong",
    )
    sample["unanimous_to_none"] = (
        (true_ids != NONE_CLASS_ID) & predicted_none.all(axis=1)
    )
    return sample


def make_class_summary(sample: pd.DataFrame) -> pd.DataFrame:
    columns = list(FEATURES) + [
        "error_rate_across_seeds",
        "predicted_none_rate_across_seeds",
        "unanimous_to_none",
    ]
    summary = sample.groupby(["label_id", "label"], sort=True)[columns].agg(
        ["count", "mean", "std", "median"]
    )
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def make_error_group_summary(sample: pd.DataFrame) -> pd.DataFrame:
    defect = sample.loc[sample["label_id"] != NONE_CLASS_ID].copy()
    rows = []
    order = ("all_correct", "partially_wrong", "all_wrong")
    for group_name in order:
        group = defect.loc[defect["error_group"] == group_name]
        for feature in FEATURES:
            values = group[feature].dropna().to_numpy(dtype=np.float64)
            rows.append(
                {
                    "error_group": group_name,
                    "feature": feature,
                    "n": len(values),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                    "q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def safe_spearman(x, y) -> tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan, int(valid.sum())
    return float(spearmanr(x[valid], y[valid]).statistic), int(valid.sum())


def within_class_percentiles(defect: pd.DataFrame) -> pd.DataFrame:
    ranked = defect.copy()
    for feature in FEATURES:
        ranked[feature] = ranked.groupby("label_id", sort=False)[feature].transform(
            lambda values: values.rank(method="average", pct=True)
        )
    return ranked


def stratified_bootstrap_indices(labels: np.ndarray, rng) -> np.ndarray:
    chunks = []
    for class_id in range(NONE_CLASS_ID):
        indices = np.flatnonzero(labels == class_id)
        chunks.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(chunks)


def correlation_matrix(frame: pd.DataFrame) -> np.ndarray:
    matrix = frame[list(FEATURES) + list(OUTCOMES)].to_numpy(dtype=np.float64)
    correlation = spearmanr(matrix, axis=0, nan_policy="omit").statistic
    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.shape != (len(FEATURES) + len(OUTCOMES),) * 2:
        raise ValueError("Unexpected Spearman correlation matrix shape.")
    return correlation


def make_sample_correlations(sample: pd.DataFrame):
    defect = sample.loc[sample["label_id"] != NONE_CLASS_ID].reset_index(drop=True)
    representations = {
        "raw": defect,
        "within_class_percentile": within_class_percentiles(defect),
    }
    point = {name: correlation_matrix(frame) for name, frame in representations.items()}
    bootstrap = {
        name: np.full(
            (BOOTSTRAP_REPLICATES, len(FEATURES), len(OUTCOMES)),
            np.nan,
            dtype=np.float64,
        )
        for name in representations
    }
    labels = defect["label_id"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = stratified_bootstrap_indices(labels, rng)
        for name, frame in representations.items():
            corr = correlation_matrix(frame.iloc[indices])
            bootstrap[name][replicate] = corr[
                : len(FEATURES), len(FEATURES) :
            ]
        if (replicate + 1) % 200 == 0:
            print(
                f"BOOTSTRAP_PROGRESS {replicate + 1}/{BOOTSTRAP_REPLICATES}",
                flush=True,
            )

    rows = []
    for name, frame in representations.items():
        for feature_index, feature in enumerate(FEATURES):
            for outcome_index, outcome in enumerate(OUTCOMES):
                x = frame[feature].to_numpy(dtype=np.float64)
                y = frame[outcome].to_numpy(dtype=np.float64)
                _, valid_n = safe_spearman(x, y)
                values = bootstrap[name][:, feature_index, outcome_index]
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "representation": name,
                        "feature": feature,
                        "outcome": outcome,
                        "valid_n": valid_n,
                        "spearman_rho": point[name][
                            feature_index, len(FEATURES) + outcome_index
                        ],
                        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
                        "bootstrap_replicates_valid": len(finite),
                        "bootstrap_ci_lower_2_5pct": (
                            float(np.quantile(finite, 0.025)) if len(finite) else np.nan
                        ),
                        "bootstrap_ci_upper_97_5pct": (
                            float(np.quantile(finite, 0.975)) if len(finite) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows), bootstrap


def make_class_correlations(class_summary: pd.DataFrame) -> pd.DataFrame:
    defect = class_summary.loc[class_summary["label_id"] != NONE_CLASS_ID].copy()
    rows = []
    for feature in FEATURES:
        x = defect[f"{feature}_mean"].to_numpy(dtype=np.float64)
        for outcome in OUTCOMES:
            y = defect[f"{outcome}_mean"].to_numpy(dtype=np.float64)
            rho, valid_n = safe_spearman(x, y)
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n_classes": valid_n,
                    "spearman_rho_descriptive": rho,
                    "interpretation_limit": "descriptive_only_n_equals_8",
                }
            )
    return pd.DataFrame(rows)


def plot_error_groups(sample: pd.DataFrame, staging: Path):
    defect = sample.loc[sample["label_id"] != NONE_CLASS_ID]
    groups = ("all_correct", "partially_wrong", "all_wrong")
    labels = ("All correct", "Partly wrong", "All wrong")
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    for axis, feature in zip(axes.flat, FEATURES):
        data = [
            defect.loc[defect["error_group"] == group, feature].dropna().to_numpy()
            for group in groups
        ]
        axis.boxplot(data, tick_labels=labels, showfliers=False)
        axis.set_title(feature.replace("_", " "), fontsize=10)
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Defect morphology by three-seed error stability group", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(
            staging / f"error_group_morphology_distributions.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_class_scatter(class_summary: pd.DataFrame, staging: Path):
    defect = class_summary.loc[class_summary["label_id"] != NONE_CLASS_ID].copy()
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    for axis, feature in zip(axes.flat, FEATURES):
        x = defect[f"{feature}_mean"].to_numpy(dtype=np.float64)
        y_error = defect["error_rate_across_seeds_mean"].to_numpy(dtype=np.float64)
        y_none = defect["predicted_none_rate_across_seeds_mean"].to_numpy(dtype=np.float64)
        axis.scatter(x, y_error, label="Any error", marker="o", s=28)
        axis.scatter(x, y_none, label="Predicted none", marker="x", s=32)
        for row_index, class_name in enumerate(defect["label"]):
            if np.isfinite(x[row_index]):
                axis.annotate(
                    class_name,
                    (x[row_index], y_error[row_index]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                )
        axis.set_title(feature.replace("_", " "), fontsize=10)
        axis.set_ylabel("Mean rate", fontsize=8)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Class-level morphology and error rates (descriptive, n=8 classes)",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            staging / f"class_feature_error_scatter.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, float_format="%.10g")


def create_manifests(staging: Path, durations: dict):
    artifact_hashes = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in {"run_manifest.json", "artifact_manifest.json"}
    }
    run_manifest = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "test_samples": EXPECTED_TEST_SAMPLES,
        "defect_samples": int(EXPECTED_TEST_COUNTS[:NONE_CLASS_ID].sum()),
        "seeds": list(SEEDS),
        "bootstrap": {
            "method": "stratified by true defect class with replacement",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "durations_seconds": durations,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "interpretation_constraints": [
            "Associations are not causal effects.",
            "Failed-die cells are not manual pixel-level defect masks.",
            "Class-level n=8 correlations are descriptive only.",
        ],
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
    (staging / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "analysis_id": ANALYSIS_ID,
                "created_at": datetime.now().astimezone().isoformat(),
                "artifact_sha256": full_hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def publish_artifacts():
    for source_name, target in PUBLISHED_ARTIFACTS.items():
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite published artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(OUTPUT_DIR / source_name, temporary)
        temporary.replace(target)


def run_once(images, test_metadata, predictions):
    ensure_outputs_absent()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    os.close(descriptor)
    staging = Path(
        tempfile.mkdtemp(prefix=f"{ANALYSIS_ID}_", dir=PROJECT_ROOT / "tmp")
    )
    started = time.perf_counter()
    durations = {}
    try:
        phase = time.perf_counter()
        sample = build_sample_table(images, test_metadata, predictions)
        durations["morphology_and_error_table"] = time.perf_counter() - phase

        phase = time.perf_counter()
        class_summary = make_class_summary(sample)
        group_summary = make_error_group_summary(sample)
        sample_correlations, bootstrap = make_sample_correlations(sample)
        class_correlations = make_class_correlations(class_summary)
        durations["summaries_and_bootstrap"] = time.perf_counter() - phase

        phase = time.perf_counter()
        write_csv(sample, staging / "sample_morphology_error_table.csv")
        write_csv(class_summary, staging / "per_class_morphology_error_summary.csv")
        write_csv(group_summary, staging / "error_group_morphology_summary.csv")
        write_csv(sample_correlations, staging / "sample_level_correlations.csv")
        write_csv(class_correlations, staging / "class_level_descriptive_correlations.csv")
        np.savez_compressed(
            staging / "bootstrap_correlation_replicates.npz",
            raw=bootstrap["raw"],
            within_class_percentile=bootstrap["within_class_percentile"],
            features=np.asarray(FEATURES),
            outcomes=np.asarray(OUTCOMES),
        )
        plot_error_groups(sample, staging)
        plot_class_scatter(class_summary, staging)
        durations["write_tables_and_figures"] = time.perf_counter() - phase
        durations["total_before_manifest"] = time.perf_counter() - started
        create_manifests(staging, durations)
        staging.replace(OUTPUT_DIR)
        publish_artifacts()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        LOCK_FILE.unlink(missing_ok=True)

    print(f"ANALYSIS_OUTPUT {OUTPUT_DIR}")
    print("Error-morphology quantification completed.")


def main():
    args = parse_args()
    if LOCK_FILE.exists():
        raise RuntimeError(f"Analysis lock already exists: {LOCK_FILE}")
    ensure_outputs_absent()
    images, test_metadata, predictions = load_and_validate_inputs()
    print(f"PREFLIGHT_TEST_SAMPLES {len(test_metadata)}")
    print(f"PREFLIGHT_DEFECT_SAMPLES {int(EXPECTED_TEST_COUNTS[:NONE_CLASS_ID].sum())}")
    print(f"PREFLIGHT_BOOTSTRAP_REPLICATES {BOOTSTRAP_REPLICATES}")
    print("PREFLIGHT_INPUT_HASHES_OK")
    if args.preflight:
        print("Preflight completed; no analysis outputs were written.")
        return
    run_once(images, test_metadata, predictions)


if __name__ == "__main__":
    main()
