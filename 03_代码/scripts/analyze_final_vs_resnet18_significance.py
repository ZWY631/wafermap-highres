#!/usr/bin/env python3
"""Paired statistical comparison of frozen predictions without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_EVALUATION_ID = "20260729_highres_ce_multiseed_final_test"
ANALYSIS_ID = "20260729_final_vs_resnet18_statistical_significance"

BASELINE_PREDICTIONS = (
    PROJECT_ROOT / "04_实验" / "metrics" / "resnet18_baseline_full_predictions.csv"
)
BASELINE_SUMMARY = (
    PROJECT_ROOT / "04_实验" / "metrics" / "resnet18_baseline_full_test_summary.json"
)
FINAL_EVALUATION_DIR = (
    PROJECT_ROOT / "04_实验" / "metrics" / FINAL_EVALUATION_ID
)
FINAL_PREDICTION_DIR = FINAL_EVALUATION_DIR / "predictions"
FINAL_ARTIFACT_MANIFEST = FINAL_EVALUATION_DIR / "artifact_manifest.json"
FINAL_PER_SEED_SUMMARY = FINAL_EVALUATION_DIR / "test_per_seed_summary.csv"

EXPECTED_BASELINE_PREDICTIONS_SHA256 = (
    "1e64acde099bafaa5e1e3cd6c4460cd43c476ded886216f8c02196b942cecd68"
)
EXPECTED_BASELINE_SUMMARY_SHA256 = (
    "b77c5d3342db482d3d31d7473fca1f23745fd80be9b2492c7ff6e61d9cf8b7d8"
)
EXPECTED_FINAL_ARTIFACT_MANIFEST_SHA256 = (
    "70d62a8d3f122871bedb609000dbde71d6238b5f69dcf75dfc587bb83643d33c"
)

EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_TEST_SAMPLES = 25943
CLASS_NAMES = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
N_CLASSES = len(CLASS_NAMES)

BOOTSTRAP_REPLICATES = 10_000
RANDOMIZATION_REPLICATES = 10_000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
BOOTSTRAP_RANDOM_SEED_BASE = 2026072900
RANDOMIZATION_RANDOM_SEED_BASE = 2026073900

METRIC_FILENAMES = (
    "paired_observed_metrics.csv",
    "paired_accuracy_mcnemar.csv",
    "paired_bootstrap_summary.csv",
    "paired_macro_f1_randomization.csv",
    "bootstrap_replicates.csv",
    "macro_f1_randomization_replicates.csv",
    "analysis_summary.json",
    "analysis_manifest.json",
)
PAPER_TABLE_NAME = "table_final_vs_resnet18_statistical_significance.csv"
FIGURE_STEM = "final_vs_resnet18_statistical_significance"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved ResNet18 and final-model predictions with paired "
            "tests. No model, checkpoint, or wafer-image dataset is loaded."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate prediction files and observed metrics only.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run paired statistical analyses and create formal outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "metrics_dir": output_root / "04_实验" / "metrics" / ANALYSIS_ID,
        "figure_dir": output_root / "05_结果" / "figures" / "model_results",
        "table_dir": output_root / "05_结果" / "tables",
    }


def expected_output_files(output_root: Path) -> list[Path]:
    paths = output_paths(output_root)
    files = [paths["metrics_dir"] / name for name in METRIC_FILENAMES]
    files.extend(
        [
            paths["table_dir"] / PAPER_TABLE_NAME,
            paths["figure_dir"] / f"{FIGURE_STEM}.png",
            paths["figure_dir"] / f"{FIGURE_STEM}.pdf",
        ]
    )
    return files


def ensure_outputs_are_available(output_root: Path):
    paths = output_paths(output_root)
    existing = []
    if paths["metrics_dir"].exists():
        existing.append(paths["metrics_dir"])
    existing.extend(path for path in expected_output_files(output_root) if path.exists())
    if existing:
        formatted = "\n".join(f"- {path}" for path in sorted(set(existing)))
        raise FileExistsError(
            "Refusing to overwrite existing significance-analysis outputs:\n"
            f"{formatted}"
        )


def normalize_correct(series: pd.Series, source: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if normalized.isna().any():
        raise ValueError(f"Invalid correct values in {source}")
    return normalized.astype(bool)


def validate_prediction_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {
        "source_index",
        "lotName",
        "waferIndex",
        "label",
        "split",
        "label_id",
        "predicted_label_id",
        "predicted_label",
        "correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns in {source}: {missing}")
    if len(frame) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} rows in {source}, found {len(frame)}."
        )
    if frame["source_index"].duplicated().any():
        raise ValueError(f"Duplicate source_index values in {source}")
    if not frame["split"].eq("test").all():
        raise ValueError(f"Non-test rows found in {source}")
    if set(frame["label"].astype(str)) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected class names in {source}")

    true_ids = frame["label_id"].to_numpy(dtype=np.int64)
    predicted_ids = frame["predicted_label_id"].to_numpy(dtype=np.int64)
    if np.any((true_ids < 0) | (true_ids >= N_CLASSES)):
        raise ValueError(f"Invalid true label IDs in {source}")
    if np.any((predicted_ids < 0) | (predicted_ids >= N_CLASSES)):
        raise ValueError(f"Invalid predicted label IDs in {source}")
    if not np.array_equal(np.asarray(CLASS_NAMES)[true_ids], frame["label"]):
        raise ValueError(f"True label names are inconsistent in {source}")
    if not np.array_equal(
        np.asarray(CLASS_NAMES)[predicted_ids], frame["predicted_label"]
    ):
        raise ValueError(f"Predicted label names are inconsistent in {source}")

    frame = frame.copy()
    frame["correct"] = normalize_correct(frame["correct"], source)
    expected_correct = true_ids == predicted_ids
    if not np.array_equal(expected_correct, frame["correct"].to_numpy()):
        raise ValueError(f"Correct flags are inconsistent in {source}")
    return frame.sort_values("source_index").reset_index(drop=True)


def verify_metric_summary(
    frame: pd.DataFrame,
    expected_accuracy: float,
    expected_macro_f1: float,
    source: Path,
):
    confusion = confusion_matrix_from_labels(
        frame["label_id"].to_numpy(dtype=np.int64),
        frame["predicted_label_id"].to_numpy(dtype=np.int64),
    )
    accuracy, macro_f1 = metrics_from_confusion(confusion)
    if not np.isclose(float(accuracy), expected_accuracy, rtol=0, atol=1e-12):
        raise ValueError(f"Accuracy does not match saved summary: {source}")
    if not np.isclose(float(macro_f1), expected_macro_f1, rtol=0, atol=1e-12):
        raise ValueError(f"Macro-F1 does not match saved summary: {source}")


def load_and_validate_inputs() -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict]:
    expected_files = (
        (BASELINE_PREDICTIONS, EXPECTED_BASELINE_PREDICTIONS_SHA256),
        (BASELINE_SUMMARY, EXPECTED_BASELINE_SUMMARY_SHA256),
        (FINAL_ARTIFACT_MANIFEST, EXPECTED_FINAL_ARTIFACT_MANIFEST_SHA256),
    )
    for path, expected_hash in expected_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen input hash mismatch: {path}")

    artifact_manifest = load_json(FINAL_ARTIFACT_MANIFEST)
    if artifact_manifest.get("evaluation_id") != FINAL_EVALUATION_ID:
        raise ValueError("Final artifact manifest evaluation ID is incorrect.")
    artifact_hashes = {
        item["path"]: item["sha256"] for item in artifact_manifest["files"]
    }

    summary_relative = "test_per_seed_summary.csv"
    expected_summary_hash = artifact_hashes.get(summary_relative)
    if not expected_summary_hash:
        raise ValueError("Final per-seed summary hash is missing from the manifest.")
    if sha256_file(FINAL_PER_SEED_SUMMARY) != expected_summary_hash:
        raise ValueError("Final per-seed summary hash mismatch.")

    baseline = validate_prediction_frame(
        pd.read_csv(BASELINE_PREDICTIONS), BASELINE_PREDICTIONS
    )
    baseline_summary = load_json(BASELINE_SUMMARY)
    verify_metric_summary(
        baseline,
        float(baseline_summary["accuracy"]),
        float(baseline_summary["macro_f1"]),
        BASELINE_SUMMARY,
    )

    final_summary = pd.read_csv(FINAL_PER_SEED_SUMMARY).set_index("seed")
    if set(final_summary.index.astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError("Unexpected seeds in final per-seed summary.")

    final_predictions = {}
    key_columns = [
        "source_index",
        "lotName",
        "waferIndex",
        "label",
        "split",
        "label_id",
    ]
    baseline_keys = baseline[key_columns]
    for seed in EXPECTED_SEEDS:
        filename = f"predictions_seed{seed}.csv"
        relative_path = f"predictions/{filename}"
        expected_hash = artifact_hashes.get(relative_path)
        if not expected_hash:
            raise ValueError(f"Prediction hash missing from manifest: {relative_path}")
        canonical_path = FINAL_PREDICTION_DIR / filename
        # Only the canonical files under 04_实验/metrics are paper evidence;
        # the 05_结果 working copies (seed-42 row 25832 corrupted) are not validated.
        if not canonical_path.is_file():
            raise FileNotFoundError(f"Missing final prediction file: {canonical_path}")
        if sha256_file(canonical_path) != expected_hash:
            raise ValueError(f"Final prediction hash mismatch: {canonical_path}")

        frame = validate_prediction_frame(pd.read_csv(canonical_path), canonical_path)
        if "true_label_id" not in frame.columns:
            raise ValueError(f"Missing true_label_id in {canonical_path}")
        if not np.array_equal(
            frame["true_label_id"].to_numpy(dtype=np.int64),
            frame["label_id"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"true_label_id is inconsistent in {canonical_path}")
        pd.testing.assert_frame_equal(
            baseline_keys,
            frame[key_columns],
            check_dtype=False,
            check_exact=True,
        )

        expected_row = final_summary.loc[seed]
        verify_metric_summary(
            frame,
            float(expected_row["accuracy"]),
            float(expected_row["macro_f1"]),
            FINAL_PER_SEED_SUMMARY,
        )
        final_predictions[seed] = frame

    input_hashes = {
        str(BASELINE_PREDICTIONS.relative_to(PROJECT_ROOT)): (
            EXPECTED_BASELINE_PREDICTIONS_SHA256
        ),
        str(BASELINE_SUMMARY.relative_to(PROJECT_ROOT)): (
            EXPECTED_BASELINE_SUMMARY_SHA256
        ),
        str(FINAL_ARTIFACT_MANIFEST.relative_to(PROJECT_ROOT)): (
            EXPECTED_FINAL_ARTIFACT_MANIFEST_SHA256
        ),
        str(FINAL_PER_SEED_SUMMARY.relative_to(PROJECT_ROOT)): expected_summary_hash,
    }
    for seed in EXPECTED_SEEDS:
        filename = f"predictions_seed{seed}.csv"
        relative_path = f"predictions/{filename}"
        prediction_hash = artifact_hashes[relative_path]
        input_hashes[
            str((FINAL_PREDICTION_DIR / filename).relative_to(PROJECT_ROOT))
        ] = prediction_hash

    return baseline, final_predictions, input_hashes


def confusion_matrix_from_labels(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("True and predicted labels must have the same shape.")
    encoded = true_labels * n_classes + predicted_labels
    return np.bincount(encoded, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )


def metrics_from_confusion(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confusion = np.asarray(confusion)
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    actual_support = confusion.sum(axis=-1)
    predicted_support = confusion.sum(axis=-2)
    denominator = actual_support + predicted_support
    class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    total = confusion.sum(axis=(-2, -1))
    accuracy = true_positive.sum(axis=-1) / total
    macro_f1 = class_f1.mean(axis=-1)
    return accuracy, macro_f1


def mcnemar_exact(
    baseline_correct: np.ndarray,
    final_correct: np.ndarray,
) -> dict:
    baseline_correct = np.asarray(baseline_correct, dtype=bool)
    final_correct = np.asarray(final_correct, dtype=bool)
    if baseline_correct.shape != final_correct.shape:
        raise ValueError("Paired correctness arrays must have the same shape.")

    both_correct = int(np.sum(baseline_correct & final_correct))
    baseline_only = int(np.sum(baseline_correct & ~final_correct))
    final_only = int(np.sum(~baseline_correct & final_correct))
    both_wrong = int(np.sum(~baseline_correct & ~final_correct))
    discordant = baseline_only + final_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(baseline_only, final_only),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return {
        "both_correct": both_correct,
        "baseline_only_correct": baseline_only,
        "final_only_correct": final_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values.")
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        ranked * (count - np.arange(count))
    )
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def paired_stratified_bootstrap(
    true_labels: np.ndarray,
    baseline_predictions: np.ndarray,
    final_predictions: np.ndarray,
    replicates: int,
    random_seed: int,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive.")
    true_labels = np.asarray(true_labels, dtype=np.int64)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    final_predictions = np.asarray(final_predictions, dtype=np.int64)
    if not (
        true_labels.shape
        == baseline_predictions.shape
        == final_predictions.shape
    ):
        raise ValueError("Paired label arrays must have the same shape.")

    rng = np.random.default_rng(random_seed)
    baseline_confusions = np.zeros(
        (replicates, n_classes, n_classes), dtype=np.int64
    )
    final_confusions = np.zeros_like(baseline_confusions)

    for true_class in range(n_classes):
        mask = true_labels == true_class
        support = int(mask.sum())
        if support == 0:
            raise ValueError(f"Class {true_class} has no samples.")
        joint_ids = (
            baseline_predictions[mask] * n_classes + final_predictions[mask]
        )
        joint_counts = np.bincount(
            joint_ids, minlength=n_classes**2
        ).reshape(n_classes, n_classes)
        draws = rng.multinomial(
            support,
            joint_counts.reshape(-1) / support,
            size=replicates,
        ).reshape(replicates, n_classes, n_classes)
        baseline_confusions[:, true_class, :] = draws.sum(axis=2)
        final_confusions[:, true_class, :] = draws.sum(axis=1)

    baseline_accuracy, baseline_macro_f1 = metrics_from_confusion(
        baseline_confusions
    )
    final_accuracy, final_macro_f1 = metrics_from_confusion(final_confusions)
    return (
        (final_accuracy - baseline_accuracy) * 100.0,
        (final_macro_f1 - baseline_macro_f1) * 100.0,
    )


def paired_macro_f1_randomization(
    true_labels: np.ndarray,
    baseline_predictions: np.ndarray,
    final_predictions: np.ndarray,
    replicates: int,
    random_seed: int,
    n_classes: int = N_CLASSES,
) -> tuple[np.ndarray, float, int]:
    if replicates <= 0:
        raise ValueError("Randomization replicate count must be positive.")
    true_labels = np.asarray(true_labels, dtype=np.int64)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    final_predictions = np.asarray(final_predictions, dtype=np.int64)
    if not (
        true_labels.shape
        == baseline_predictions.shape
        == final_predictions.shape
    ):
        raise ValueError("Paired label arrays must have the same shape.")

    observed_baseline = confusion_matrix_from_labels(
        true_labels, baseline_predictions, n_classes
    )
    observed_final = confusion_matrix_from_labels(
        true_labels, final_predictions, n_classes
    )
    _, observed_baseline_macro_f1 = metrics_from_confusion(observed_baseline)
    _, observed_final_macro_f1 = metrics_from_confusion(observed_final)
    observed_delta = float(
        (observed_final_macro_f1 - observed_baseline_macro_f1) * 100.0
    )

    rng = np.random.default_rng(random_seed)
    randomized_baseline = np.zeros(
        (replicates, n_classes, n_classes), dtype=np.int64
    )
    randomized_final = np.zeros_like(randomized_baseline)

    for true_class in range(n_classes):
        mask = true_labels == true_class
        joint_ids = (
            baseline_predictions[mask] * n_classes + final_predictions[mask]
        )
        joint_counts = np.bincount(
            joint_ids, minlength=n_classes**2
        ).reshape(n_classes, n_classes)
        for baseline_class in range(n_classes):
            for final_class in range(n_classes):
                count = int(joint_counts[baseline_class, final_class])
                if count == 0:
                    continue
                if baseline_class == final_class:
                    randomized_baseline[:, true_class, baseline_class] += count
                    randomized_final[:, true_class, final_class] += count
                    continue
                kept = rng.binomial(count, 0.5, size=replicates)
                swapped = count - kept
                randomized_baseline[:, true_class, baseline_class] += kept
                randomized_final[:, true_class, final_class] += kept
                randomized_baseline[:, true_class, final_class] += swapped
                randomized_final[:, true_class, baseline_class] += swapped

    _, randomized_baseline_macro_f1 = metrics_from_confusion(
        randomized_baseline
    )
    _, randomized_final_macro_f1 = metrics_from_confusion(randomized_final)
    randomized_delta = (
        randomized_final_macro_f1 - randomized_baseline_macro_f1
    ) * 100.0
    extreme_count = int(
        np.count_nonzero(np.abs(randomized_delta) >= abs(observed_delta) - 1e-12)
    )
    p_value = float((extreme_count + 1) / (replicates + 1))
    return randomized_delta, p_value, extreme_count


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail], method="linear")
    return float(lower), float(upper)


def build_observed_metrics(
    baseline: pd.DataFrame,
    final_predictions: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    true_labels = baseline["label_id"].to_numpy(dtype=np.int64)
    baseline_ids = baseline["predicted_label_id"].to_numpy(dtype=np.int64)
    baseline_confusion = confusion_matrix_from_labels(true_labels, baseline_ids)
    baseline_accuracy, baseline_macro_f1 = metrics_from_confusion(
        baseline_confusion
    )
    rows = []
    for seed in EXPECTED_SEEDS:
        final_ids = final_predictions[seed]["predicted_label_id"].to_numpy(
            dtype=np.int64
        )
        final_confusion = confusion_matrix_from_labels(true_labels, final_ids)
        final_accuracy, final_macro_f1 = metrics_from_confusion(final_confusion)
        rows.append(
            {
                "seed": seed,
                "test_samples": len(true_labels),
                "baseline_accuracy": float(baseline_accuracy),
                "final_accuracy": float(final_accuracy),
                "accuracy_delta_percentage_points": float(
                    (final_accuracy - baseline_accuracy) * 100.0
                ),
                "baseline_macro_f1": float(baseline_macro_f1),
                "final_macro_f1": float(final_macro_f1),
                "macro_f1_delta_percentage_points": float(
                    (final_macro_f1 - baseline_macro_f1) * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(
    baseline: pd.DataFrame,
    final_predictions: dict[int, pd.DataFrame],
) -> dict:
    observed = build_observed_metrics(baseline, final_predictions)
    true_labels = baseline["label_id"].to_numpy(dtype=np.int64)
    baseline_ids = baseline["predicted_label_id"].to_numpy(dtype=np.int64)
    baseline_correct = baseline["correct"].to_numpy(dtype=bool)

    mcnemar_rows = []
    bootstrap_summary_rows = []
    randomization_rows = []
    bootstrap_frames = []
    randomization_frames = []

    for seed in EXPECTED_SEEDS:
        final = final_predictions[seed]
        final_ids = final["predicted_label_id"].to_numpy(dtype=np.int64)
        final_correct = final["correct"].to_numpy(dtype=bool)
        observed_row = observed.loc[observed["seed"].eq(seed)].iloc[0]

        mcnemar_result = mcnemar_exact(baseline_correct, final_correct)
        mcnemar_result.update(
            {
                "seed": seed,
                "accuracy_delta_percentage_points": observed_row[
                    "accuracy_delta_percentage_points"
                ],
            }
        )
        mcnemar_rows.append(mcnemar_result)

        bootstrap_seed = BOOTSTRAP_RANDOM_SEED_BASE + seed
        accuracy_deltas, macro_f1_deltas = paired_stratified_bootstrap(
            true_labels,
            baseline_ids,
            final_ids,
            BOOTSTRAP_REPLICATES,
            bootstrap_seed,
        )
        accuracy_lower, accuracy_upper = percentile_interval(accuracy_deltas)
        macro_lower, macro_upper = percentile_interval(macro_f1_deltas)
        bootstrap_summary_rows.append(
            {
                "seed": seed,
                "replicates": BOOTSTRAP_REPLICATES,
                "random_seed": bootstrap_seed,
                "confidence_level": CONFIDENCE_LEVEL,
                "observed_accuracy_delta_percentage_points": observed_row[
                    "accuracy_delta_percentage_points"
                ],
                "accuracy_delta_bootstrap_mean": float(accuracy_deltas.mean()),
                "accuracy_delta_ci_lower": accuracy_lower,
                "accuracy_delta_ci_upper": accuracy_upper,
                "accuracy_superiority_probability": float(
                    np.mean(accuracy_deltas > 0)
                ),
                "observed_macro_f1_delta_percentage_points": observed_row[
                    "macro_f1_delta_percentage_points"
                ],
                "macro_f1_delta_bootstrap_mean": float(macro_f1_deltas.mean()),
                "macro_f1_delta_ci_lower": macro_lower,
                "macro_f1_delta_ci_upper": macro_upper,
                "macro_f1_superiority_probability": float(
                    np.mean(macro_f1_deltas > 0)
                ),
            }
        )
        bootstrap_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "replicate": np.arange(1, BOOTSTRAP_REPLICATES + 1),
                    "accuracy_delta_percentage_points": accuracy_deltas,
                    "macro_f1_delta_percentage_points": macro_f1_deltas,
                }
            )
        )

        randomization_seed = RANDOMIZATION_RANDOM_SEED_BASE + seed
        randomized_deltas, p_value, extreme_count = paired_macro_f1_randomization(
            true_labels,
            baseline_ids,
            final_ids,
            RANDOMIZATION_REPLICATES,
            randomization_seed,
        )
        randomization_rows.append(
            {
                "seed": seed,
                "replicates": RANDOMIZATION_REPLICATES,
                "random_seed": randomization_seed,
                "observed_macro_f1_delta_percentage_points": observed_row[
                    "macro_f1_delta_percentage_points"
                ],
                "extreme_replicates": extreme_count,
                "approximate_two_sided_p_value": p_value,
            }
        )
        randomization_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "replicate": np.arange(1, RANDOMIZATION_REPLICATES + 1),
                    "macro_f1_delta_percentage_points": randomized_deltas,
                }
            )
        )

    mcnemar = pd.DataFrame(mcnemar_rows).sort_values("seed").reset_index(drop=True)
    mcnemar["holm_adjusted_p_value"] = holm_adjust(
        mcnemar["exact_two_sided_p_value"].to_numpy()
    )
    mcnemar["significant_after_holm_0_05"] = (
        mcnemar["holm_adjusted_p_value"] < ALPHA
    )

    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows).sort_values(
        "seed"
    ).reset_index(drop=True)
    randomization = pd.DataFrame(randomization_rows).sort_values(
        "seed"
    ).reset_index(drop=True)
    randomization["holm_adjusted_p_value"] = holm_adjust(
        randomization["approximate_two_sided_p_value"].to_numpy()
    )
    randomization["significant_after_holm_0_05"] = (
        randomization["holm_adjusted_p_value"] < ALPHA
    )

    return {
        "observed": observed,
        "mcnemar": mcnemar,
        "bootstrap_summary": bootstrap_summary,
        "randomization": randomization,
        "bootstrap_replicates": pd.concat(bootstrap_frames, ignore_index=True),
        "randomization_replicates": pd.concat(
            randomization_frames, ignore_index=True
        ),
    }


def build_paper_table(analysis: dict) -> pd.DataFrame:
    observed = analysis["observed"]
    mcnemar = analysis["mcnemar"]
    bootstrap = analysis["bootstrap_summary"]
    randomization = analysis["randomization"]
    table = observed.merge(mcnemar, on=["seed", "accuracy_delta_percentage_points"])
    table = table.merge(bootstrap, on="seed")
    table = table.merge(
        randomization[
            [
                "seed",
                "approximate_two_sided_p_value",
                "holm_adjusted_p_value",
                "significant_after_holm_0_05",
            ]
        ].rename(
            columns={
                "holm_adjusted_p_value": "macro_f1_holm_adjusted_p_value",
                "significant_after_holm_0_05": (
                    "macro_f1_significant_after_holm_0_05"
                ),
            }
        ),
        on="seed",
    )
    return table[
        [
            "seed",
            "baseline_accuracy",
            "final_accuracy",
            "accuracy_delta_percentage_points",
            "accuracy_delta_ci_lower",
            "accuracy_delta_ci_upper",
            "exact_two_sided_p_value",
            "holm_adjusted_p_value",
            "significant_after_holm_0_05",
            "baseline_macro_f1",
            "final_macro_f1",
            "macro_f1_delta_percentage_points",
            "macro_f1_delta_ci_lower",
            "macro_f1_delta_ci_upper",
            "approximate_two_sided_p_value",
            "macro_f1_holm_adjusted_p_value",
            "macro_f1_significant_after_holm_0_05",
        ]
    ].rename(
        columns={
            "exact_two_sided_p_value": "accuracy_mcnemar_exact_p_value",
            "holm_adjusted_p_value": "accuracy_holm_adjusted_p_value",
            "significant_after_holm_0_05": (
                "accuracy_significant_after_holm_0_05"
            ),
            "approximate_two_sided_p_value": (
                "macro_f1_randomization_p_value"
            ),
        }
    )


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_effect_sizes(analysis: dict, output_png: Path, output_pdf: Path):
    configure_plot_style()
    observed = analysis["observed"].set_index("seed").loc[list(EXPECTED_SEEDS)]
    bootstrap = analysis["bootstrap_summary"].set_index("seed").loc[
        list(EXPECTED_SEEDS)
    ]
    y_positions = np.arange(len(EXPECTED_SEEDS), dtype=np.float64)
    offset = 0.12

    figure, axis = plt.subplots(figsize=(10.2, 5.4))
    metrics = (
        (
            "Accuracy",
            "accuracy_delta_percentage_points",
            "accuracy_delta_ci_lower",
            "accuracy_delta_ci_upper",
            "#7F8C8D",
            "s",
            -offset,
        ),
        (
            "Macro-F1",
            "macro_f1_delta_percentage_points",
            "macro_f1_delta_ci_lower",
            "macro_f1_delta_ci_upper",
            "#2A9D8F",
            "o",
            offset,
        ),
    )
    all_lower = []
    all_upper = []
    for label, observed_column, lower_column, upper_column, color, marker, shift in metrics:
        values = observed[observed_column].to_numpy(dtype=np.float64)
        lower = bootstrap[lower_column].to_numpy(dtype=np.float64)
        upper = bootstrap[upper_column].to_numpy(dtype=np.float64)
        all_lower.extend(lower)
        all_upper.extend(upper)
        axis.errorbar(
            values,
            y_positions + shift,
            xerr=np.vstack([values - lower, upper - values]),
            fmt=marker,
            markersize=7,
            capsize=4,
            linewidth=1.6,
            color=color,
            label=label,
        )

    lower_limit = min(0.0, min(all_lower))
    upper_limit = max(all_upper)
    span = max(upper_limit - lower_limit, 1.0)
    axis.set_xlim(lower_limit - span * 0.08, upper_limit + span * 0.12)
    axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
    axis.set_yticks(y_positions, [f"Seed {seed}" for seed in EXPECTED_SEEDS])
    axis.invert_yaxis()
    axis.set_xlabel("Final model minus ResNet18 (percentage points)")
    axis.set_title("Paired Improvement on the Fixed WM-811K Test Split")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    figure.text(
        0.5,
        0.018,
        (
            "Points are observed paired differences; error bars are 95% "
            "stratified paired-bootstrap intervals (10,000 replicates)."
        ),
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=[0, 0.055, 1, 1])
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def build_summary(analysis: dict) -> dict:
    observed = analysis["observed"]
    mcnemar = analysis["mcnemar"]
    bootstrap = analysis["bootstrap_summary"]
    randomization = analysis["randomization"]
    return {
        "analysis_id": ANALYSIS_ID,
        "source_evaluation_id": FINAL_EVALUATION_ID,
        "analysis_type": "post_test_paired_statistical_comparison",
        "new_model_inference_performed": False,
        "wafer_image_dataset_loaded": False,
        "saved_prediction_files_loaded": True,
        "test_predictions_reused": True,
        "baseline": "ResNet18 Baseline",
        "final_model": "HighRes ShuffleNetV2 + CrossEntropyLoss without ECA",
        "seeds": list(EXPECTED_SEEDS),
        "test_samples": EXPECTED_TEST_SAMPLES,
        "baseline_training_runs": 1,
        "final_training_runs": len(EXPECTED_SEEDS),
        "architecture_level_random_seed_inference": False,
        "alpha": ALPHA,
        "multiple_comparison_adjustment": "Holm correction across three seeds",
        "accuracy_test": "two-sided exact McNemar test",
        "effect_interval": {
            "method": "paired bootstrap stratified by true class",
            "confidence_level": CONFIDENCE_LEVEL,
            "replicates_per_seed": BOOTSTRAP_REPLICATES,
        },
        "macro_f1_test": {
            "method": "two-sided paired approximate randomization test",
            "replicates_per_seed": RANDOMIZATION_REPLICATES,
            "plus_one_correction": True,
        },
        "accuracy_delta_percentage_points": {
            str(int(row["seed"])): float(row["accuracy_delta_percentage_points"])
            for _, row in observed.iterrows()
        },
        "macro_f1_delta_percentage_points": {
            str(int(row["seed"])): float(row["macro_f1_delta_percentage_points"])
            for _, row in observed.iterrows()
        },
        "accuracy_all_seeds_significant_after_holm": bool(
            mcnemar["significant_after_holm_0_05"].all()
        ),
        "macro_f1_all_seeds_significant_after_holm": bool(
            randomization["significant_after_holm_0_05"].all()
        ),
        "accuracy_all_bootstrap_intervals_above_zero": bool(
            (bootstrap["accuracy_delta_ci_lower"] > 0).all()
        ),
        "macro_f1_all_bootstrap_intervals_above_zero": bool(
            (bootstrap["macro_f1_delta_ci_lower"] > 0).all()
        ),
        "interpretation_scope": (
            "Tests quantify paired prediction differences conditional on the "
            "saved trained models. Because only one ResNet18 training run is "
            "available, they do not estimate full architecture-level training-"
            "seed uncertainty. The fixed WM-811K benchmark split also had "
            "historical baseline access and is not an untouched external holdout."
        ),
    }


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, float_format="%.10f")


def write_outputs(
    output_root: Path,
    analysis: dict,
    input_hashes: dict[str, str],
):
    paths = output_paths(output_root)
    metrics_dir = paths["metrics_dir"]
    figure_dir = paths["figure_dir"]
    table_dir = paths["table_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    write_csv(analysis["observed"], metrics_dir / "paired_observed_metrics.csv")
    write_csv(analysis["mcnemar"], metrics_dir / "paired_accuracy_mcnemar.csv")
    write_csv(
        analysis["bootstrap_summary"],
        metrics_dir / "paired_bootstrap_summary.csv",
    )
    write_csv(
        analysis["randomization"],
        metrics_dir / "paired_macro_f1_randomization.csv",
    )
    write_csv(
        analysis["bootstrap_replicates"],
        metrics_dir / "bootstrap_replicates.csv",
    )
    write_csv(
        analysis["randomization_replicates"],
        metrics_dir / "macro_f1_randomization_replicates.csv",
    )

    summary = build_summary(analysis)
    (metrics_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paper_table = build_paper_table(analysis)
    write_csv(paper_table, table_dir / PAPER_TABLE_NAME)

    output_png = figure_dir / f"{FIGURE_STEM}.png"
    output_pdf = figure_dir / f"{FIGURE_STEM}.pdf"
    plot_effect_sizes(analysis, output_png, output_pdf)

    output_hashes = {}
    for path in expected_output_files(output_root):
        if path.name == "analysis_manifest.json":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Expected output was not created: {path}")
        output_hashes[str(path.relative_to(output_root))] = sha256_file(path)

    manifest = {
        "analysis_id": ANALYSIS_ID,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "new_model_inference_performed": False,
        "wafer_image_dataset_loaded": False,
        "saved_prediction_files_loaded": True,
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
    }
    (metrics_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_observed(observed: pd.DataFrame):
    print("Input validation: PASS")
    print(
        "No model, checkpoint, or wafer-image data was loaded; "
        "no inference was performed."
    )
    for _, row in observed.iterrows():
        print(
            f"Seed {int(row['seed'])}: accuracy delta "
            f"{row['accuracy_delta_percentage_points']:+.4f} pp, "
            f"Macro-F1 delta {row['macro_f1_delta_percentage_points']:+.4f} pp"
        )


def print_results(analysis: dict):
    mcnemar = analysis["mcnemar"].set_index("seed")
    bootstrap = analysis["bootstrap_summary"].set_index("seed")
    randomization = analysis["randomization"].set_index("seed")
    for seed in EXPECTED_SEEDS:
        print(
            f"Seed {seed}: McNemar Holm p="
            f"{mcnemar.loc[seed, 'holm_adjusted_p_value']:.6g}; "
            f"Macro-F1 95% CI ["
            f"{bootstrap.loc[seed, 'macro_f1_delta_ci_lower']:.4f}, "
            f"{bootstrap.loc[seed, 'macro_f1_delta_ci_upper']:.4f}] pp; "
            f"randomization Holm p="
            f"{randomization.loc[seed, 'holm_adjusted_p_value']:.6g}"
        )


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.run:
        ensure_outputs_are_available(output_root)

    baseline, final_predictions, input_hashes = load_and_validate_inputs()
    observed = build_observed_metrics(baseline, final_predictions)
    print_observed(observed)

    if args.check_inputs:
        print("Preflight complete. No statistical resampling was run and no files were created.")
        return

    analysis = run_analysis(baseline, final_predictions)
    print_results(analysis)
    write_outputs(output_root, analysis, input_hashes)
    paths = output_paths(output_root)
    print(f"Metrics saved to: {paths['metrics_dir']}")
    print(f"Figure saved to: {paths['figure_dir'] / (FIGURE_STEM + '.png')}")
    print(f"Paper table saved to: {paths['table_dir'] / PAPER_TABLE_NAME}")


if __name__ == "__main__":
    main()
