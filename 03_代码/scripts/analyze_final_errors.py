#!/usr/bin/env python3
"""Analyze saved final-test predictions without running model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_EVALUATION_ID = "20260729_highres_ce_multiseed_final_test"
ANALYSIS_ID = "20260729_highres_ce_multiseed_error_analysis"

CANONICAL_EVALUATION_DIR = (
    PROJECT_ROOT / "04_实验" / "metrics" / FINAL_EVALUATION_ID
)
PREDICTION_DIR = CANONICAL_EVALUATION_DIR / "predictions"
ARTIFACT_MANIFEST = CANONICAL_EVALUATION_DIR / "artifact_manifest.json"
FREEZE_MANIFEST = (
    PROJECT_ROOT / "00_项目管理" / "20260729_最终模型冻结清单.json"
)

EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "70d62a8d3f122871bedb609000dbde71d6238b5f69dcf75dfc587bb83643d33c"
)
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "c682922b1c291f03a41a81e5d15ebfe08e4456e77c816f7943041fc11fcef8db"
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

KEY_COLUMNS = (
    "source_index",
    "lotName",
    "waferIndex",
    "label",
    "split",
    "label_id",
    "true_label_id",
)

PROBABILITY_COLUMNS = tuple(
    f"prob_{class_id}_{class_name.lower().replace('-', '_')}"
    for class_id, class_name in enumerate(CLASS_NAMES)
)

REPRESENTATIVE_PAIRS = (
    ("Edge-Loc", "none"),
    ("Loc", "none"),
    ("Center", "none"),
    ("Scratch", "none"),
)

REPRESENTATIVE_QUANTILES = (
    ("25th percentile", 0.25),
    ("50th percentile", 0.50),
    ("75th percentile", 0.75),
)

METRIC_FILENAMES = (
    "per_seed_error_summary.csv",
    "error_stability_summary.csv",
    "error_overlap_exact.csv",
    "per_class_error_summary.csv",
    "error_pair_by_seed.csv",
    "error_pair_aggregate.csv",
    "sample_error_consensus.csv",
    "common_errors_all_seeds.csv",
    "unanimous_error_pairs.csv",
    "representative_error_samples.csv",
    "analysis_summary.json",
    "analysis_manifest.json",
)

PAPER_TABLE_FILENAMES = (
    "table_final_error_stability.csv",
    "table_final_error_by_class.csv",
    "table_final_error_pairs_multiseed.csv",
    "table_final_unanimous_error_pairs.csv",
)

FIGURE_STEMS = (
    "final_highres_ce_common_error_rate_by_class",
    "final_highres_ce_top_error_pairs",
    "final_highres_ce_unanimous_error_examples",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the saved three-seed final-test predictions. This script "
            "does not load a model and does not perform inference."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate frozen inputs and print expected error counts only.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Create error-analysis tables, figures, and an audit manifest.",
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
    metrics_dir = output_root / "04_实验" / "metrics" / ANALYSIS_ID
    figure_dir = output_root / "05_结果" / "figures" / "model_results"
    table_dir = output_root / "05_结果" / "tables"
    return {
        "metrics_dir": metrics_dir,
        "figure_dir": figure_dir,
        "table_dir": table_dir,
    }


def expected_output_files(output_root: Path) -> list[Path]:
    paths = output_paths(output_root)
    files = [paths["metrics_dir"] / name for name in METRIC_FILENAMES]
    files.extend(paths["table_dir"] / name for name in PAPER_TABLE_FILENAMES)
    for stem in FIGURE_STEMS:
        files.append(paths["figure_dir"] / f"{stem}.png")
        files.append(paths["figure_dir"] / f"{stem}.pdf")
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
            "Refusing to overwrite existing error-analysis outputs:\n"
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


def validate_artifact_manifests() -> tuple[dict, dict, dict[str, str]]:
    if sha256_file(ARTIFACT_MANIFEST) != EXPECTED_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("Final-test artifact manifest hash has changed.")
    if sha256_file(FREEZE_MANIFEST) != EXPECTED_FREEZE_MANIFEST_SHA256:
        raise ValueError("Final model freeze manifest hash has changed.")

    artifact_manifest = load_json(ARTIFACT_MANIFEST)
    freeze_manifest = load_json(FREEZE_MANIFEST)

    if artifact_manifest.get("evaluation_id") != FINAL_EVALUATION_ID:
        raise ValueError("Artifact manifest evaluation ID is incorrect.")
    if freeze_manifest.get("final_evaluation_id") != FINAL_EVALUATION_ID:
        raise ValueError("Freeze manifest evaluation ID is incorrect.")

    artifact_hashes = {
        item["path"]: item["sha256"] for item in artifact_manifest["files"]
    }
    return artifact_manifest, freeze_manifest, artifact_hashes


def validate_prediction_file(
    path: Path,
    expected_hash: str,
) -> pd.DataFrame:
    """Validate one canonical prediction file against the artifact-manifest hash.

    The 05_结果/predictions working copies are intentionally NOT validated:
    the seed-42 working copy is known to contain a textually corrupted row
    (row 25832), and the canonical files under 04_实验/metrics are the paper
    evidence.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Prediction CSV hash mismatch: {path}")

    frame = pd.read_csv(path)
    required_columns = set(KEY_COLUMNS) | {
        "predicted_label_id",
        "predicted_label",
        "confidence",
        "correct",
    } | set(PROBABILITY_COLUMNS)
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    if len(frame) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} predictions in {path}, "
            f"found {len(frame)}."
        )
    if frame["source_index"].duplicated().any():
        raise ValueError(f"Duplicate source_index values in {path}")
    if not frame["source_index"].is_monotonic_increasing:
        raise ValueError(f"source_index order changed in {path}")
    if set(frame["label"].astype(str)) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected class names in {path}")
    if not frame["split"].eq("test").all():
        raise ValueError(f"Non-test rows found in {path}")

    probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities found in {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"Probabilities do not sum to one in {path}")

    predicted_ids = probabilities.argmax(axis=1)
    recorded_ids = frame["predicted_label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(predicted_ids, recorded_ids):
        raise ValueError(f"Predicted IDs are inconsistent in {path}")

    expected_names = np.asarray(CLASS_NAMES)[recorded_ids]
    if not np.array_equal(expected_names, frame["predicted_label"].astype(str)):
        raise ValueError(f"Predicted labels are inconsistent in {path}")

    recorded_confidence = frame["confidence"].to_numpy(dtype=np.float64)
    if not np.allclose(
        probabilities.max(axis=1),
        recorded_confidence,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(f"Confidence values are inconsistent in {path}")

    label_ids = frame["label_id"].to_numpy(dtype=np.int64)
    true_label_ids = frame["true_label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(label_ids, true_label_ids):
        raise ValueError(f"True label IDs are inconsistent in {path}")
    if not np.array_equal(np.asarray(CLASS_NAMES)[label_ids], frame["label"]):
        raise ValueError(f"True class names are inconsistent in {path}")

    frame["correct"] = normalize_correct(frame["correct"], path)
    expected_correct = label_ids == recorded_ids
    if not np.array_equal(expected_correct, frame["correct"].to_numpy()):
        raise ValueError(f"Correct flags are inconsistent in {path}")
    return frame


def load_and_validate_predictions(
    artifact_hashes: dict[str, str],
) -> dict[int, pd.DataFrame]:
    predictions = {}
    reference_keys = None

    for seed in EXPECTED_SEEDS:
        filename = f"predictions_seed{seed}.csv"
        relative_path = f"predictions/{filename}"
        if relative_path not in artifact_hashes:
            raise ValueError(f"Prediction hash missing from manifest: {relative_path}")

        frame = validate_prediction_file(
            PREDICTION_DIR / filename,
            artifact_hashes[relative_path],
        )
        current_keys = frame[list(KEY_COLUMNS)].reset_index(drop=True)
        if reference_keys is None:
            reference_keys = current_keys
        else:
            pd.testing.assert_frame_equal(
                reference_keys,
                current_keys,
                check_dtype=False,
                check_exact=True,
            )
        predictions[seed] = frame

    return predictions


def validate_dataset(
    freeze_manifest: dict,
    reference_prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, np.memmap]:
    dataset_info = freeze_manifest["dataset"]
    image_path = PROJECT_ROOT / dataset_info["processed_images"]
    metadata_path = PROJECT_ROOT / dataset_info["metadata"]

    for path, expected_hash in (
        (image_path, dataset_info["processed_images_sha256"]),
        (metadata_path, dataset_info["metadata_sha256"]),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed dataset file: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Processed dataset hash mismatch: {path}")

    images = np.load(image_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    if images.shape != (len(metadata), 64, 64):
        raise ValueError(
            f"Image/metadata geometry mismatch: {images.shape}, {len(metadata)} rows"
        )
    if images.dtype != np.uint8:
        raise ValueError(f"Unexpected processed image dtype: {images.dtype}")
    if metadata["source_index"].duplicated().any():
        raise ValueError("metadata.csv contains duplicate source_index values.")

    test_metadata = metadata.loc[metadata["split"] == "test"].copy()
    test_metadata = test_metadata.reset_index(names="image_row")
    if len(test_metadata) != EXPECTED_TEST_SAMPLES:
        raise ValueError("Processed test metadata sample count is incorrect.")

    metadata_keys = test_metadata[
        ["source_index", "lotName", "waferIndex", "label", "split", "label_id"]
    ].reset_index(drop=True)
    prediction_keys = reference_prediction[
        ["source_index", "lotName", "waferIndex", "label", "split", "label_id"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        metadata_keys,
        prediction_keys,
        check_dtype=False,
        check_exact=True,
    )

    return test_metadata, images


def combine_predictions(
    predictions: dict[int, pd.DataFrame],
    test_metadata: pd.DataFrame,
) -> pd.DataFrame:
    first_seed = EXPECTED_SEEDS[0]
    combined = predictions[first_seed][list(KEY_COLUMNS)].copy()

    image_rows = test_metadata[["source_index", "image_row"]]
    combined = combined.merge(
        image_rows,
        on="source_index",
        how="left",
        validate="one_to_one",
    )
    if combined["image_row"].isna().any():
        raise ValueError("Some predictions could not be mapped to processed images.")
    combined["image_row"] = combined["image_row"].astype(np.int64)

    for seed in EXPECTED_SEEDS:
        frame = predictions[seed]
        seed_columns = frame[
            ["source_index", "predicted_label", "confidence", "correct"]
        ].rename(
            columns={
                "predicted_label": f"predicted_label_seed{seed}",
                "confidence": f"confidence_seed{seed}",
                "correct": f"correct_seed{seed}",
            }
        )
        combined = combined.merge(
            seed_columns,
            on="source_index",
            how="left",
            validate="one_to_one",
        )

    return combined


def build_sample_consensus(combined: pd.DataFrame) -> pd.DataFrame:
    sample_summary = combined.copy()
    correct_columns = [f"correct_seed{seed}" for seed in EXPECTED_SEEDS]
    prediction_columns = [
        f"predicted_label_seed{seed}" for seed in EXPECTED_SEEDS
    ]
    confidence_columns = [f"confidence_seed{seed}" for seed in EXPECTED_SEEDS]

    sample_summary["wrong_seed_count"] = (
        ~sample_summary[correct_columns].astype(bool)
    ).sum(axis=1)
    sample_summary["all_seeds_wrong"] = (
        sample_summary["wrong_seed_count"] == len(EXPECTED_SEEDS)
    )
    sample_summary["unanimous_prediction"] = (
        sample_summary[prediction_columns].nunique(axis=1) == 1
    )
    sample_summary["unanimous_wrong"] = (
        sample_summary["all_seeds_wrong"]
        & sample_summary["unanimous_prediction"]
    )
    sample_summary["unanimous_predicted_label"] = np.where(
        sample_summary["unanimous_prediction"],
        sample_summary[prediction_columns[0]],
        "",
    )
    sample_summary["mean_prediction_confidence"] = sample_summary[
        confidence_columns
    ].mean(axis=1)

    return sample_summary.sort_values(
        ["wrong_seed_count", "mean_prediction_confidence", "source_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_per_seed_summary(
    predictions: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for seed in EXPECTED_SEEDS:
        frame = predictions[seed]
        errors = frame.loc[~frame["correct"]]
        rows.append(
            {
                "seed": seed,
                "test_samples": len(frame),
                "correct_predictions": int(frame["correct"].sum()),
                "error_count": int(len(errors)),
                "error_rate_percent": float((~frame["correct"]).mean() * 100),
                "accuracy_percent": float(frame["correct"].mean() * 100),
                "mean_error_confidence": float(errors["confidence"].mean()),
                "median_error_confidence": float(errors["confidence"].median()),
                "high_confidence_errors_ge_0_9": int(
                    (errors["confidence"] >= 0.9).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_stability_summary(sample_summary: pd.DataFrame) -> pd.DataFrame:
    counts = sample_summary["wrong_seed_count"].value_counts().reindex(
        range(len(EXPECTED_SEEDS) + 1), fill_value=0
    )
    labels = {
        0: "correct_by_all_three_seeds",
        1: "wrong_by_one_seed",
        2: "wrong_by_two_seeds",
        3: "wrong_by_all_three_seeds",
    }
    rows = []
    for wrong_seed_count, count in counts.items():
        rows.append(
            {
                "wrong_seed_count": int(wrong_seed_count),
                "category": labels[int(wrong_seed_count)],
                "sample_count": int(count),
                "sample_percent": float(count / len(sample_summary) * 100),
            }
        )
    return pd.DataFrame(rows)


def build_exact_overlap_summary(sample_summary: pd.DataFrame) -> pd.DataFrame:
    wrong_columns = []
    overlap_frame = sample_summary[["source_index"]].copy()
    for seed in EXPECTED_SEEDS:
        column = f"wrong_seed{seed}"
        overlap_frame[column] = ~sample_summary[f"correct_seed{seed}"].astype(bool)
        wrong_columns.append(column)

    grouped = (
        overlap_frame.groupby(wrong_columns, sort=False)
        .size()
        .rename("sample_count")
        .reset_index()
    )
    grouped["wrong_seed_count"] = grouped[wrong_columns].sum(axis=1)
    grouped["sample_percent"] = grouped["sample_count"] / len(sample_summary) * 100
    grouped["overlap_label"] = [
        "all_correct"
        if not wrong_seeds
        else "wrong_seed" + "_seed".join(str(seed) for seed in wrong_seeds)
        for wrong_seeds in (
            [
                seed
                for seed, column in zip(EXPECTED_SEEDS, wrong_columns)
                if bool(row[column])
            ]
            for _, row in grouped.iterrows()
        )
    ]
    return grouped.sort_values(
        ["wrong_seed_count", "overlap_label"],
        ascending=[True, True],
    ).reset_index(drop=True)


def build_per_class_summary(
    predictions: dict[int, pd.DataFrame],
    sample_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_samples = sample_summary.loc[sample_summary["label"] == class_name]
        support = len(class_samples)
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "support_per_seed": support,
        }
        error_counts = []
        error_rates = []
        for seed in EXPECTED_SEEDS:
            frame = predictions[seed]
            class_frame = frame.loc[frame["label"] == class_name]
            error_count = int((~class_frame["correct"]).sum())
            error_rate = float(error_count / support * 100)
            row[f"error_count_seed{seed}"] = error_count
            row[f"error_rate_percent_seed{seed}"] = error_rate
            error_counts.append(error_count)
            error_rates.append(error_rate)

        row["error_count_mean"] = float(np.mean(error_counts))
        row["error_count_sample_std"] = float(np.std(error_counts, ddof=1))
        row["error_rate_percent_mean"] = float(np.mean(error_rates))
        row["error_rate_percent_sample_std"] = float(
            np.std(error_rates, ddof=1)
        )
        row["recall_percent_mean"] = 100.0 - row["error_rate_percent_mean"]
        row["common_error_count_all_seeds"] = int(
            class_samples["all_seeds_wrong"].sum()
        )
        row["common_error_rate_percent"] = float(
            row["common_error_count_all_seeds"] / support * 100
        )
        row["unanimous_wrong_count"] = int(
            class_samples["unanimous_wrong"].sum()
        )
        row["unanimous_wrong_rate_percent"] = float(
            row["unanimous_wrong_count"] / support * 100
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_error_pair_tables(
    predictions: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in EXPECTED_SEEDS:
        frame = predictions[seed]
        supports = frame["label"].value_counts().to_dict()
        errors = frame.loc[~frame["correct"]]
        grouped = errors.groupby(["label", "predicted_label"], sort=False)
        grouped_counts = grouped.size().to_dict()
        grouped_confidence = grouped["confidence"].mean().to_dict()

        for true_class in CLASS_NAMES:
            for predicted_class in CLASS_NAMES:
                if true_class == predicted_class:
                    continue
                pair = (true_class, predicted_class)
                count = int(grouped_counts.get(pair, 0))
                rows.append(
                    {
                        "seed": seed,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "true_class_support": int(supports[true_class]),
                        "error_count": count,
                        "true_class_error_rate_percent": float(
                            count / supports[true_class] * 100
                        ),
                        "mean_error_confidence": float(
                            grouped_confidence.get(pair, np.nan)
                        ),
                    }
                )

    by_seed = pd.DataFrame(rows)
    aggregate_rows = []
    for (true_class, predicted_class), group in by_seed.groupby(
        ["true_class", "predicted_class"], sort=False
    ):
        counts = group["error_count"].to_numpy(dtype=np.float64)
        rates = group["true_class_error_rate_percent"].to_numpy(dtype=np.float64)
        nonzero_confidences = group["mean_error_confidence"].dropna()
        aggregate_rows.append(
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "true_class_support_per_seed": int(
                    group["true_class_support"].iloc[0]
                ),
                "error_count_mean": float(counts.mean()),
                "error_count_sample_std": float(counts.std(ddof=1)),
                "error_count_min": int(counts.min()),
                "error_count_max": int(counts.max()),
                "true_class_error_rate_percent_mean": float(rates.mean()),
                "true_class_error_rate_percent_sample_std": float(
                    rates.std(ddof=1)
                ),
                "mean_error_confidence_across_seeds": float(
                    nonzero_confidences.mean()
                    if len(nonzero_confidences)
                    else np.nan
                ),
            }
        )

    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["error_count_mean", "true_class", "predicted_class"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return by_seed, aggregate


def build_unanimous_error_pairs(
    sample_summary: pd.DataFrame,
) -> pd.DataFrame:
    unanimous_errors = sample_summary.loc[sample_summary["unanimous_wrong"]]
    supports = sample_summary["label"].value_counts().to_dict()
    pairs = (
        unanimous_errors.groupby(
            ["label", "unanimous_predicted_label"], sort=False
        )
        .agg(
            sample_count=("source_index", "size"),
            distinct_lot_count=("lotName", "nunique"),
            mean_prediction_confidence=("mean_prediction_confidence", "mean"),
        )
        .reset_index()
        .rename(
            columns={
                "label": "true_class",
                "unanimous_predicted_label": "predicted_class",
            }
        )
    )
    pairs["percent_of_unanimous_errors"] = (
        pairs["sample_count"] / len(unanimous_errors) * 100
    )
    pairs["percent_of_true_class_support"] = [
        count / supports[true_class] * 100
        for true_class, count in zip(pairs["true_class"], pairs["sample_count"])
    ]
    return pairs.sort_values(
        ["sample_count", "true_class", "predicted_class"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def select_representative_samples(
    sample_summary: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = []
    for true_class, predicted_class in REPRESENTATIVE_PAIRS:
        group = sample_summary.loc[
            sample_summary["unanimous_wrong"]
            & sample_summary["label"].eq(true_class)
            & sample_summary["unanimous_predicted_label"].eq(predicted_class)
        ].sort_values(
            ["mean_prediction_confidence", "source_index"],
            ascending=[True, True],
        )
        if len(group) < len(REPRESENTATIVE_QUANTILES):
            raise ValueError(
                f"Not enough unanimous errors for {true_class} -> "
                f"{predicted_class}: {len(group)}"
            )

        used_positions = set()
        for quantile_name, quantile in REPRESENTATIVE_QUANTILES:
            position = int(round(quantile * (len(group) - 1)))
            if position in used_positions:
                for candidate in range(len(group)):
                    if candidate not in used_positions:
                        position = candidate
                        break
            used_positions.add(position)
            row = group.iloc[position].copy()
            row["selection_quantile"] = quantile
            row["selection_label"] = quantile_name
            row["pair_sample_count"] = len(group)
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows)
    pair_order = {pair: index for index, pair in enumerate(REPRESENTATIVE_PAIRS)}
    quantile_order = {
        label: index for index, (label, _) in enumerate(REPRESENTATIVE_QUANTILES)
    }
    selected["pair_order"] = [
        pair_order[(true_class, predicted_class)]
        for true_class, predicted_class in zip(
            selected["label"], selected["unanimous_predicted_label"]
        )
    ]
    selected["quantile_order"] = selected["selection_label"].map(quantile_order)
    return selected.sort_values(
        ["pair_order", "quantile_order"]
    ).reset_index(drop=True)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, figure_dir: Path, stem: str):
    figure.savefig(
        figure_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        figure_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_common_error_rate_by_class(
    per_class: pd.DataFrame,
    figure_dir: Path,
):
    plot_data = per_class.sort_values(
        "common_error_rate_percent", ascending=True
    ).reset_index(drop=True)
    colors = [
        "#4C78A8" if class_name == "none" else "#D55E00"
        for class_name in plot_data["class_name"]
    ]

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    bars = axis.barh(
        plot_data["class_name"],
        plot_data["common_error_rate_percent"],
        color=colors,
    )
    axis.set_title("Samples Misclassified by All Three Random Seeds")
    axis.set_xlabel("Common error rate within each true class (%)")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)

    maximum = max(float(plot_data["common_error_rate_percent"].max()), 1.0)
    axis.set_xlim(0, maximum * 1.28)
    for bar, (_, row) in zip(bars, plot_data.iterrows()):
        axis.text(
            bar.get_width() + maximum * 0.02,
            bar.get_y() + bar.get_height() / 2,
            (
                f"{int(row['common_error_count_all_seeds'])}/"
                f"{int(row['support_per_seed'])}"
            ),
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    save_figure(
        figure,
        figure_dir,
        "final_highres_ce_common_error_rate_by_class",
    )


def plot_top_error_pairs(
    error_pairs: pd.DataFrame,
    figure_dir: Path,
):
    plot_data = error_pairs.head(10).copy().sort_values(
        "error_count_mean", ascending=True
    )
    plot_data["pair"] = (
        plot_data["true_class"] + "  ->  " + plot_data["predicted_class"]
    )
    colors = [
        "#D55E00" if predicted == "none" else "#4C78A8"
        for predicted in plot_data["predicted_class"]
    ]

    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    bars = axis.barh(
        plot_data["pair"],
        plot_data["error_count_mean"],
        xerr=plot_data["error_count_sample_std"],
        capsize=3,
        color=colors,
        error_kw={"elinewidth": 1, "ecolor": "#333333"},
    )
    axis.set_title("Top Misclassification Directions Across Three Seeds")
    axis.set_xlabel("Mean number of errors per seed (error bar: sample SD)")
    axis.grid(axis="x", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)

    maximum = max(
        float(
            (
                plot_data["error_count_mean"]
                + plot_data["error_count_sample_std"]
            ).max()
        ),
        1.0,
    )
    axis.set_xlim(0, maximum * 1.18)
    for bar, value, deviation in zip(
        bars,
        plot_data["error_count_mean"],
        plot_data["error_count_sample_std"],
    ):
        axis.text(
            value + deviation + maximum * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=9,
        )

    axis.legend(
        handles=[
            Patch(color="#D55E00", label="Defect predicted as none"),
            Patch(color="#4C78A8", label="Other class confusion"),
        ],
        loc="lower right",
        frameon=False,
    )
    figure.tight_layout()
    save_figure(
        figure,
        figure_dir,
        "final_highres_ce_top_error_pairs",
    )


def plot_representative_errors(
    selected: pd.DataFrame,
    images: np.memmap,
    figure_dir: Path,
):
    color_map = ListedColormap(["#FFFFFF", "#B8C2CC", "#D1495B"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], color_map.N)
    figure, axes = plt.subplots(4, 3, figsize=(9.2, 10.8))

    for row_index, (true_class, predicted_class) in enumerate(
        REPRESENTATIVE_PAIRS
    ):
        pair_rows = selected.loc[
            selected["label"].eq(true_class)
            & selected["unanimous_predicted_label"].eq(predicted_class)
        ].sort_values("quantile_order")
        for column_index, (_, row) in enumerate(pair_rows.iterrows()):
            axis = axes[row_index, column_index]
            image = images[int(row["image_row"])]
            axis.imshow(
                image,
                cmap=color_map,
                norm=norm,
                interpolation="nearest",
            )
            title_lines = []
            if row_index == 0:
                title_lines.append(str(row["selection_label"]))
            title_lines.extend(
                [
                    f"source {int(row['source_index'])}",
                    f"mean confidence {row['mean_prediction_confidence']:.3f}",
                ]
            )
            axis.set_title("\n".join(title_lines), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#9CA3AF")
                spine.set_linewidth(0.7)

        axes[row_index, 0].text(
            -0.20,
            0.5,
            f"{true_class}\n-> {predicted_class}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    figure.suptitle(
        "Representative Unanimous Errors Across Three Random Seeds",
        fontsize=14,
        y=0.995,
    )
    figure.legend(
        handles=[
            Patch(facecolor="#FFFFFF", edgecolor="#9CA3AF", label="0: outside"),
            Patch(facecolor="#B8C2CC", label="1: pass die"),
            Patch(facecolor="#D1495B", label="2: fail die"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=[0.10, 0.045, 1, 0.97])
    save_figure(
        figure,
        figure_dir,
        "final_highres_ce_unanimous_error_examples",
    )


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, float_format="%.8f")


def build_summary(
    per_seed: pd.DataFrame,
    stability: pd.DataFrame,
    exact_overlap: pd.DataFrame,
    sample_summary: pd.DataFrame,
    unanimous_pairs: pd.DataFrame,
) -> dict:
    stability_counts = {
        str(int(row["wrong_seed_count"])): int(row["sample_count"])
        for _, row in stability.iterrows()
    }
    exact_overlap_counts = {
        str(row["overlap_label"]): int(row["sample_count"])
        for _, row in exact_overlap.iterrows()
    }
    top_pairs = []
    for _, row in unanimous_pairs.head(10).iterrows():
        top_pairs.append(
            {
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "sample_count": int(row["sample_count"]),
            }
        )
    return {
        "analysis_id": ANALYSIS_ID,
        "source_evaluation_id": FINAL_EVALUATION_ID,
        "analysis_type": "post_test_descriptive_error_analysis",
        "test_inference_performed": False,
        "model_or_threshold_changes_permitted": False,
        "seeds": list(EXPECTED_SEEDS),
        "test_samples_per_seed": EXPECTED_TEST_SAMPLES,
        "errors_per_seed": {
            str(int(row["seed"])): int(row["error_count"])
            for _, row in per_seed.iterrows()
        },
        "samples_by_wrong_seed_count": stability_counts,
        "samples_by_exact_seed_overlap": exact_overlap_counts,
        "samples_wrong_at_least_one_seed": int(
            (sample_summary["wrong_seed_count"] >= 1).sum()
        ),
        "samples_wrong_all_three_seeds": int(
            sample_summary["all_seeds_wrong"].sum()
        ),
        "samples_unanimously_wrong_same_class": int(
            sample_summary["unanimous_wrong"].sum()
        ),
        "all_three_wrong_but_predictions_differ": int(
            (
                sample_summary["all_seeds_wrong"]
                & ~sample_summary["unanimous_prediction"]
            ).sum()
        ),
        "representative_sample_rule": (
            "For four prespecified unanimous defect-to-none pairs, sort by "
            "mean confidence across seeds and select the 25th, 50th, and "
            "75th percentile rows with source_index as the deterministic tie-breaker."
        ),
        "top_unanimous_error_pairs": top_pairs,
    }


def write_analysis_outputs(
    output_root: Path,
    predictions: dict[int, pd.DataFrame],
    images: np.memmap,
    per_seed: pd.DataFrame,
    stability: pd.DataFrame,
    exact_overlap: pd.DataFrame,
    per_class: pd.DataFrame,
    pair_by_seed: pd.DataFrame,
    pair_aggregate: pd.DataFrame,
    sample_summary: pd.DataFrame,
    unanimous_pairs: pd.DataFrame,
    selected_samples: pd.DataFrame,
    input_hashes: dict[str, str],
):
    paths = output_paths(output_root)
    metrics_dir = paths["metrics_dir"]
    figure_dir = paths["figure_dir"]
    table_dir = paths["table_dir"]

    metrics_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    common_errors = sample_summary.loc[sample_summary["all_seeds_wrong"]].copy()

    write_csv(per_seed, metrics_dir / "per_seed_error_summary.csv")
    write_csv(stability, metrics_dir / "error_stability_summary.csv")
    write_csv(exact_overlap, metrics_dir / "error_overlap_exact.csv")
    write_csv(per_class, metrics_dir / "per_class_error_summary.csv")
    write_csv(pair_by_seed, metrics_dir / "error_pair_by_seed.csv")
    write_csv(pair_aggregate, metrics_dir / "error_pair_aggregate.csv")
    write_csv(sample_summary, metrics_dir / "sample_error_consensus.csv")
    write_csv(common_errors, metrics_dir / "common_errors_all_seeds.csv")
    write_csv(unanimous_pairs, metrics_dir / "unanimous_error_pairs.csv")
    write_csv(selected_samples, metrics_dir / "representative_error_samples.csv")

    summary = build_summary(
        per_seed,
        stability,
        exact_overlap,
        sample_summary,
        unanimous_pairs,
    )
    (metrics_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_csv(stability, table_dir / "table_final_error_stability.csv")
    write_csv(per_class, table_dir / "table_final_error_by_class.csv")
    write_csv(
        pair_aggregate.head(12),
        table_dir / "table_final_error_pairs_multiseed.csv",
    )
    write_csv(
        unanimous_pairs.head(12),
        table_dir / "table_final_unanimous_error_pairs.csv",
    )

    configure_plot_style()
    plot_common_error_rate_by_class(per_class, figure_dir)
    plot_top_error_pairs(pair_aggregate, figure_dir)
    plot_representative_errors(selected_samples, images, figure_dir)

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
        "source_evaluation_id": FINAL_EVALUATION_ID,
        "test_inference_performed": False,
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
    }
    (metrics_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_analysis():
    _, freeze_manifest, artifact_hashes = validate_artifact_manifests()
    predictions = load_and_validate_predictions(artifact_hashes)
    test_metadata, images = validate_dataset(
        freeze_manifest,
        predictions[EXPECTED_SEEDS[0]],
    )
    combined = combine_predictions(predictions, test_metadata)
    sample_summary = build_sample_consensus(combined)
    per_seed = build_per_seed_summary(predictions)
    stability = build_stability_summary(sample_summary)
    exact_overlap = build_exact_overlap_summary(sample_summary)
    per_class = build_per_class_summary(predictions, sample_summary)
    pair_by_seed, pair_aggregate = build_error_pair_tables(predictions)
    unanimous_pairs = build_unanimous_error_pairs(sample_summary)
    selected_samples = select_representative_samples(sample_summary)

    dataset_info = freeze_manifest["dataset"]
    input_hashes = {
        str(ARTIFACT_MANIFEST.relative_to(PROJECT_ROOT)): (
            EXPECTED_ARTIFACT_MANIFEST_SHA256
        ),
        str(FREEZE_MANIFEST.relative_to(PROJECT_ROOT)): (
            EXPECTED_FREEZE_MANIFEST_SHA256
        ),
        dataset_info["processed_images"]: dataset_info["processed_images_sha256"],
        dataset_info["metadata"]: dataset_info["metadata_sha256"],
    }
    for seed in EXPECTED_SEEDS:
        relative_path = f"predictions/predictions_seed{seed}.csv"
        input_hashes[
            str((PREDICTION_DIR / f"predictions_seed{seed}.csv").relative_to(PROJECT_ROOT))
        ] = artifact_hashes[relative_path]

    return {
        "predictions": predictions,
        "images": images,
        "per_seed": per_seed,
        "stability": stability,
        "exact_overlap": exact_overlap,
        "per_class": per_class,
        "pair_by_seed": pair_by_seed,
        "pair_aggregate": pair_aggregate,
        "sample_summary": sample_summary,
        "unanimous_pairs": unanimous_pairs,
        "selected_samples": selected_samples,
        "input_hashes": input_hashes,
    }


def print_summary(analysis: dict):
    per_seed = analysis["per_seed"]
    stability = analysis["stability"]
    sample_summary = analysis["sample_summary"]

    print("Input validation: PASS")
    print("No model or checkpoint was loaded; no inference was performed.")
    print(f"Samples per seed: {EXPECTED_TEST_SAMPLES:,}")
    for _, row in per_seed.iterrows():
        print(
            f"Seed {int(row['seed'])}: {int(row['error_count'])} errors, "
            f"accuracy {row['accuracy_percent']:.4f}%"
        )
    print("Cross-seed error stability:")
    for _, row in stability.iterrows():
        print(
            f"  Wrong by {int(row['wrong_seed_count'])} seed(s): "
            f"{int(row['sample_count']):,} samples"
        )
    print(
        "  Wrong by all three with the same predicted class: "
        f"{int(sample_summary['unanimous_wrong'].sum()):,} samples"
    )


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    if args.run:
        ensure_outputs_are_available(output_root)

    analysis = prepare_analysis()
    print_summary(analysis)

    if args.check_inputs:
        print("Preflight complete. No files were created.")
        return

    write_analysis_outputs(
        output_root=output_root,
        predictions=analysis["predictions"],
        images=analysis["images"],
        per_seed=analysis["per_seed"],
        stability=analysis["stability"],
        exact_overlap=analysis["exact_overlap"],
        per_class=analysis["per_class"],
        pair_by_seed=analysis["pair_by_seed"],
        pair_aggregate=analysis["pair_aggregate"],
        sample_summary=analysis["sample_summary"],
        unanimous_pairs=analysis["unanimous_pairs"],
        selected_samples=analysis["selected_samples"],
        input_hashes=analysis["input_hashes"],
    )
    paths = output_paths(output_root)
    print(f"Metrics saved to: {paths['metrics_dir']}")
    print(f"Figures saved to: {paths['figure_dir']}")
    print(f"Paper tables saved to: {paths['table_dir']}")


if __name__ == "__main__":
    main()
