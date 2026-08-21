#!/usr/bin/env python3
"""Corrected paired post-test analysis for Kang stacking vs HighRes.

This is a post-hoc analysis of the already frozen test predictions.  It does
not load wafer images or checkpoints.  The original 20260810 analysis is kept
untouched; this version writes to a new, independent output namespace and
corrects the label of the randomization extreme-count field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KANG_DIR = PROJECT_ROOT / "04_实验/metrics/20260807_kang_kang_stacking_lot_reproduction"
HIGHRES_DIR = PROJECT_ROOT / "05_结果/predictions/20260729_highres_ce_multiseed_final_test"
ANALYSIS_ID = "20260817_kang_vs_highres_significance_corrected"
OUTPUT_DIR = PROJECT_ROOT / f"04_实验/metrics/{ANALYSIS_ID}"
PAPER_TABLE = PROJECT_ROOT / f"05_结果/tables/table_kang_vs_highres_paired_significance_corrected.csv"
SEEDS = (42, 123, 2026)
SAMPLES = 25_943
REPLICATES = 10_000
N_CLASSES = 9
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


def load_helpers():
    source = Path(__file__).with_name("analyze_final_vs_resnet18_significance.py")
    spec = importlib.util.spec_from_file_location("paired_helpers_kang_corrected", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load paired-test helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run corrected paired significance tests on frozen Kang and HighRes "
            "test predictions. No images or checkpoints are loaded."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate frozen inputs and print observed metrics without writing files.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run the tests and create the new corrected output namespace.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=REPLICATES,
        help=f"Bootstrap/randomization repetitions (default: {REPLICATES}).",
    )
    return parser.parse_args()


def _integer_column(frame: pd.DataFrame, column: str, source: Path) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.all(np.equal(values, np.floor(values))):
        raise ValueError(f"Non-integer values in {column}: {source}")
    return values.to_numpy(dtype=np.int64)


def load_prediction(path: Path, source_name: str) -> tuple[pd.DataFrame, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "source_index",
        "label",
        "label_id",
        "predicted_label_id",
        "predicted_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source_name} prediction {path}: {missing}")
    if len(frame) != SAMPLES:
        raise ValueError(f"Expected {SAMPLES} rows in {path}, found {len(frame)}")

    frame = frame.copy()
    source_index = _integer_column(frame, "source_index", path)
    true_ids = _integer_column(frame, "label_id", path)
    predicted_ids = _integer_column(frame, "predicted_label_id", path)
    if np.unique(source_index).size != SAMPLES:
        raise ValueError(f"source_index is not unique in {path}")
    if np.any((true_ids < 0) | (true_ids >= N_CLASSES)):
        raise ValueError(f"Invalid true label IDs in {path}")
    if np.any((predicted_ids < 0) | (predicted_ids >= N_CLASSES)):
        raise ValueError(f"Invalid predicted label IDs in {path}")
    labels = frame["label"].astype(str).to_numpy()
    predicted_labels = frame["predicted_label"].astype(str).to_numpy()
    expected_labels = np.asarray(CLASS_NAMES, dtype=object)
    if not np.array_equal(expected_labels[true_ids], labels):
        raise ValueError(f"label and label_id disagree in {path}")
    expected_predicted_labels = expected_labels[predicted_ids]
    label_mismatch = expected_predicted_labels != predicted_labels
    # One frozen HighRes CSV row contains the harmless text typo ``nne`` for
    # class ``none``.  Metrics are computed from integer IDs; retain the raw
    # file hash and record this metadata anomaly instead of rewriting it.
    tolerated_typo = (predicted_ids == 8) & (predicted_labels == "nne")
    if np.any(label_mismatch & ~tolerated_typo):
        raise ValueError(f"predicted_label and predicted_label_id disagree in {path}")
    if "true_label_id" in frame.columns:
        true_column = _integer_column(frame, "true_label_id", path)
        if not np.array_equal(true_column, true_ids):
            raise ValueError(f"true_label_id and label_id disagree in {path}")
    if "split" in frame.columns and not frame["split"].astype(str).eq("test").all():
        raise ValueError(f"Non-test rows found in {path}")
    if "correct" in frame.columns:
        normalized = frame["correct"].astype(str).str.strip().str.lower()
        observed = normalized.map({"true": True, "false": False})
        if observed.isna().any() or not np.array_equal(
            observed.to_numpy(dtype=bool), true_ids == predicted_ids
        ):
            raise ValueError(f"correct flag is inconsistent in {path}")

    keep = pd.DataFrame(
        {
            "source_index": source_index,
            "true_label_id": true_ids,
            "predicted_label_id": predicted_ids,
        }
    )
    return keep.sort_values("source_index").reset_index(drop=True), int(tolerated_typo.sum())


def paired_rows_for_seed(seed: int, helpers, replicates: int, random_seed: int) -> dict:
    kang_path = KANG_DIR / f"predictions_seed{seed}.csv"
    highres_path = HIGHRES_DIR / f"predictions_seed{seed}.csv"
    kang, kang_metadata_mismatches = load_prediction(kang_path, "Kang")
    highres, highres_metadata_mismatches = load_prediction(highres_path, "HighRes")
    if not np.array_equal(kang["source_index"], highres["source_index"]):
        raise ValueError(f"source_index mismatch for seed {seed}")
    if not np.array_equal(kang["true_label_id"], highres["true_label_id"]):
        raise ValueError(f"true labels mismatch for seed {seed}")

    y_true = kang["true_label_id"].to_numpy(dtype=np.int64)
    y_kang = kang["predicted_label_id"].to_numpy(dtype=np.int64)
    y_highres = highres["predicted_label_id"].to_numpy(dtype=np.int64)
    kang_correct = y_kang == y_true
    highres_correct = y_highres == y_true
    mcnemar = helpers.mcnemar_exact(kang_correct, highres_correct)
    accuracy_delta, f1_delta = helpers.paired_stratified_bootstrap(
        y_true,
        y_kang,
        y_highres,
        replicates,
        random_seed,
    )
    _, randomization_p, randomization_extreme_count = helpers.paired_macro_f1_randomization(
        y_true,
        y_kang,
        y_highres,
        replicates,
        random_seed + 100_000,
    )
    kang_conf = helpers.confusion_matrix_from_labels(y_true, y_kang)
    highres_conf = helpers.confusion_matrix_from_labels(y_true, y_highres)
    kang_acc, kang_f1 = helpers.metrics_from_confusion(kang_conf)
    highres_acc, highres_f1 = helpers.metrics_from_confusion(highres_conf)
    return {
        "seed": seed,
        "accuracy_kang": float(kang_acc),
        "accuracy_highres": float(highres_acc),
        "accuracy_delta_pp_highres_minus_kang": float((highres_acc - kang_acc) * 100),
        "accuracy_delta_ci_low_pp": float(np.quantile(accuracy_delta, 0.025)),
        "accuracy_delta_ci_high_pp": float(np.quantile(accuracy_delta, 0.975)),
        "macro_f1_kang": float(kang_f1),
        "macro_f1_highres": float(highres_f1),
        "macro_f1_delta_pp_highres_minus_kang": float((highres_f1 - kang_f1) * 100),
        "macro_f1_delta_ci_low_pp": float(np.quantile(f1_delta, 0.025)),
        "macro_f1_delta_ci_high_pp": float(np.quantile(f1_delta, 0.975)),
        "mcnemar_kang_only_correct": mcnemar["baseline_only_correct"],
        "mcnemar_highres_only_correct": mcnemar["final_only_correct"],
        "mcnemar_discordant_pairs": mcnemar["discordant_pairs"],
        "mcnemar_p_raw": mcnemar["exact_two_sided_p_value"],
        "macro_f1_randomization_extreme_count": randomization_extreme_count,
        "macro_f1_randomization_p_raw": randomization_p,
        "samples": len(y_true),
        "kang_metadata_label_mismatches_tolerated": kang_metadata_mismatches,
        "highres_metadata_label_mismatches_tolerated": highres_metadata_mismatches,
    }, {str(kang_path.relative_to(PROJECT_ROOT)): sha256_file(kang_path), str(highres_path.relative_to(PROJECT_ROOT)): sha256_file(highres_path)}


def validate_inputs(helpers) -> tuple[pd.DataFrame, dict]:
    rows = []
    hashes = {}
    for index, seed in enumerate(SEEDS):
        row, input_hashes = paired_rows_for_seed(
            seed,
            helpers,
            replicates=20,
            random_seed=2026081700 + index,
        )
        rows.append(row)
        hashes.update(input_hashes)
    return pd.DataFrame(rows), hashes


def run(replicates: int):
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if OUTPUT_DIR.exists() or PAPER_TABLE.exists():
        raise FileExistsError(
            "Refusing to overwrite corrected significance outputs: "
            f"{OUTPUT_DIR} or {PAPER_TABLE}"
        )
    helpers = load_helpers()
    rows = []
    input_hashes = {}
    for index, seed in enumerate(SEEDS):
        row, hashes = paired_rows_for_seed(
            seed,
            helpers,
            replicates=replicates,
            random_seed=2026081700 + index,
        )
        rows.append(row)
        input_hashes.update(hashes)
    result = pd.DataFrame(rows)
    result["mcnemar_p_holm"] = helpers.holm_adjust(result["mcnemar_p_raw"].to_numpy())
    result["macro_f1_randomization_p_holm"] = helpers.holm_adjust(
        result["macro_f1_randomization_p_raw"].to_numpy()
    )
    result["accuracy_significant_holm_0_05"] = result["mcnemar_p_holm"] < 0.05
    result["macro_f1_significant_holm_0_05"] = (
        result["macro_f1_randomization_p_holm"] < 0.05
    )
    aggregate = pd.DataFrame(
        [
            {
                "accuracy_delta_pp_mean": result["accuracy_delta_pp_highres_minus_kang"].mean(),
                "accuracy_delta_pp_sample_std": result["accuracy_delta_pp_highres_minus_kang"].std(ddof=1),
                "macro_f1_delta_pp_mean": result["macro_f1_delta_pp_highres_minus_kang"].mean(),
                "macro_f1_delta_pp_sample_std": result["macro_f1_delta_pp_highres_minus_kang"].std(ddof=1),
                "accuracy_all_seeds_significant": bool(result["accuracy_significant_holm_0_05"].all()),
                "macro_f1_all_seeds_significant": bool(result["macro_f1_significant_holm_0_05"].all()),
                "samples_per_seed": SAMPLES,
                "seeds": ",".join(str(seed) for seed in SEEDS),
            }
        ]
    )
    OUTPUT_DIR.mkdir(parents=True)
    result.to_csv(OUTPUT_DIR / "paired_per_seed.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "paired_aggregate.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(PAPER_TABLE, index=False)
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "samples_per_seed": SAMPLES,
        "seeds": list(SEEDS),
        "bootstrap_replicates": replicates,
        "randomization_replicates": replicates,
        "holm_scope": "three seed comparisons separately for McNemar and Macro-F1 randomization",
        "corrected_fields": {
            "mcnemar_discordant_pairs": "number of paired correctness disagreements",
            "macro_f1_randomization_extreme_count": "number of randomized deltas at least as extreme as observed",
        },
        "input_hashes": input_hashes,
        "test_split_role": "frozen test predictions; post-test analysis only",
        "images_or_checkpoints_loaded": False,
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


def main() -> None:
    args = parse_args()
    helpers = load_helpers()
    if args.check_inputs:
        observed, _ = validate_inputs(helpers)
        print("INPUT_VALIDATION_OK")
        print(observed.to_string(index=False))
        return
    run(args.replicates)


if __name__ == "__main__":
    main()
