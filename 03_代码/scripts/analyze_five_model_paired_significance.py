#!/usr/bin/env python3
"""Paired significance tests for HighRes ShuffleNetV2 versus four baselines."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "20260801_five_model_multiseed_unified_test"
ANALYSIS_ID = "20260801_five_model_paired_significance"
INPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / EVALUATION_ID
PREDICTION_DIR = INPUT_DIR / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / ANALYSIS_ID
PAPER_TABLE = (
    PROJECT_ROOT
    / "05_结果"
    / "tables"
    / "table_five_model_paired_significance.csv"
)
SEEDS = (42, 123, 2026)
BASELINES = (
    ("resnet18", "ResNet18"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("mobilenet_v3_small", "MobileNetV3-Small"),
    ("standard_shufflenet_v2", "Standard ShuffleNetV2"),
)
REPLICATES = 10_000


def load_helpers():
    path = Path(__file__).with_name("analyze_final_vs_resnet18_significance.py")
    spec = importlib.util.spec_from_file_location("paired_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction(model_id: str, seed: int) -> pd.DataFrame:
    path = PREDICTION_DIR / f"{model_id}_seed{seed}.csv"
    frame = pd.read_csv(path)
    required = {"source_index", "true_label_id", "predicted_label_id"}
    if not required.issubset(frame.columns) or len(frame) != 25943:
        raise ValueError(f"Invalid prediction file: {path}")
    frame = frame.sort_values("source_index").reset_index(drop=True)
    if frame["source_index"].duplicated().any():
        raise ValueError(f"Duplicate samples: {path}")
    return frame[list(required)]


def main():
    if OUTPUT_DIR.exists() or PAPER_TABLE.exists():
        raise FileExistsError("Refusing to overwrite paired significance outputs.")
    helpers = load_helpers()
    rows = []
    input_hashes = {}

    for baseline_index, (baseline_id, baseline_name) in enumerate(BASELINES):
        for seed_index, seed in enumerate(SEEDS):
            final_path = PREDICTION_DIR / f"highres_shufflenet_v2_seed{seed}.csv"
            baseline_path = PREDICTION_DIR / f"{baseline_id}_seed{seed}.csv"
            final = load_prediction("highres_shufflenet_v2", seed)
            baseline = load_prediction(baseline_id, seed)
            if not np.array_equal(final["source_index"], baseline["source_index"]):
                raise ValueError("Paired sample order mismatch.")
            if not np.array_equal(final["true_label_id"], baseline["true_label_id"]):
                raise ValueError("Paired true labels mismatch.")

            y_true = final["true_label_id"].to_numpy(dtype=np.int64)
            y_final = final["predicted_label_id"].to_numpy(dtype=np.int64)
            y_baseline = baseline["predicted_label_id"].to_numpy(dtype=np.int64)
            final_correct = y_final == y_true
            baseline_correct = y_baseline == y_true
            mcnemar = helpers.mcnemar_exact(baseline_correct, final_correct)
            accuracy_delta, f1_delta = helpers.paired_stratified_bootstrap(
                y_true, y_baseline, y_final, REPLICATES,
                2026080100 + baseline_index * 10 + seed_index,
            )
            _, randomization_p, discordant_predictions = (
                helpers.paired_macro_f1_randomization(
                    y_true, y_baseline, y_final, REPLICATES,
                    2026081100 + baseline_index * 10 + seed_index,
                )
            )
            baseline_confusion = helpers.confusion_matrix_from_labels(y_true, y_baseline)
            final_confusion = helpers.confusion_matrix_from_labels(y_true, y_final)
            baseline_acc, baseline_f1 = helpers.metrics_from_confusion(baseline_confusion)
            final_acc, final_f1 = helpers.metrics_from_confusion(final_confusion)
            rows.append({
                "baseline_id": baseline_id,
                "baseline": baseline_name,
                "seed": seed,
                "accuracy_baseline": float(baseline_acc),
                "accuracy_highres": float(final_acc),
                "accuracy_delta_pp": float((final_acc - baseline_acc) * 100),
                "accuracy_delta_ci_low_pp": float(np.quantile(accuracy_delta, 0.025)),
                "accuracy_delta_ci_high_pp": float(np.quantile(accuracy_delta, 0.975)),
                "macro_f1_baseline": float(baseline_f1),
                "macro_f1_highres": float(final_f1),
                "macro_f1_delta_pp": float((final_f1 - baseline_f1) * 100),
                "macro_f1_delta_ci_low_pp": float(np.quantile(f1_delta, 0.025)),
                "macro_f1_delta_ci_high_pp": float(np.quantile(f1_delta, 0.975)),
                "mcnemar_baseline_only_correct": mcnemar["baseline_only_correct"],
                "mcnemar_highres_only_correct": mcnemar["final_only_correct"],
                "mcnemar_p_raw": mcnemar["exact_two_sided_p_value"],
                "macro_f1_randomization_discordant_predictions": discordant_predictions,
                "macro_f1_randomization_p_raw": randomization_p,
            })
            input_hashes[str(final_path.relative_to(PROJECT_ROOT))] = sha256_file(final_path)
            input_hashes[str(baseline_path.relative_to(PROJECT_ROOT))] = sha256_file(baseline_path)

    results = pd.DataFrame(rows)
    results["mcnemar_p_holm"] = helpers.holm_adjust(
        results["mcnemar_p_raw"].to_numpy()
    )
    results["macro_f1_randomization_p_holm"] = helpers.holm_adjust(
        results["macro_f1_randomization_p_raw"].to_numpy()
    )
    results["accuracy_significant_holm_0_05"] = results["mcnemar_p_holm"] < 0.05
    results["macro_f1_significant_holm_0_05"] = (
        results["macro_f1_randomization_p_holm"] < 0.05
    )

    aggregate = results.groupby(["baseline_id", "baseline"], sort=False).agg(
        accuracy_delta_pp_mean=("accuracy_delta_pp", "mean"),
        accuracy_delta_pp_sample_std=("accuracy_delta_pp", "std"),
        macro_f1_delta_pp_mean=("macro_f1_delta_pp", "mean"),
        macro_f1_delta_pp_sample_std=("macro_f1_delta_pp", "std"),
        accuracy_all_seeds_significant=("accuracy_significant_holm_0_05", "all"),
        macro_f1_all_seeds_significant=("macro_f1_significant_holm_0_05", "all"),
    ).reset_index()

    OUTPUT_DIR.mkdir(parents=True)
    results.to_csv(OUTPUT_DIR / "paired_per_seed.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "paired_aggregate.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(PAPER_TABLE, index=False)
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "comparison_count": len(results),
        "bootstrap_replicates": REPLICATES,
        "randomization_replicates": REPLICATES,
        "holm_scope": "12 comparisons separately for McNemar and Macro-F1 randomization",
        "input_hashes": input_hashes,
        "test_split_role": "fixed benchmark with historical access",
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
