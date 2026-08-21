#!/usr/bin/env python3
"""Paired post-test significance analysis for Kang stacking vs HighRes."""

from __future__ import annotations

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
OUTPUT_DIR = PROJECT_ROOT / "04_实验/metrics/20260810_kang_vs_highres_significance"
PAPER_TABLE = PROJECT_ROOT / "05_结果/tables/table_kang_vs_highres_paired_significance.csv"
SEEDS = (42, 123, 2026)
SAMPLES = 25_943
REPLICATES = 10_000


def load_helpers():
    source = Path(__file__).with_name("analyze_final_vs_resnet18_significance.py")
    spec = importlib.util.spec_from_file_location("paired_helpers_kang", source)
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


def load_prediction(path: Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    true_column = "true_label_id" if "true_label_id" in frame else "label_id"
    required = {"source_index", true_column, "predicted_label_id"}
    if len(frame) != SAMPLES or not required.issubset(frame.columns):
        raise ValueError(f"Invalid prediction file: {path}")
    frame = frame[["source_index", true_column, "predicted_label_id"]].rename(
        columns={true_column: "true_label_id"}
    ).sort_values("source_index").reset_index(drop=True)
    if frame["source_index"].duplicated().any():
        raise ValueError(f"Duplicate source_index in {source_name}: {path}")
    return frame


def main():
    if OUTPUT_DIR.exists() or PAPER_TABLE.exists():
        raise FileExistsError("Refusing to overwrite Kang significance outputs.")
    helpers = load_helpers()
    rows = []
    input_hashes = {}
    for index, seed in enumerate(SEEDS):
        kang_path = KANG_DIR / f"predictions_seed{seed}.csv"
        highres_path = HIGHRES_DIR / f"predictions_seed{seed}.csv"
        kang = load_prediction(kang_path, "Kang")
        highres = load_prediction(highres_path, "HighRes")
        if not np.array_equal(kang["source_index"], highres["source_index"]):
            raise ValueError(f"Sample order mismatch for seed {seed}")
        if not np.array_equal(kang["true_label_id"], highres["true_label_id"]):
            raise ValueError(f"True labels mismatch for seed {seed}")
        y_true = kang["true_label_id"].to_numpy(dtype=np.int64)
        y_kang = kang["predicted_label_id"].to_numpy(dtype=np.int64)
        y_highres = highres["predicted_label_id"].to_numpy(dtype=np.int64)
        kang_correct = y_kang == y_true
        highres_correct = y_highres == y_true
        mcnemar = helpers.mcnemar_exact(kang_correct, highres_correct)
        accuracy_delta, f1_delta = helpers.paired_stratified_bootstrap(
            y_true, y_kang, y_highres, REPLICATES, 2026081000 + index
        )
        _, randomization_p, discordant = helpers.paired_macro_f1_randomization(
            y_true, y_kang, y_highres, REPLICATES, 2026082000 + index
        )
        kang_conf = helpers.confusion_matrix_from_labels(y_true, y_kang)
        highres_conf = helpers.confusion_matrix_from_labels(y_true, y_highres)
        kang_acc, kang_f1 = helpers.metrics_from_confusion(kang_conf)
        highres_acc, highres_f1 = helpers.metrics_from_confusion(highres_conf)
        rows.append({
            "seed": seed,
            "accuracy_kang": kang_acc,
            "accuracy_highres": highres_acc,
            "accuracy_delta_pp_highres_minus_kang": (highres_acc - kang_acc) * 100,
            "accuracy_delta_ci_low_pp": float(np.quantile(accuracy_delta, 0.025)),
            "accuracy_delta_ci_high_pp": float(np.quantile(accuracy_delta, 0.975)),
            "macro_f1_kang": kang_f1,
            "macro_f1_highres": highres_f1,
            "macro_f1_delta_pp_highres_minus_kang": (highres_f1 - kang_f1) * 100,
            "macro_f1_delta_ci_low_pp": float(np.quantile(f1_delta, 0.025)),
            "macro_f1_delta_ci_high_pp": float(np.quantile(f1_delta, 0.975)),
            "mcnemar_kang_only_correct": mcnemar["baseline_only_correct"],
            "mcnemar_highres_only_correct": mcnemar["final_only_correct"],
            "mcnemar_p_raw": mcnemar["exact_two_sided_p_value"],
            "macro_f1_randomization_discordant_predictions": discordant,
            "macro_f1_randomization_p_raw": randomization_p,
        })
        input_hashes[str(kang_path.relative_to(PROJECT_ROOT))] = sha256_file(kang_path)
        input_hashes[str(highres_path.relative_to(PROJECT_ROOT))] = sha256_file(highres_path)
    result = pd.DataFrame(rows)
    result["mcnemar_p_holm"] = helpers.holm_adjust(result["mcnemar_p_raw"].to_numpy())
    result["macro_f1_randomization_p_holm"] = helpers.holm_adjust(
        result["macro_f1_randomization_p_raw"].to_numpy()
    )
    result["accuracy_significant_holm_0_05"] = result["mcnemar_p_holm"] < 0.05
    result["macro_f1_significant_holm_0_05"] = result["macro_f1_randomization_p_holm"] < 0.05
    aggregate = pd.DataFrame([{
        "accuracy_delta_pp_mean": result["accuracy_delta_pp_highres_minus_kang"].mean(),
        "accuracy_delta_pp_sample_std": result["accuracy_delta_pp_highres_minus_kang"].std(ddof=1),
        "macro_f1_delta_pp_mean": result["macro_f1_delta_pp_highres_minus_kang"].mean(),
        "macro_f1_delta_pp_sample_std": result["macro_f1_delta_pp_highres_minus_kang"].std(ddof=1),
        "accuracy_all_seeds_significant": result["accuracy_significant_holm_0_05"].all(),
        "macro_f1_all_seeds_significant": result["macro_f1_significant_holm_0_05"].all(),
    }])
    OUTPUT_DIR.mkdir(parents=True)
    result.to_csv(OUTPUT_DIR / "paired_per_seed.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "paired_aggregate.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(PAPER_TABLE, index=False)
    manifest = {
        "analysis_id": "20260810_kang_vs_highres_significance",
        "completed_at": datetime.now().astimezone().isoformat(),
        "samples_per_seed": SAMPLES,
        "seeds": list(SEEDS),
        "bootstrap_replicates": REPLICATES,
        "randomization_replicates": REPLICATES,
        "holm_scope": "three seed comparisons separately for McNemar and Macro-F1 randomization",
        "input_hashes": input_hashes,
        "test_split_role": "frozen test predictions; post-test analysis only",
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
