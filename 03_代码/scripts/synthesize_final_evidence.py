#!/usr/bin/env python3
"""Synthesize frozen experiment evidence into paper-ready tables and notes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ID = "20260729_final_evidence_synthesis"

FREEZE_MANIFEST = (
    PROJECT_ROOT / "00_项目管理" / "20260729_最终模型冻结清单.json"
)
FINAL_TEST_DIR = (
    PROJECT_ROOT
    / "04_实验"
    / "metrics"
    / "20260729_highres_ce_multiseed_final_test"
)
FINAL_TEST_MANIFEST = FINAL_TEST_DIR / "artifact_manifest.json"
FINAL_TEST_SUMMARY = FINAL_TEST_DIR / "test_aggregate_summary.json"
BASELINE_SUMMARY = (
    PROJECT_ROOT / "04_实验" / "metrics" / "resnet18_baseline_full_test_summary.json"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "04_实验"
    / "metrics"
    / "20260729_highres_ce_eca_multiseed_validation"
)
VALIDATION_SELECTION = VALIDATION_DIR / "selection_summary.json"
VALIDATION_AGGREGATE = VALIDATION_DIR / "aggregate_validation.csv"
VALIDATION_PAIRED = VALIDATION_DIR / "paired_eca_minus_no_eca.csv"

ERROR_DIR = (
    PROJECT_ROOT
    / "04_实验"
    / "metrics"
    / "20260729_highres_ce_multiseed_error_analysis"
)
ERROR_MANIFEST = ERROR_DIR / "analysis_manifest.json"
ERROR_SUMMARY = ERROR_DIR / "analysis_summary.json"
ERROR_PER_CLASS = ERROR_DIR / "per_class_error_summary.csv"
ERROR_UNANIMOUS_PAIRS = ERROR_DIR / "unanimous_error_pairs.csv"

TRAINING_DIR = (
    PROJECT_ROOT
    / "04_实验"
    / "metrics"
    / "20260729_highres_ce_multiseed_training_curves"
)
TRAINING_MANIFEST = TRAINING_DIR / "analysis_manifest.json"
TRAINING_SUMMARY = TRAINING_DIR / "training_summary.json"

EFFICIENCY_DIR = (
    PROJECT_ROOT
    / "04_实验"
    / "benchmarks"
    / "20260729_final_model_complexity_mps_benchmark"
)
EFFICIENCY_MANIFEST = EFFICIENCY_DIR / "analysis_manifest.json"
MODEL_COMPLEXITY = EFFICIENCY_DIR / "model_complexity.csv"
MPS_BENCHMARK = EFFICIENCY_DIR / "mps_benchmark_summary.csv"
EFFICIENCY_SUMMARY = EFFICIENCY_DIR / "efficiency_summary.json"

SIGNIFICANCE_DIR = (
    PROJECT_ROOT
    / "04_实验"
    / "metrics"
    / "20260729_final_vs_resnet18_statistical_significance"
)
SIGNIFICANCE_MANIFEST = SIGNIFICANCE_DIR / "analysis_manifest.json"
PAIRED_OBSERVED = SIGNIFICANCE_DIR / "paired_observed_metrics.csv"
PAIRED_MCNEMAR = SIGNIFICANCE_DIR / "paired_accuracy_mcnemar.csv"
PAIRED_BOOTSTRAP = SIGNIFICANCE_DIR / "paired_bootstrap_summary.csv"
PAIRED_RANDOMIZATION = SIGNIFICANCE_DIR / "paired_macro_f1_randomization.csv"
SIGNIFICANCE_SUMMARY = SIGNIFICANCE_DIR / "analysis_summary.json"

EXPECTED_MANIFEST_HASHES = {
    FREEZE_MANIFEST: (
        "c682922b1c291f03a41a81e5d15ebfe08e4456e77c816f7943041fc11fcef8db"
    ),
    FINAL_TEST_MANIFEST: (
        "70d62a8d3f122871bedb609000dbde71d6238b5f69dcf75dfc587bb83643d33c"
    ),
    ERROR_MANIFEST: (
        "9532bc25419a03be950940f6ca1c0a2310f96ff598b539b3188b0f04b7a89b32"
    ),
    TRAINING_MANIFEST: (
        "b8a7a737995af2626cb9c0d067f7b38724aa67fa0b47e056dbfc45f34be09e4a"
    ),
    EFFICIENCY_MANIFEST: (
        "e84e91d9f098b06a07a0807eadaae56fec961af2f2547a5f4683c61012c88dc0"
    ),
    SIGNIFICANCE_MANIFEST: (
        "e29a9410ff833042d1b96bec94f175de2758bfea75e54b4607073348ccc303d2"
    ),
}

EXPECTED_VALIDATION_HASHES = {
    VALIDATION_SELECTION: (
        "5334249345238007a731245b5e1a5e7dae5c7b543d091ad70421d2d18b7555ef"
    ),
    VALIDATION_AGGREGATE: (
        "6bea58ce9905c951decc34634022d73b45b1d0c71c0dba6a0e42817afd6a5366"
    ),
    VALIDATION_PAIRED: (
        "0e67d0ac1c41bc5342ae494301cc19f9d3c462236514253a5ba948f60ce5457c"
    ),
}

OVERALL_TABLE_NAME = "table_final_overall_results.csv"
CLAIMS_TABLE_NAME = "table_final_claims_evidence.csv"
WRITING_NOTES_NAME = "00_论文写作数据底稿.md"
METRIC_FILENAMES = (
    "final_evidence_summary.json",
    "analysis_manifest.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine frozen validation, test, efficiency, significance, "
            "training, and error-analysis outputs without training or inference."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate all source artifacts and print the main final metrics.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Create the final overall tables and writing evidence notes.",
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
        "table_dir": output_root / "05_结果" / "tables",
        "manuscript_dir": output_root / "06_论文" / "manuscript",
    }


def expected_output_files(output_root: Path) -> list[Path]:
    paths = output_paths(output_root)
    return [
        *(paths["metrics_dir"] / name for name in METRIC_FILENAMES),
        paths["table_dir"] / OVERALL_TABLE_NAME,
        paths["table_dir"] / CLAIMS_TABLE_NAME,
        paths["manuscript_dir"] / WRITING_NOTES_NAME,
    ]


def ensure_outputs_are_available(output_root: Path):
    paths = output_paths(output_root)
    existing = []
    if paths["metrics_dir"].exists():
        existing.append(paths["metrics_dir"])
    existing.extend(path for path in expected_output_files(output_root) if path.exists())
    if existing:
        formatted = "\n".join(f"- {path}" for path in sorted(set(existing)))
        raise FileExistsError(
            "Refusing to overwrite existing final-evidence outputs:\n"
            f"{formatted}"
        )


def validate_file_hash(path: Path, expected_hash: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence source: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Evidence source hash mismatch: {path}")
    return actual_hash


def validate_manifest_output(
    manifest: dict,
    path: Path,
    manifest_key: str,
) -> str:
    expected_hash = manifest.get("output_sha256", {}).get(manifest_key)
    if not expected_hash:
        raise ValueError(f"Output hash missing from source manifest: {manifest_key}")
    return validate_file_hash(path, expected_hash)


def load_and_validate_sources() -> dict:
    input_hashes = {}
    manifests = {}
    for path, expected_hash in EXPECTED_MANIFEST_HASHES.items():
        validate_file_hash(path, expected_hash)
        input_hashes[str(path.relative_to(PROJECT_ROOT))] = expected_hash
        manifests[path] = load_json(path)

    for path, expected_hash in EXPECTED_VALIDATION_HASHES.items():
        validate_file_hash(path, expected_hash)
        input_hashes[str(path.relative_to(PROJECT_ROOT))] = expected_hash

    freeze_manifest = manifests[FREEZE_MANIFEST]
    final_manifest = manifests[FINAL_TEST_MANIFEST]
    error_manifest = manifests[ERROR_MANIFEST]
    training_manifest = manifests[TRAINING_MANIFEST]
    efficiency_manifest = manifests[EFFICIENCY_MANIFEST]
    significance_manifest = manifests[SIGNIFICANCE_MANIFEST]

    if freeze_manifest.get("selected_architecture_id") != "highres_ce_no_eca":
        raise ValueError("Frozen final architecture changed.")
    if final_manifest.get("evaluation_id") != (
        "20260729_highres_ce_multiseed_final_test"
    ):
        raise ValueError("Final test evaluation ID changed.")

    final_output_hashes = {
        item["path"]: item["sha256"] for item in final_manifest["files"]
    }
    final_summary_hash = final_output_hashes.get("test_aggregate_summary.json")
    if not final_summary_hash:
        raise ValueError("Final test summary hash is missing from its manifest.")
    validate_file_hash(FINAL_TEST_SUMMARY, final_summary_hash)
    input_hashes[str(FINAL_TEST_SUMMARY.relative_to(PROJECT_ROOT))] = (
        final_summary_hash
    )

    historical = {
        item.get("path", item.get("result")): item["sha256"]
        for item in freeze_manifest.get("historical_test_access", [])
        if item.get("path", item.get("result"))
    }
    baseline_relative = str(BASELINE_SUMMARY.relative_to(PROJECT_ROOT))
    baseline_hash = historical.get(baseline_relative)
    if not baseline_hash:
        raise ValueError("Baseline summary hash is missing from the freeze manifest.")
    validate_file_hash(BASELINE_SUMMARY, baseline_hash)
    input_hashes[baseline_relative] = baseline_hash

    manifest_sources = (
        (
            error_manifest,
            ERROR_SUMMARY,
            str(ERROR_SUMMARY.relative_to(PROJECT_ROOT)),
        ),
        (
            error_manifest,
            ERROR_PER_CLASS,
            str(ERROR_PER_CLASS.relative_to(PROJECT_ROOT)),
        ),
        (
            error_manifest,
            ERROR_UNANIMOUS_PAIRS,
            str(ERROR_UNANIMOUS_PAIRS.relative_to(PROJECT_ROOT)),
        ),
        (
            training_manifest,
            TRAINING_SUMMARY,
            str(TRAINING_SUMMARY.relative_to(PROJECT_ROOT)),
        ),
        (
            efficiency_manifest,
            MODEL_COMPLEXITY,
            str(MODEL_COMPLEXITY.relative_to(PROJECT_ROOT)),
        ),
        (
            efficiency_manifest,
            MPS_BENCHMARK,
            str(MPS_BENCHMARK.relative_to(PROJECT_ROOT)),
        ),
        (
            efficiency_manifest,
            EFFICIENCY_SUMMARY,
            str(EFFICIENCY_SUMMARY.relative_to(PROJECT_ROOT)),
        ),
        (
            significance_manifest,
            PAIRED_OBSERVED,
            str(PAIRED_OBSERVED.relative_to(PROJECT_ROOT)),
        ),
        (
            significance_manifest,
            PAIRED_MCNEMAR,
            str(PAIRED_MCNEMAR.relative_to(PROJECT_ROOT)),
        ),
        (
            significance_manifest,
            PAIRED_BOOTSTRAP,
            str(PAIRED_BOOTSTRAP.relative_to(PROJECT_ROOT)),
        ),
        (
            significance_manifest,
            PAIRED_RANDOMIZATION,
            str(PAIRED_RANDOMIZATION.relative_to(PROJECT_ROOT)),
        ),
        (
            significance_manifest,
            SIGNIFICANCE_SUMMARY,
            str(SIGNIFICANCE_SUMMARY.relative_to(PROJECT_ROOT)),
        ),
    )
    for manifest, path, key in manifest_sources:
        source_hash = validate_manifest_output(manifest, path, key)
        input_hashes[key] = source_hash

    sources = {
        "freeze": freeze_manifest,
        "final_test": load_json(FINAL_TEST_SUMMARY),
        "baseline": load_json(BASELINE_SUMMARY),
        "validation_selection": load_json(VALIDATION_SELECTION),
        "validation_aggregate": pd.read_csv(VALIDATION_AGGREGATE),
        "validation_paired": pd.read_csv(VALIDATION_PAIRED),
        "error_summary": load_json(ERROR_SUMMARY),
        "error_per_class": pd.read_csv(ERROR_PER_CLASS),
        "error_pairs": pd.read_csv(ERROR_UNANIMOUS_PAIRS),
        "training": load_json(TRAINING_SUMMARY),
        "complexity": pd.read_csv(MODEL_COMPLEXITY),
        "benchmark": pd.read_csv(MPS_BENCHMARK),
        "efficiency": load_json(EFFICIENCY_SUMMARY),
        "paired_observed": pd.read_csv(PAIRED_OBSERVED),
        "mcnemar": pd.read_csv(PAIRED_MCNEMAR),
        "bootstrap": pd.read_csv(PAIRED_BOOTSTRAP),
        "randomization": pd.read_csv(PAIRED_RANDOMIZATION),
        "significance": load_json(SIGNIFICANCE_SUMMARY),
        "input_hashes": input_hashes,
    }
    validate_source_invariants(sources)
    return sources


def validate_source_invariants(sources: dict):
    freeze = sources["freeze"]
    final_test = sources["final_test"]
    baseline = sources["baseline"]
    validation = sources["validation_aggregate"]
    complexity = sources["complexity"]
    benchmark = sources["benchmark"]
    paired_observed = sources["paired_observed"]

    expected_seeds = {42, 123, 2026}
    if int(final_test.get("n_seeds", -1)) != 3:
        raise ValueError("Final test must contain three seeds.")
    if set(final_test.get("seeds", [])) != expected_seeds:
        raise ValueError("Final test seed set changed.")
    if int(baseline.get("test_samples", -1)) != 25943:
        raise ValueError("Baseline test sample count changed.")
    if int(freeze["dataset"]["test_samples"]) != 25943:
        raise ValueError("Frozen test sample count changed.")
    if set(validation["architecture_id"]) != {
        "highres_ce_no_eca",
        "highres_eca_ce",
    }:
        raise ValueError("Validation architecture set changed.")
    if set(complexity["model_id"]) != {
        "resnet18_baseline",
        "highres_shufflenetv2_ce",
    }:
        raise ValueError("Complexity model set changed.")
    if set(benchmark["batch_size"].astype(int)) != {1, 128}:
        raise ValueError("MPS benchmark batch sizes changed.")
    if set(paired_observed["seed"].astype(int)) != expected_seeds:
        raise ValueError("Paired statistical comparison seed set changed.")

    final_accuracy = final_test["metrics"]["accuracy"]["mean"]
    final_macro_f1 = final_test["metrics"]["macro_f1"]["mean"]
    if not np.isclose(
        paired_observed["final_accuracy"].mean(), final_accuracy, atol=1e-12
    ):
        raise ValueError("Final accuracy disagrees across frozen outputs.")
    if not np.isclose(
        paired_observed["final_macro_f1"].mean(), final_macro_f1, atol=1e-12
    ):
        raise ValueError("Final Macro-F1 disagrees across frozen outputs.")
    if not np.isclose(
        paired_observed["baseline_accuracy"].iloc[0],
        baseline["accuracy"],
        atol=1e-12,
    ):
        raise ValueError("Baseline accuracy disagrees across frozen outputs.")


def percentage_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        raise ValueError("Percentage change baseline cannot be zero.")
    return (new_value / old_value - 1.0) * 100.0


def build_overall_table(sources: dict) -> pd.DataFrame:
    final_metrics = sources["final_test"]["metrics"]
    baseline = sources["baseline"]
    complexity = sources["complexity"].set_index("model_id")
    benchmark = sources["benchmark"]

    model_specs = (
        (
            "resnet18_baseline",
            "ResNet18 Baseline",
            1,
            baseline["accuracy"],
            np.nan,
            baseline["macro_f1"],
            np.nan,
            baseline["balanced_accuracy"],
            np.nan,
        ),
        (
            "highres_shufflenetv2_ce",
            "Final HighRes ShuffleNetV2 + CE (no ECA)",
            3,
            final_metrics["accuracy"]["mean"],
            final_metrics["accuracy"]["sample_std"],
            final_metrics["macro_f1"]["mean"],
            final_metrics["macro_f1"]["sample_std"],
            final_metrics["balanced_accuracy"]["mean"],
            final_metrics["balanced_accuracy"]["sample_std"],
        ),
    )

    rows = []
    for (
        model_id,
        model_name,
        training_runs,
        accuracy,
        accuracy_std,
        macro_f1,
        macro_f1_std,
        balanced_accuracy,
        balanced_accuracy_std,
    ) in model_specs:
        complexity_row = complexity.loc[model_id]
        batch1 = benchmark.loc[
            benchmark["model_id"].eq(model_id)
            & benchmark["batch_size"].eq(1)
        ].iloc[0]
        batch128 = benchmark.loc[
            benchmark["model_id"].eq(model_id)
            & benchmark["batch_size"].eq(128)
        ].iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "model": model_name,
                "training_runs": training_runs,
                "test_samples_per_run": 25943,
                "test_accuracy_percent_mean": accuracy * 100.0,
                "test_accuracy_percent_sample_std": (
                    accuracy_std * 100.0 if np.isfinite(accuracy_std) else np.nan
                ),
                "test_macro_f1_percent_mean": macro_f1 * 100.0,
                "test_macro_f1_percent_sample_std": (
                    macro_f1_std * 100.0 if np.isfinite(macro_f1_std) else np.nan
                ),
                "test_balanced_accuracy_percent_mean": balanced_accuracy * 100.0,
                "test_balanced_accuracy_percent_sample_std": (
                    balanced_accuracy_std * 100.0
                    if np.isfinite(balanced_accuracy_std)
                    else np.nan
                ),
                "parameter_count": int(complexity_row["parameter_count"]),
                "parameters_million": complexity_row["parameters_million"],
                "state_dict_memory_mib": complexity_row["state_dict_memory_mib"],
                "conv_linear_flops_million": complexity_row[
                    "conv_linear_flops_million"
                ],
                "estimated_macs_million": complexity_row[
                    "estimated_macs_million"
                ],
                "mps_batch1_latency_ms_mean": batch1["batch_latency_ms_mean"],
                "mps_batch1_latency_ms_sample_std": batch1[
                    "batch_latency_ms_sample_std"
                ],
                "mps_batch128_samples_per_second_mean": batch128[
                    "samples_per_second_mean"
                ],
                "mps_batch128_samples_per_second_sample_std": batch128[
                    "samples_per_second_sample_std"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_evidence_summary(sources: dict, overall: pd.DataFrame) -> dict:
    baseline_row = overall.loc[
        overall["model_id"].eq("resnet18_baseline")
    ].iloc[0]
    final_row = overall.loc[
        overall["model_id"].eq("highres_shufflenetv2_ce")
    ].iloc[0]
    validation = sources["validation_aggregate"].set_index("architecture_id")
    no_eca = validation.loc["highres_ce_no_eca"]
    with_eca = validation.loc["highres_eca_ce"]
    mcnemar = sources["mcnemar"]
    bootstrap = sources["bootstrap"]
    randomization = sources["randomization"]
    error_summary = sources["error_summary"]
    error_per_class = sources["error_per_class"].sort_values(
        "common_error_rate_percent", ascending=False
    )
    top_error_pairs = sources["error_pairs"].head(5)
    training = sources["training"]

    accuracy_gain = (
        final_row["test_accuracy_percent_mean"]
        - baseline_row["test_accuracy_percent_mean"]
    )
    macro_f1_gain = (
        final_row["test_macro_f1_percent_mean"]
        - baseline_row["test_macro_f1_percent_mean"]
    )
    balanced_accuracy_gain = (
        final_row["test_balanced_accuracy_percent_mean"]
        - baseline_row["test_balanced_accuracy_percent_mean"]
    )
    parameter_reduction = -percentage_change(
        final_row["parameter_count"], baseline_row["parameter_count"]
    )
    state_dict_reduction = -percentage_change(
        final_row["state_dict_memory_mib"], baseline_row["state_dict_memory_mib"]
    )
    flops_change = percentage_change(
        final_row["conv_linear_flops_million"],
        baseline_row["conv_linear_flops_million"],
    )
    latency_change = percentage_change(
        final_row["mps_batch1_latency_ms_mean"],
        baseline_row["mps_batch1_latency_ms_mean"],
    )
    throughput_change = percentage_change(
        final_row["mps_batch128_samples_per_second_mean"],
        baseline_row["mps_batch128_samples_per_second_mean"],
    )

    class_difficulties = [
        {
            "class_name": row["class_name"],
            "support": int(row["support_per_seed"]),
            "mean_recall_percent": float(row["recall_percent_mean"]),
            "common_error_rate_percent": float(row["common_error_rate_percent"]),
        }
        for _, row in error_per_class.head(5).iterrows()
    ]
    error_pairs = [
        {
            "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "sample_count": int(row["sample_count"]),
        }
        for _, row in top_error_pairs.iterrows()
    ]

    return {
        "analysis_id": ANALYSIS_ID,
        "task": "WM-811K wafer bin-map nine-class failure-pattern classification",
        "task_is_object_detection": False,
        "final_model": "HighRes ShuffleNetV2 + CrossEntropyLoss without ECA",
        "selection": {
            "split": "validation",
            "seeds": [42, 123, 2026],
            "no_eca_validation_macro_f1_percent_mean": float(
                no_eca["val_macro_f1_mean"] * 100.0
            ),
            "no_eca_validation_macro_f1_percent_sample_std": float(
                no_eca["val_macro_f1_sample_std"] * 100.0
            ),
            "eca_validation_macro_f1_percent_mean": float(
                with_eca["val_macro_f1_mean"] * 100.0
            ),
            "eca_validation_macro_f1_percent_sample_std": float(
                with_eca["val_macro_f1_sample_std"] * 100.0
            ),
            "no_eca_minus_eca_macro_f1_percentage_points": float(
                (no_eca["val_macro_f1_mean"] - with_eca["val_macro_f1_mean"])
                * 100.0
            ),
        },
        "final_test": {
            "seeds": [42, 123, 2026],
            "test_samples_per_seed": 25943,
            "accuracy_percent_mean": float(
                final_row["test_accuracy_percent_mean"]
            ),
            "accuracy_percent_sample_std": float(
                final_row["test_accuracy_percent_sample_std"]
            ),
            "macro_f1_percent_mean": float(
                final_row["test_macro_f1_percent_mean"]
            ),
            "macro_f1_percent_sample_std": float(
                final_row["test_macro_f1_percent_sample_std"]
            ),
            "balanced_accuracy_percent_mean": float(
                final_row["test_balanced_accuracy_percent_mean"]
            ),
            "balanced_accuracy_percent_sample_std": float(
                final_row["test_balanced_accuracy_percent_sample_std"]
            ),
        },
        "saved_resnet18_baseline": {
            "training_runs": 1,
            "accuracy_percent": float(
                baseline_row["test_accuracy_percent_mean"]
            ),
            "macro_f1_percent": float(
                baseline_row["test_macro_f1_percent_mean"]
            ),
            "balanced_accuracy_percent": float(
                baseline_row["test_balanced_accuracy_percent_mean"]
            ),
        },
        "final_minus_baseline": {
            "accuracy_percentage_points": float(accuracy_gain),
            "macro_f1_percentage_points": float(macro_f1_gain),
            "balanced_accuracy_percentage_points": float(
                balanced_accuracy_gain
            ),
            "accuracy_all_three_holm_adjusted_p_below_0_05": bool(
                (mcnemar["holm_adjusted_p_value"] < 0.05).all()
            ),
            "accuracy_largest_holm_adjusted_p": float(
                mcnemar["holm_adjusted_p_value"].max()
            ),
            "macro_f1_all_three_holm_adjusted_p_below_0_05": bool(
                (randomization["holm_adjusted_p_value"] < 0.05).all()
            ),
            "macro_f1_largest_holm_adjusted_p": float(
                randomization["holm_adjusted_p_value"].max()
            ),
            "accuracy_all_bootstrap_ci_lower_above_zero": bool(
                (bootstrap["accuracy_delta_ci_lower"] > 0).all()
            ),
            "macro_f1_all_bootstrap_ci_lower_above_zero": bool(
                (bootstrap["macro_f1_delta_ci_lower"] > 0).all()
            ),
        },
        "complexity_and_mps": {
            "resnet18_parameter_count": int(baseline_row["parameter_count"]),
            "final_parameter_count": int(final_row["parameter_count"]),
            "parameter_reduction_percent": float(parameter_reduction),
            "state_dict_footprint_reduction_percent": float(
                state_dict_reduction
            ),
            "flops_change_percent": float(flops_change),
            "mps_batch1_latency_change_percent": float(latency_change),
            "mps_batch128_throughput_change_percent": float(
                throughput_change
            ),
            "hardware": sources["efficiency"]["hardware"],
        },
        "training": {
            "epochs_per_seed": int(training["epochs_per_seed"]),
            "best_epochs": training["best_epoch_by_seed"],
            "training_minutes_per_seed_mean": float(
                training["aggregate"]["training_minutes_per_seed"]["mean"]
            ),
            "training_minutes_per_seed_sample_std": float(
                training["aggregate"]["training_minutes_per_seed"][
                    "sample_std"
                ]
            ),
        },
        "error_analysis": {
            "samples_wrong_at_least_one_seed": int(
                error_summary["samples_wrong_at_least_one_seed"]
            ),
            "samples_wrong_all_three_seeds": int(
                error_summary["samples_wrong_all_three_seeds"]
            ),
            "samples_unanimously_wrong_same_class": int(
                error_summary["samples_unanimously_wrong_same_class"]
            ),
            "highest_common_error_rate_classes": class_difficulties,
            "top_unanimous_error_pairs": error_pairs,
        },
        "scope_limits": [
            "The paired tests are conditional on saved models; only one ResNet18 training run is available.",
            "The fixed benchmark test split had historical baseline access and is not an untouched external holdout.",
            "No external-dataset generalization experiment was performed.",
            "The project classifies wafer bin maps; it does not localize AOI defects or estimate micron-scale geometry.",
            "Parameter count is lower, but FLOPs and M1/MPS latency are higher than ResNet18.",
            "No controlled multi-seed ablation isolates the high-resolution architecture change from all other factors.",
            "The final model does not contain ECA and does not use Focal Loss.",
        ],
    }


def build_claims_table(evidence: dict) -> pd.DataFrame:
    selection = evidence["selection"]
    final_test = evidence["final_test"]
    comparison = evidence["final_minus_baseline"]
    efficiency = evidence["complexity_and_mps"]
    errors = evidence["error_analysis"]
    sources = {
        "scope": "00_项目管理/20260729_最终模型冻结清单.json",
        "selection": (
            "04_实验/metrics/20260729_highres_ce_eca_multiseed_validation/"
            "aggregate_validation.csv"
        ),
        "test": (
            "04_实验/metrics/20260729_highres_ce_multiseed_final_test/"
            "test_aggregate_summary.json"
        ),
        "significance": (
            "04_实验/metrics/20260729_final_vs_resnet18_statistical_"
            "significance/analysis_summary.json"
        ),
        "efficiency": (
            "04_实验/benchmarks/20260729_final_model_complexity_mps_"
            "benchmark/model_complexity.csv"
        ),
        "errors": (
            "04_实验/metrics/20260729_highres_ce_multiseed_error_analysis/"
            "analysis_summary.json"
        ),
    }
    rows = [
        {
            "claim_id": "C01",
            "topic": "task_scope",
            "status": "supported",
            "paper_ready_claim": (
                "This study addresses nine-class failure-pattern classification "
                "on WM-811K wafer bin maps."
            ),
            "key_value": "9 classes; 25,943 test maps per seed",
            "evidence_source": sources["scope"],
            "interpretation_limit": "Classification only; not AOI object detection.",
        },
        {
            "claim_id": "C02",
            "topic": "final_model",
            "status": "supported",
            "paper_ready_claim": (
                "The selected model is HighRes ShuffleNetV2 trained with "
                "CrossEntropyLoss and without ECA."
            ),
            "key_value": "1,262,397 parameters",
            "evidence_source": sources["scope"],
            "interpretation_limit": "Do not describe ECA or Focal Loss as final components.",
        },
        {
            "claim_id": "C03",
            "topic": "validation_selection",
            "status": "supported_with_scope_limit",
            "paper_ready_claim": (
                "The no-ECA variant was selected because its three-seed mean "
                "validation Macro-F1 was slightly higher than the ECA variant."
            ),
            "key_value": (
                f"{selection['no_eca_validation_macro_f1_percent_mean']:.4f}% "
                f"vs {selection['eca_validation_macro_f1_percent_mean']:.4f}% "
                f"(+{selection['no_eca_minus_eca_macro_f1_percentage_points']:.4f} pp)"
            ),
            "evidence_source": sources["selection"],
            "interpretation_limit": "Selection evidence, not a general proof that ECA is harmful.",
        },
        {
            "claim_id": "C04",
            "topic": "final_test_performance",
            "status": "supported",
            "paper_ready_claim": (
                "Across three random seeds, the final model achieved "
                f"{final_test['accuracy_percent_mean']:.4f}% +/- "
                f"{final_test['accuracy_percent_sample_std']:.4f}% accuracy "
                f"and {final_test['macro_f1_percent_mean']:.4f}% +/- "
                f"{final_test['macro_f1_percent_sample_std']:.4f}% Macro-F1."
            ),
            "key_value": "mean +/- sample SD across seeds 42, 123, and 2026",
            "evidence_source": sources["test"],
            "interpretation_limit": "Fixed WM-811K benchmark test split.",
        },
        {
            "claim_id": "C05",
            "topic": "baseline_gain",
            "status": "supported_with_scope_limit",
            "paper_ready_claim": (
                "Relative to the saved ResNet18 baseline, the final model "
                f"improved mean accuracy by {comparison['accuracy_percentage_points']:.4f} "
                f"percentage points and mean Macro-F1 by "
                f"{comparison['macro_f1_percentage_points']:.4f} percentage points."
            ),
            "key_value": "one baseline run vs three final-model runs",
            "evidence_source": f"{sources['test']}; {sources['significance']}",
            "interpretation_limit": "Conditional on saved models; baseline training-seed uncertainty is unavailable.",
        },
        {
            "claim_id": "C06",
            "topic": "paired_significance",
            "status": "supported_with_scope_limit",
            "paper_ready_claim": (
                "All three paired accuracy and Macro-F1 comparisons remained "
                "significant after Holm correction at alpha=0.05."
            ),
            "key_value": (
                f"largest adjusted p: accuracy {comparison['accuracy_largest_holm_adjusted_p']:.6f}; "
                f"Macro-F1 {comparison['macro_f1_largest_holm_adjusted_p']:.6f}"
            ),
            "evidence_source": sources["significance"],
            "interpretation_limit": "Sample-level inference conditional on fixed trained models.",
        },
        {
            "claim_id": "C07",
            "topic": "parameter_efficiency",
            "status": "supported",
            "paper_ready_claim": (
                "The final model reduced parameter count by "
                f"{efficiency['parameter_reduction_percent']:.2f}% relative to ResNet18."
            ),
            "key_value": (
                f"{efficiency['final_parameter_count']:,} vs "
                f"{efficiency['resnet18_parameter_count']:,} parameters"
            ),
            "evidence_source": sources["efficiency"],
            "interpretation_limit": "Use 'parameter-efficient' or 'compact', not automatically 'faster'.",
        },
        {
            "claim_id": "C08",
            "topic": "state_dict_footprint",
            "status": "supported",
            "paper_ready_claim": (
                "The serialized state-dict tensor footprint was reduced by "
                f"{efficiency['state_dict_footprint_reduction_percent']:.2f}%."
            ),
            "key_value": "4.8778 MiB vs 42.6655 MiB",
            "evidence_source": sources["efficiency"],
            "interpretation_limit": "State-dict tensor footprint, not a complete deployed application size.",
        },
        {
            "claim_id": "C09",
            "topic": "compute_speed",
            "status": "not_supported",
            "paper_ready_claim": "Do not claim that the final model is computationally faster than ResNet18.",
            "key_value": (
                f"FLOPs {efficiency['flops_change_percent']:+.2f}%; "
                f"M1/MPS batch-1 latency {efficiency['mps_batch1_latency_change_percent']:+.2f}%"
            ),
            "evidence_source": sources["efficiency"],
            "interpretation_limit": "The final model was slower on the measured Apple M1 Pro MPS setup.",
        },
        {
            "claim_id": "C10",
            "topic": "persistent_errors",
            "status": "supported",
            "paper_ready_claim": (
                "The main persistent weaknesses involved Loc, Scratch, and "
                "Edge-Loc patterns, often confused with none."
            ),
            "key_value": (
                f"{errors['samples_wrong_all_three_seeds']} samples wrong in all "
                f"three seeds; {errors['samples_unanimously_wrong_same_class']} "
                "with the same wrong class"
            ),
            "evidence_source": sources["errors"],
            "interpretation_limit": "Post-test descriptive analysis; no subsequent model tuning.",
        },
        {
            "claim_id": "C11",
            "topic": "external_generalization",
            "status": "not_supported",
            "paper_ready_claim": "Do not claim external-dataset or factory-line generalization.",
            "key_value": "No external validation dataset evaluated",
            "evidence_source": sources["scope"],
            "interpretation_limit": "Discuss deployment only as future work.",
        },
        {
            "claim_id": "C12",
            "topic": "independent_test",
            "status": "not_supported",
            "paper_ready_claim": "Do not call the test split an untouched independent holdout.",
            "key_value": "Historical baseline test access is documented",
            "evidence_source": sources["scope"],
            "interpretation_limit": "Call it the fixed WM-811K benchmark test split.",
        },
        {
            "claim_id": "C13",
            "topic": "highres_causal_ablation",
            "status": "not_supported",
            "paper_ready_claim": (
                "Do not attribute the entire gain causally to the high-resolution "
                "architecture modification."
            ),
            "key_value": "No controlled multi-seed standard-ShuffleNetV2 + CE ablation",
            "evidence_source": sources["selection"],
            "interpretation_limit": "Present the result as a system-level comparison.",
        },
        {
            "claim_id": "C14",
            "topic": "aoi_detection",
            "status": "not_supported",
            "paper_ready_claim": "Do not claim defect localization, bounding boxes, or micron-scale AOI detection.",
            "key_value": "Input is a wafer bin map; output is one of nine classes",
            "evidence_source": sources["scope"],
            "interpretation_limit": "Use 'wafer failure-pattern classification'.",
        },
    ]
    return pd.DataFrame(rows)


def render_writing_notes(evidence: dict, claims: pd.DataFrame) -> str:
    selection = evidence["selection"]
    final_test = evidence["final_test"]
    baseline = evidence["saved_resnet18_baseline"]
    comparison = evidence["final_minus_baseline"]
    efficiency = evidence["complexity_and_mps"]
    training = evidence["training"]
    errors = evidence["error_analysis"]
    difficult = errors["highest_common_error_rate_classes"]
    pairs = errors["top_unanimous_error_pairs"]

    supported = claims.loc[claims["status"] != "not_supported"]
    unsupported = claims.loc[claims["status"] == "not_supported"]
    supported_lines = "\n".join(
        f"{index}. {row['paper_ready_claim']}"
        for index, (_, row) in enumerate(supported.iterrows(), start=1)
    )
    unsupported_lines = "\n".join(
        f"{index}. {row['paper_ready_claim']}"
        for index, (_, row) in enumerate(unsupported.iterrows(), start=1)
    )
    difficult_lines = "\n".join(
        (
            f"- {item['class_name']}: support={item['support']}, mean recall="
            f"{item['mean_recall_percent']:.2f}%, common error rate="
            f"{item['common_error_rate_percent']:.2f}%"
        )
        for item in difficult
    )
    pair_lines = "\n".join(
        f"- {item['true_class']} -> {item['predicted_class']}: {item['sample_count']} samples"
        for item in pairs
    )

    return f"""# 论文写作数据底稿

> 本文件只汇总已冻结、已审计的实验结果。不得据此继续调模型，也不得超出文末的表述边界。

## 1. 课题最终定位

- 任务：WM-811K 晶圆 Bin Map 九分类失效模式识别。
- 输入：64x64 单通道晶圆 Bin Map；像素值表示晶粒测试状态。
- 输出：Center、Donut、Edge-Loc、Edge-Ring、Loc、Near-full、Random、Scratch、none 共九类。
- 最终模型：HighRes ShuffleNetV2 + CrossEntropyLoss，不含 ECA，不使用 Focal Loss。
- 本课题不是 AOI 光学图像目标检测，不输出缺陷框，也不能声称检测微米级缺陷位置。

## 2. 模型选择依据

模型选择只使用验证集，随机种子为 42、123、2026：

- 无 ECA：验证 Macro-F1 = {selection['no_eca_validation_macro_f1_percent_mean']:.4f}% +/- {selection['no_eca_validation_macro_f1_percent_sample_std']:.4f}%。
- 有 ECA：验证 Macro-F1 = {selection['eca_validation_macro_f1_percent_mean']:.4f}% +/- {selection['eca_validation_macro_f1_percent_sample_std']:.4f}%。
- 无 ECA 高 {selection['no_eca_minus_eca_macro_f1_percentage_points']:.4f} 个百分点，因此选择无 ECA 版本。
- 该结果只能说明本项目三次验证运行中无 ECA 的均值略高，不能推广成“ECA 普遍无效”。

## 3. 正式测试结果

最终模型三随机种子正式测试：

| 指标 | 均值 +/- 样本标准差 |
|---|---:|
| Accuracy | {final_test['accuracy_percent_mean']:.4f}% +/- {final_test['accuracy_percent_sample_std']:.4f}% |
| Macro-F1 | {final_test['macro_f1_percent_mean']:.4f}% +/- {final_test['macro_f1_percent_sample_std']:.4f}% |
| Balanced Accuracy | {final_test['balanced_accuracy_percent_mean']:.4f}% +/- {final_test['balanced_accuracy_percent_sample_std']:.4f}% |

已保存的单次 ResNet18 基线：Accuracy={baseline['accuracy_percent']:.4f}%，Macro-F1={baseline['macro_f1_percent']:.4f}%，Balanced Accuracy={baseline['balanced_accuracy_percent']:.4f}%。

最终模型相对该基线的均值变化：

- Accuracy：+{comparison['accuracy_percentage_points']:.4f} 个百分点。
- Macro-F1：+{comparison['macro_f1_percentage_points']:.4f} 个百分点。
- Balanced Accuracy：+{comparison['balanced_accuracy_percentage_points']:.4f} 个百分点。
- 三个种子的 Accuracy 精确 McNemar 检验经 Holm 校正后均 p<0.05，最大校正 p={comparison['accuracy_largest_holm_adjusted_p']:.6f}。
- 三个种子的 Macro-F1 配对随机化检验经 Holm 校正后均 p<0.05，最大校正 p={comparison['macro_f1_largest_holm_adjusted_p']:.6f}。
- 所有 Accuracy 和 Macro-F1 的 95% 分层配对 Bootstrap 区间下界均大于 0。

统计结论只针对已保存模型在同一固定测试集上的配对预测。由于 ResNet18 只有一次训练结果，不能声称已经完整估计两种架构的随机种子不确定性。

## 4. 参数量、计算量与速度

- 参数量：1,262,397，对比 ResNet18 的 11,174,857，减少 {efficiency['parameter_reduction_percent']:.2f}%。
- state-dict 张量占用：4.8778 MiB，对比 42.6655 MiB，减少 {efficiency['state_dict_footprint_reduction_percent']:.2f}%。
- Conv/Linear FLOPs：比 ResNet18 增加 {efficiency['flops_change_percent']:.2f}%。
- Apple M1 Pro / MPS，batch=1 前向延迟：比 ResNet18 增加 {efficiency['mps_batch1_latency_change_percent']:.2f}%。
- Apple M1 Pro / MPS，batch=128 吞吐量：比 ResNet18 变化 {efficiency['mps_batch128_throughput_change_percent']:.2f}%。

因此论文可以写“parameter-efficient”或“compact model”，不能写“推理更快”或“计算量更低”。嵌入式/FPGA 部署尚未实测，只能列为未来工作。

## 5. 训练稳定性

- 每个种子训练 30 轮。
- 最佳轮次：seed 42={training['best_epochs']['42']}，seed 123={training['best_epochs']['123']}，seed 2026={training['best_epochs']['2026']}。
- 单种子训练时间：{training['training_minutes_per_seed_mean']:.1f} +/- {training['training_minutes_per_seed_sample_std']:.1f} 分钟。

## 6. 错误分析

- 25,943 个测试样本中，652 个至少被一个种子分错。
- 381 个样本被三个种子全部分错，其中 359 个被三个种子一致分到同一错误类别。

共同错误率最高的类别：

{difficult_lines}

数量最多的一致误分类方向：

{pair_lines}

这部分只能作为冻结模型后的描述性分析，不得用来继续调整模型或阈值。

## 7. 可以直接写入论文的结论

{supported_lines}

## 8. 禁止或必须降级的表述

{unsupported_lines}

## 9. 必须写入局限性

1. 固定 WM-811K benchmark test split 在项目早期已有基线访问记录，不能称为全程未接触的独立测试集。
2. 没有外部数据集验证，不能声称跨产线、跨设备或工厂落地泛化。
3. ResNet18 只有一次训练运行，配对显著性检验不等于完整的架构级随机种子检验。
4. 缺少“标准 ShuffleNetV2 + CE”三随机种子的受控对照，不能把全部性能增益单独归因于高分辨率改造。
5. 参数量显著减少，但 FLOPs 和 M1/MPS 实测延迟增加，不能把“轻参数”写成“高速度”。
6. 类别极不平衡，none 类占多数；Loc、Scratch、Edge-Loc 仍是主要困难类别。

## 10. 写论文时优先使用的现有表图

- 总结果表：`05_结果/tables/table_final_overall_results.csv`
- 结论证据表：`05_结果/tables/table_final_claims_evidence.csv`
- 正式测试表：`05_结果/tables/table_final_highres_ce_multiseed_test.csv`
- 验证模型选择表：`05_结果/tables/table_multiseed_validation_model_selection.csv`
- 统计显著性表：`05_结果/tables/table_final_vs_resnet18_statistical_significance.csv`
- 参数量与 MPS 对比表：`05_结果/tables/table_final_model_complexity_mps.csv`
- 三种子混淆矩阵：`05_结果/figures/model_results/final_highres_ce_multiseed_confusion_matrix_mean.png`
- 三种子训练曲线：`05_结果/figures/model_results/final_highres_ce_multiseed_training_curves.png`
- 参数量与速度图：`05_结果/figures/model_results/final_model_complexity_mps_comparison.png`
- 统计显著性图：`05_结果/figures/model_results/final_vs_resnet18_statistical_significance.png`
- 典型错误样本：`05_结果/figures/model_results/final_highres_ce_unanimous_error_examples.png`
"""


def write_csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, float_format="%.10f")


def write_outputs(
    output_root: Path,
    overall: pd.DataFrame,
    claims: pd.DataFrame,
    evidence: dict,
    input_hashes: dict[str, str],
):
    paths = output_paths(output_root)
    metrics_dir = paths["metrics_dir"]
    table_dir = paths["table_dir"]
    manuscript_dir = paths["manuscript_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=False)
    table_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    write_csv(overall, table_dir / OVERALL_TABLE_NAME)
    write_csv(claims, table_dir / CLAIMS_TABLE_NAME)
    (metrics_dir / "final_evidence_summary.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (manuscript_dir / WRITING_NOTES_NAME).write_text(
        render_writing_notes(evidence, claims),
        encoding="utf-8",
    )

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
        "training_performed": False,
        "model_inference_performed": False,
        "wafer_image_dataset_loaded": False,
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
    }
    (metrics_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_summary(evidence: dict):
    final_test = evidence["final_test"]
    comparison = evidence["final_minus_baseline"]
    efficiency = evidence["complexity_and_mps"]
    print("Input validation: PASS")
    print("No training or inference was performed.")
    print(
        f"Final test: accuracy {final_test['accuracy_percent_mean']:.4f}% +/- "
        f"{final_test['accuracy_percent_sample_std']:.4f}%, Macro-F1 "
        f"{final_test['macro_f1_percent_mean']:.4f}% +/- "
        f"{final_test['macro_f1_percent_sample_std']:.4f}%"
    )
    print(
        f"Vs ResNet18: accuracy {comparison['accuracy_percentage_points']:+.4f} pp, "
        f"Macro-F1 {comparison['macro_f1_percentage_points']:+.4f} pp, "
        f"parameters -{efficiency['parameter_reduction_percent']:.2f}%"
    )
    print(
        f"Efficiency boundary: FLOPs {efficiency['flops_change_percent']:+.2f}%, "
        f"M1/MPS batch-1 latency "
        f"{efficiency['mps_batch1_latency_change_percent']:+.2f}%"
    )


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.run:
        ensure_outputs_are_available(output_root)

    sources = load_and_validate_sources()
    overall = build_overall_table(sources)
    evidence = build_evidence_summary(sources, overall)
    claims = build_claims_table(evidence)
    print_summary(evidence)

    if args.check_inputs:
        print("Preflight complete. No files were created.")
        return

    write_outputs(
        output_root,
        overall,
        claims,
        evidence,
        sources["input_hashes"],
    )
    paths = output_paths(output_root)
    print(f"Overall table saved to: {paths['table_dir'] / OVERALL_TABLE_NAME}")
    print(f"Claims table saved to: {paths['table_dir'] / CLAIMS_TABLE_NAME}")
    print(f"Writing notes saved to: {paths['manuscript_dir'] / WRITING_NOTES_NAME}")


if __name__ == "__main__":
    main()
