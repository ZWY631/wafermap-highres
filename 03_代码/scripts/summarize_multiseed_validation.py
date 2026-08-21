#!/usr/bin/env python3
"""Summarize paired multi-seed validation experiments before final testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import IMAGE_SIZE, WM811K_CLASS_NAMES
from wafermap.paths import EXPERIMENT_DIR, PROJECT_ROOT, RESULT_DIR


EVALUATION_ID = "20260729_highres_ce_eca_multiseed_validation"
OUTPUT_DIR = EXPERIMENT_DIR / "metrics" / EVALUATION_ID
PAPER_TABLE = (
    RESULT_DIR
    / "tables"
    / "table_multiseed_validation_model_selection.csv"
)

SEEDS = (42, 123, 2026)
EXPECTED_EPOCHS = 30

ARCHITECTURES = (
    {
        "architecture_id": "highres_ce_no_eca",
        "architecture": "HighRes ShuffleNetV2 + CE (no ECA)",
        "model_name": "ShuffleNetV2HighRes",
        "base_run_name": "shufflenet_v2_highres_ce_full",
    },
    {
        "architecture_id": "highres_eca_ce",
        "architecture": "HighRes ShuffleNetV2 + ECA + CE",
        "model_name": "ShuffleNetV2HighResECA",
        "base_run_name": "shufflenet_v2_highres_eca_ce_full",
    },
)

REQUIRED_COLUMNS = (
    "epoch",
    "val_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize the six paired multi-seed validation "
            "runs used for final model selection."
        )
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate all inputs and output paths without creating files.",
    )
    return parser.parse_args()


def run_name(base_run_name: str, seed: int) -> str:
    if seed == 42:
        return base_run_name
    return f"{base_run_name}_seed{seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_outputs_are_available():
    existing = [path for path in (OUTPUT_DIR, PAPER_TABLE) if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing validation summaries:\n"
            f"{formatted}"
        )


def validate_history(history_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not history_file.is_file():
        raise FileNotFoundError(f"Missing history CSV: {history_file}")

    history = pd.read_csv(history_file)
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in history.columns
    ]
    if missing_columns:
        raise ValueError(
            f"History CSV {history_file} is missing columns: {missing_columns}"
        )

    if len(history) != EXPECTED_EPOCHS:
        raise ValueError(
            f"Expected {EXPECTED_EPOCHS} epochs in {history_file}, "
            f"found {len(history)}."
        )

    epochs = history["epoch"].astype(int).to_numpy()
    expected = np.arange(1, EXPECTED_EPOCHS + 1)
    if not np.array_equal(epochs, expected):
        raise ValueError(f"Epoch sequence is incomplete in {history_file}.")

    if history[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"Validation history contains missing values: {history_file}")

    best_index = history["val_macro_f1"].idxmax()
    return history, history.loc[best_index]


def validate_checkpoint(
    checkpoint_file: Path,
    expected_run_name: str,
    expected_model_name: str,
    expected_seed: int,
    best_row: pd.Series,
) -> dict:
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    expected_values = {
        "run_name": expected_run_name,
        "model_name": expected_model_name,
        "loss_name": "CrossEntropyLoss",
        "image_size": IMAGE_SIZE,
        "random_seed": expected_seed,
        "class_names": list(WM811K_CLASS_NAMES),
    }
    for key, expected_value in expected_values.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint field mismatch for {checkpoint_file}: "
                f"{key}={actual_value!r}, expected {expected_value!r}."
            )

    if int(checkpoint.get("epoch", -1)) != int(best_row["epoch"]):
        raise ValueError(
            f"Best epoch mismatch between CSV and checkpoint: {checkpoint_file}"
        )

    checkpoint_macro_f1 = float(checkpoint.get("best_val_macro_f1", np.nan))
    if not np.isclose(
        checkpoint_macro_f1,
        float(best_row["val_macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Best Macro-F1 mismatch between CSV and checkpoint: {checkpoint_file}"
        )

    return checkpoint


def collect_rows() -> pd.DataFrame:
    rows = []

    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            current_run = run_name(architecture["base_run_name"], seed)
            history_file = (
                EXPERIMENT_DIR / "metrics" / f"{current_run}_history.csv"
            )
            checkpoint_file = (
                EXPERIMENT_DIR
                / "checkpoints"
                / current_run
                / "best.pt"
            )

            _, best_row = validate_history(history_file)
            checkpoint = validate_checkpoint(
                checkpoint_file=checkpoint_file,
                expected_run_name=current_run,
                expected_model_name=architecture["model_name"],
                expected_seed=seed,
                best_row=best_row,
            )

            rows.append(
                {
                    "architecture_id": architecture["architecture_id"],
                    "architecture": architecture["architecture"],
                    "run_name": current_run,
                    "seed": seed,
                    "best_epoch": int(best_row["epoch"]),
                    "val_accuracy": float(best_row["val_accuracy"]),
                    "val_macro_f1": float(best_row["val_macro_f1"]),
                    "val_balanced_accuracy": float(
                        best_row["val_balanced_accuracy"]
                    ),
                    "parameter_count": int(checkpoint["parameter_count"]),
                    "history_file": str(history_file.relative_to(PROJECT_ROOT)),
                    "checkpoint_file": str(
                        checkpoint_file.relative_to(PROJECT_ROOT)
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint_file),
                }
            )

    per_seed = pd.DataFrame(rows)
    expected_rows = len(ARCHITECTURES) * len(SEEDS)
    if len(per_seed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} validated runs, found {len(per_seed)}."
        )
    return per_seed


def aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
    )
    rows = []

    for architecture_id, group in per_seed.groupby(
        "architecture_id", sort=False
    ):
        row = {
            "architecture_id": architecture_id,
            "architecture": group["architecture"].iloc[0],
            "n_seeds": len(group),
            "seeds": ",".join(str(seed) for seed in group["seed"]),
            "parameter_count": int(group["parameter_count"].iloc[0]),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_std"] = float(group[metric].std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows)


def paired_differences(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
    )
    pivot = per_seed.pivot(
        index="seed",
        columns="architecture_id",
        values=list(metric_columns),
    )
    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for metric in metric_columns:
            difference = (
                pivot.loc[seed, (metric, "highres_eca_ce")]
                - pivot.loc[seed, (metric, "highres_ce_no_eca")]
            )
            row[f"eca_minus_no_eca_{metric}"] = float(difference)
        rows.append(row)

    paired = pd.DataFrame(rows)
    mean_row = {"seed": "mean"}
    std_row = {"seed": "sample_std"}
    for column in paired.columns[1:]:
        mean_row[column] = float(paired[column].mean())
        std_row[column] = float(paired[column].std(ddof=1))

    return pd.concat(
        [paired, pd.DataFrame([mean_row, std_row])],
        ignore_index=True,
    )


def build_paper_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in aggregate.iterrows():
        output_row = {
            "architecture": row["architecture"],
            "seeds": row["seeds"],
            "parameters": int(row["parameter_count"]),
        }
        for metric, label in (
            ("val_accuracy", "accuracy_percent_mean_sd"),
            ("val_macro_f1", "macro_f1_percent_mean_sd"),
            (
                "val_balanced_accuracy",
                "balanced_accuracy_percent_mean_sd",
            ),
        ):
            mean = row[f"{metric}_mean"] * 100.0
            std = row[f"{metric}_sample_std"] * 100.0
            output_row[label] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(output_row)
    return pd.DataFrame(rows)


def write_outputs(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
):
    temp_root = Path(
        tempfile.mkdtemp(prefix=f"{EVALUATION_ID}_", dir=PROJECT_ROOT / "tmp")
    )
    staged_output_dir = temp_root / EVALUATION_ID
    staged_output_dir.mkdir(parents=True)
    staged_table = temp_root / PAPER_TABLE.name

    try:
        per_seed.to_csv(staged_output_dir / "per_seed_validation.csv", index=False)
        aggregate.to_csv(staged_output_dir / "aggregate_validation.csv", index=False)
        paired.to_csv(staged_output_dir / "paired_eca_minus_no_eca.csv", index=False)

        selected = aggregate.sort_values(
            "val_macro_f1_mean", ascending=False
        ).iloc[0]
        summary = {
            "evaluation_id": EVALUATION_ID,
            "selection_split": "validation",
            "primary_selection_metric": "val_macro_f1_mean",
            "sample_standard_deviation_ddof": 1,
            "seeds": list(SEEDS),
            "selected_architecture_id": selected["architecture_id"],
            "selected_architecture": selected["architecture"],
            "selected_val_macro_f1_mean": float(
                selected["val_macro_f1_mean"]
            ),
            "selected_val_macro_f1_sample_std": float(
                selected["val_macro_f1_sample_std"]
            ),
            "current_paired_selection_used_test_predictions": False,
            "test_split_was_untouched_during_entire_project": False,
        }
        (staged_output_dir / "selection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        build_paper_table(aggregate).to_csv(staged_table, index=False)

        OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_output_dir, OUTPUT_DIR)
        try:
            os.replace(staged_table, PAPER_TABLE)
        except Exception:
            shutil.rmtree(OUTPUT_DIR)
            raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def print_summary(per_seed: pd.DataFrame, aggregate: pd.DataFrame):
    print("多随机种子验证结果核对通过")
    print(f"已核对训练次数：{len(per_seed)}")
    for _, row in aggregate.iterrows():
        print(row["architecture"])
        print(
            "  Macro-F1: "
            f"{row['val_macro_f1_mean'] * 100:.4f}% +/- "
            f"{row['val_macro_f1_sample_std'] * 100:.4f}%"
        )


def main():
    args = parse_args()
    ensure_outputs_are_available()
    per_seed = collect_rows()
    aggregate = aggregate_metrics(per_seed)
    paired = paired_differences(per_seed)
    print_summary(per_seed, aggregate)

    if args.preflight:
        print("预检查模式：未创建任何文件。")
        return

    write_outputs(per_seed, aggregate, paired)
    print(f"原始统计：{OUTPUT_DIR}")
    print(f"论文表格：{PAPER_TABLE}")


if __name__ == "__main__":
    main()
