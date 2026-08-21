#!/usr/bin/env python3
"""Plot final three-seed training histories without training or inference."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_MANIFEST = (
    PROJECT_ROOT / "00_项目管理" / "20260729_最终模型冻结清单.json"
)
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "c682922b1c291f03a41a81e5d15ebfe08e4456e77c816f7943041fc11fcef8db"
)

ANALYSIS_ID = "20260729_highres_ce_multiseed_training_curves"
EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_EPOCHS = 30

HISTORY_COLUMNS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "train_balanced_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
    "epoch_seconds",
)

SCORE_COLUMNS = (
    "train_accuracy",
    "train_macro_f1",
    "train_balanced_accuracy",
    "val_accuracy",
    "val_macro_f1",
    "val_balanced_accuracy",
)

AGGREGATE_COLUMNS = tuple(column for column in HISTORY_COLUMNS if column != "epoch")

METRIC_FILENAMES = (
    "combined_training_history.csv",
    "epoch_aggregate.csv",
    "best_epoch_by_seed.csv",
    "training_summary.json",
    "analysis_manifest.json",
)

FIGURE_STEM = "final_highres_ce_multiseed_training_curves"
PAPER_TABLE_NAME = "table_final_highres_ce_training_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and plot the saved 30-epoch histories for the final "
            "HighRes ShuffleNetV2 + CE model. No training or inference is run."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate histories and best-epoch records without writing files.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Create the final training-curve figure and summary tables.",
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
            paths["figure_dir"] / f"{FIGURE_STEM}.png",
            paths["figure_dir"] / f"{FIGURE_STEM}.pdf",
            paths["table_dir"] / PAPER_TABLE_NAME,
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
            "Refusing to overwrite existing training-curve outputs:\n"
            f"{formatted}"
        )


def load_freeze_manifest() -> dict:
    if not FREEZE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing freeze manifest: {FREEZE_MANIFEST}")
    if sha256_file(FREEZE_MANIFEST) != EXPECTED_FREEZE_MANIFEST_SHA256:
        raise ValueError("Final model freeze manifest hash has changed.")

    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    expected_fields = {
        "status": "frozen_before_test",
        "selected_architecture_id": "highres_ce_no_eca",
        "model_name": "ShuffleNetV2HighRes",
        "loss_name": "CrossEntropyLoss",
        "sample_standard_deviation_ddof": 1,
    }
    for key, expected_value in expected_fields.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"Freeze manifest mismatch: {key}={manifest.get(key)!r}, "
                f"expected {expected_value!r}."
            )
    if tuple(manifest.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("Freeze manifest seed list is incorrect.")
    return manifest


def validate_history(
    history_file: Path,
    seed: int,
    checkpoint_info: dict,
) -> pd.DataFrame:
    if not history_file.is_file():
        raise FileNotFoundError(f"Missing training history: {history_file}")

    history = pd.read_csv(history_file)
    missing = sorted(set(HISTORY_COLUMNS) - set(history.columns))
    if missing:
        raise ValueError(f"Missing columns in {history_file}: {missing}")
    history = history[list(HISTORY_COLUMNS)].copy()

    if len(history) != EXPECTED_EPOCHS:
        raise ValueError(
            f"Expected {EXPECTED_EPOCHS} epochs in {history_file}, "
            f"found {len(history)}."
        )
    expected_epoch_sequence = np.arange(1, EXPECTED_EPOCHS + 1)
    actual_epoch_sequence = history["epoch"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_epoch_sequence, expected_epoch_sequence):
        raise ValueError(f"Epoch sequence is incomplete in {history_file}")

    numeric = history[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite values found in {history_file}")
    for column in SCORE_COLUMNS:
        if not history[column].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"Score outside [0, 1] in {history_file}: {column}")
    for column in ("learning_rate", "train_loss", "val_loss", "epoch_seconds"):
        if not history[column].gt(0.0).all():
            raise ValueError(f"Non-positive values in {history_file}: {column}")

    best_index = history["val_macro_f1"].idxmax()
    best_row = history.loc[best_index]
    if int(best_row["epoch"]) != int(checkpoint_info["best_epoch"]):
        raise ValueError(
            f"Best epoch differs from freeze manifest for seed {seed}: "
            f"history={int(best_row['epoch'])}, "
            f"manifest={checkpoint_info['best_epoch']}"
        )
    if not np.isclose(
        float(best_row["val_macro_f1"]),
        float(checkpoint_info["validation_macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"Best validation Macro-F1 differs from freeze manifest for seed {seed}"
        )

    history.insert(0, "run_name", checkpoint_info["run_name"])
    history.insert(0, "seed", seed)
    return history


def load_histories(manifest: dict) -> tuple[dict[int, pd.DataFrame], dict[str, str]]:
    checkpoint_by_seed = {
        int(item["seed"]): item for item in manifest.get("checkpoints", [])
    }
    if tuple(sorted(checkpoint_by_seed)) != tuple(sorted(EXPECTED_SEEDS)):
        raise ValueError("Freeze manifest checkpoint list is incomplete.")

    histories = {}
    input_hashes = {
        str(FREEZE_MANIFEST.relative_to(PROJECT_ROOT)): (
            EXPECTED_FREEZE_MANIFEST_SHA256
        )
    }
    reference_learning_rate = None

    for seed in EXPECTED_SEEDS:
        checkpoint_info = checkpoint_by_seed[seed]
        history_file = (
            PROJECT_ROOT
            / "04_实验"
            / "metrics"
            / f"{checkpoint_info['run_name']}_history.csv"
        )
        history = validate_history(history_file, seed, checkpoint_info)
        current_learning_rate = history["learning_rate"].to_numpy()
        if reference_learning_rate is None:
            reference_learning_rate = current_learning_rate
        elif not np.allclose(
            reference_learning_rate,
            current_learning_rate,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("Learning-rate schedules differ across seeds.")

        histories[seed] = history
        input_hashes[str(history_file.relative_to(PROJECT_ROOT))] = sha256_file(
            history_file
        )

    return histories, input_hashes


def aggregate_histories(histories: dict[int, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(histories.values(), ignore_index=True)
    rows = []
    for epoch, group in combined.groupby("epoch", sort=True):
        row = {"epoch": int(epoch)}
        for column in AGGREGATE_COLUMNS:
            values = group[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sample_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def build_best_epoch_summary(
    histories: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for seed in EXPECTED_SEEDS:
        history = histories[seed]
        best_row = history.loc[history["val_macro_f1"].idxmax()]
        rows.append(
            {
                "seed": seed,
                "run_name": best_row["run_name"],
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": float(best_row["val_loss"]),
                "best_val_accuracy": float(best_row["val_accuracy"]),
                "best_val_macro_f1": float(best_row["val_macro_f1"]),
                "best_val_balanced_accuracy": float(
                    best_row["val_balanced_accuracy"]
                ),
                "total_training_minutes": float(
                    history["epoch_seconds"].sum() / 60.0
                ),
                "mean_epoch_seconds": float(history["epoch_seconds"].mean()),
            }
        )
    return pd.DataFrame(rows)


def mean_sample_std(values: pd.Series) -> tuple[float, float]:
    numeric = values.to_numpy(dtype=np.float64)
    return float(numeric.mean()), float(numeric.std(ddof=1))


def format_mean_sd(mean: float, sample_std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} +/- {sample_std:.{decimals}f}"


def build_paper_table(best_summary: pd.DataFrame) -> pd.DataFrame:
    best_epoch_mean, best_epoch_std = mean_sample_std(best_summary["best_epoch"])
    val_loss_mean, val_loss_std = mean_sample_std(best_summary["best_val_loss"])
    val_accuracy_mean, val_accuracy_std = mean_sample_std(
        best_summary["best_val_accuracy"] * 100
    )
    macro_f1_mean, macro_f1_std = mean_sample_std(
        best_summary["best_val_macro_f1"] * 100
    )
    balanced_mean, balanced_std = mean_sample_std(
        best_summary["best_val_balanced_accuracy"] * 100
    )
    training_mean, training_std = mean_sample_std(
        best_summary["total_training_minutes"]
    )

    return pd.DataFrame(
        [
            {
                "model": "HighRes ShuffleNetV2 + CE (no ECA)",
                "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
                "epochs_per_seed": EXPECTED_EPOCHS,
                "best_epoch_mean_sd": format_mean_sd(
                    best_epoch_mean, best_epoch_std, decimals=2
                ),
                "best_val_loss_mean_sd": format_mean_sd(
                    val_loss_mean, val_loss_std, decimals=6
                ),
                "best_val_accuracy_percent_mean_sd": format_mean_sd(
                    val_accuracy_mean, val_accuracy_std
                ),
                "best_val_macro_f1_percent_mean_sd": format_mean_sd(
                    macro_f1_mean, macro_f1_std
                ),
                "best_val_balanced_accuracy_percent_mean_sd": format_mean_sd(
                    balanced_mean, balanced_std
                ),
                "training_minutes_per_seed_mean_sd": format_mean_sd(
                    training_mean, training_std, decimals=2
                ),
            }
        ]
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


def plot_mean_with_band(
    axis: plt.Axes,
    epochs: np.ndarray,
    aggregate: pd.DataFrame,
    metric: str,
    color: str,
    label: str,
    lower_bound: float | None = None,
):
    mean = aggregate[f"{metric}_mean"].to_numpy(dtype=np.float64)
    sample_std = aggregate[f"{metric}_sample_std"].to_numpy(dtype=np.float64)
    lower = mean - sample_std
    if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)
    upper = mean + sample_std
    axis.plot(epochs, mean, color=color, linewidth=2, label=label)
    axis.fill_between(epochs, lower, upper, color=color, alpha=0.16, linewidth=0)


def plot_training_curves(
    histories: dict[int, pd.DataFrame],
    aggregate: pd.DataFrame,
    best_summary: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
):
    configure_plot_style()
    epochs = aggregate["epoch"].to_numpy(dtype=np.int64)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.8), sharex=True)

    plot_mean_with_band(
        axes[0, 0], epochs, aggregate, "train_loss", "#4C78A8", "Train", 0.0
    )
    plot_mean_with_band(
        axes[0, 0], epochs, aggregate, "val_loss", "#D55E00", "Validation", 0.0
    )
    axes[0, 0].set_title("Cross-Entropy Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(frameon=False)

    plot_mean_with_band(
        axes[0, 1],
        epochs,
        aggregate,
        "train_macro_f1",
        "#4C78A8",
        "Train",
        0.0,
    )
    plot_mean_with_band(
        axes[0, 1],
        epochs,
        aggregate,
        "val_macro_f1",
        "#D55E00",
        "Validation",
        0.0,
    )
    axes[0, 1].set_title("Macro-F1")
    axes[0, 1].set_ylabel("Macro-F1")
    axes[0, 1].legend(frameon=False)

    plot_mean_with_band(
        axes[1, 0],
        epochs,
        aggregate,
        "val_accuracy",
        "#4C78A8",
        "Validation Accuracy",
        0.0,
    )
    plot_mean_with_band(
        axes[1, 0],
        epochs,
        aggregate,
        "val_balanced_accuracy",
        "#59A14F",
        "Validation Balanced Accuracy",
        0.0,
    )
    axes[1, 0].set_title("Validation Accuracy vs. Balanced Accuracy")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend(frameon=False, loc="lower right")

    seed_colors = {42: "#4C78A8", 123: "#D55E00", 2026: "#59A14F"}
    for seed in EXPECTED_SEEDS:
        history = histories[seed]
        best_row = best_summary.loc[best_summary["seed"] == seed].iloc[0]
        axes[1, 1].plot(
            history["epoch"],
            history["val_macro_f1"],
            color=seed_colors[seed],
            linewidth=1.8,
            label=f"Seed {seed} (best epoch {int(best_row['best_epoch'])})",
        )
        axes[1, 1].scatter(
            best_row["best_epoch"],
            best_row["best_val_macro_f1"],
            color=seed_colors[seed],
            edgecolor="white",
            linewidth=0.8,
            s=52,
            zorder=3,
        )
    axes[1, 1].set_title("Validation Macro-F1 by Random Seed")
    axes[1, 1].set_ylabel("Validation Macro-F1")
    axes[1, 1].legend(frameon=False, fontsize=9)

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.set_xticks([1, 5, 10, 15, 20, 25, 30])
        axis.grid(True, linestyle="--", alpha=0.28)
        axis.set_axisbelow(True)

    figure.suptitle(
        "Final High-Resolution ShuffleNetV2 + CE: Three-Seed Training Curves",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Solid lines show the mean; shaded bands show +/- 1 sample SD (n=3).",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=[0, 0.035, 1, 0.97])
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def build_summary(best_summary: pd.DataFrame) -> dict:
    metric_map = {
        "best_epoch": "best_epoch",
        "best_val_loss": "best_validation_loss",
        "best_val_accuracy": "best_validation_accuracy",
        "best_val_macro_f1": "best_validation_macro_f1",
        "best_val_balanced_accuracy": "best_validation_balanced_accuracy",
        "total_training_minutes": "training_minutes_per_seed",
    }
    aggregate = {}
    for column, output_name in metric_map.items():
        mean, sample_std = mean_sample_std(best_summary[column])
        aggregate[output_name] = {
            "mean": mean,
            "sample_std": sample_std,
        }

    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_type": "saved_training_history_summary",
        "training_performed": False,
        "inference_performed": False,
        "selection_split": "validation",
        "primary_selection_metric": "validation Macro-F1",
        "sample_standard_deviation_ddof": 1,
        "seeds": list(EXPECTED_SEEDS),
        "epochs_per_seed": EXPECTED_EPOCHS,
        "best_epoch_by_seed": {
            str(int(row["seed"])): int(row["best_epoch"])
            for _, row in best_summary.iterrows()
        },
        "aggregate": aggregate,
    }


def write_outputs(
    output_root: Path,
    histories: dict[int, pd.DataFrame],
    aggregate: pd.DataFrame,
    best_summary: pd.DataFrame,
    input_hashes: dict[str, str],
):
    paths = output_paths(output_root)
    metrics_dir = paths["metrics_dir"]
    figure_dir = paths["figure_dir"]
    table_dir = paths["table_dir"]

    metrics_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(histories.values(), ignore_index=True)
    combined.to_csv(
        metrics_dir / "combined_training_history.csv",
        index=False,
        float_format="%.10f",
    )
    aggregate.to_csv(
        metrics_dir / "epoch_aggregate.csv",
        index=False,
        float_format="%.10f",
    )
    best_summary.to_csv(
        metrics_dir / "best_epoch_by_seed.csv",
        index=False,
        float_format="%.10f",
    )

    summary = build_summary(best_summary)
    (metrics_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paper_table = build_paper_table(best_summary)
    paper_table.to_csv(table_dir / PAPER_TABLE_NAME, index=False)

    output_png = figure_dir / f"{FIGURE_STEM}.png"
    output_pdf = figure_dir / f"{FIGURE_STEM}.pdf"
    plot_training_curves(
        histories,
        aggregate,
        best_summary,
        output_png,
        output_pdf,
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
        "inference_performed": False,
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
    manifest = load_freeze_manifest()
    histories, input_hashes = load_histories(manifest)
    aggregate = aggregate_histories(histories)
    best_summary = build_best_epoch_summary(histories)

    validation_summary = manifest["validation_summary"]
    checks = {
        "best_val_accuracy": "selected_accuracy_mean",
        "best_val_macro_f1": "selected_macro_f1_mean",
        "best_val_balanced_accuracy": "selected_balanced_accuracy_mean",
    }
    for column, manifest_key in checks.items():
        actual_mean = float(best_summary[column].mean())
        expected_mean = float(validation_summary[manifest_key])
        if not np.isclose(actual_mean, expected_mean, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Aggregate validation result mismatch for {column}: "
                f"actual={actual_mean}, expected={expected_mean}"
            )

    return histories, aggregate, best_summary, input_hashes


def print_summary(best_summary: pd.DataFrame):
    print("Input validation: PASS")
    print("No training, checkpoint loading, or inference was performed.")
    for _, row in best_summary.iterrows():
        print(
            f"Seed {int(row['seed'])}: best epoch {int(row['best_epoch'])}, "
            f"validation Macro-F1 {row['best_val_macro_f1'] * 100:.4f}%, "
            f"training time {row['total_training_minutes']:.1f} min"
        )
    macro_mean, macro_std = mean_sample_std(
        best_summary["best_val_macro_f1"] * 100
    )
    print(
        "Best validation Macro-F1 across seeds: "
        f"{macro_mean:.4f}% +/- {macro_std:.4f}%"
    )


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.run:
        ensure_outputs_are_available(output_root)

    histories, aggregate, best_summary, input_hashes = prepare_analysis()
    print_summary(best_summary)

    if args.check_inputs:
        print("Preflight complete. No files were created.")
        return

    write_outputs(
        output_root,
        histories,
        aggregate,
        best_summary,
        input_hashes,
    )
    paths = output_paths(output_root)
    print(f"Metrics saved to: {paths['metrics_dir']}")
    print(f"Figure saved to: {paths['figure_dir'] / (FIGURE_STEM + '.png')}")
    print(f"Paper table saved to: {paths['table_dir'] / PAPER_TABLE_NAME}")


if __name__ == "__main__":
    main()
