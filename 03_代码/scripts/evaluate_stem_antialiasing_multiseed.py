#!/usr/bin/env python3
"""Unified test-set evaluation for the five ShuffleNetV2 stem configurations.

Why this exists
---------------
The paper's Table 8 compares the standard stem (S2P) with the high-resolution
stem (S1N), but the two sets of test predictions were produced by *different*
evaluators and are not even column-compatible: the S2P file carries 20 columns
including per-class probabilities, while the S1N file carries only
``source_index,true_label_id,predicted_label_id``. A paired McNemar test
therefore cannot be run across them at the sample level, and S2N / S1P were
never evaluated on the test split at all.

This script closes both gaps by putting every stem configuration through one
evaluator, on one frozen split, with one checkpoint-selection rule.

The anti-aliasing question
--------------------------
``run_antialiasing_ablation_training.py`` produced the S2B and S1B runs. S2B is
resolution-matched to S2P (same 16 x 16 grid, same 1,262,397 parameters) and
aliasing-matched to S1N, so it isolates the *aliasing* contribution from the
*resolution* contribution. The decision rule was frozen before any test-set
evaluation in
``00_项目管理/20260913_混叠消融预注册判定规则.md``; this script only measures.

Protocol
--------
* Checkpoint rule: best validation Macro-F1 within 30 epochs (already fixed by
  the training runs; nothing is re-selected here).
* Batch size 256, ``num_workers=0``, ``model.eval()``, ``torch.inference_mode``,
  no transforms -- identical to the frozen final-test evaluator.
* Every configuration is evaluated once. No tuning of any kind follows.

Honest limitation, recorded in the outputs
------------------------------------------
This benchmark test split has been read before (two historical baseline models
and the frozen HighRes final test), so it is not a virgin holdout. The script
records that fact in its own manifest rather than implying otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES  # noqa: E402
from wafermap.dataset import WaferMapDataset  # noqa: E402
from wafermap.models import ShuffleNetV2Baseline  # noqa: E402
from wafermap.models_improved import ShuffleNetV2HighRes  # noqa: E402
from wafermap.stem_ablation_models import (  # noqa: E402
    ShuffleNetV2Stride1MaxPool,
    ShuffleNetV2Stride2NoPool,
)
from wafermap.stem_antialiasing_models import (  # noqa: E402
    ShuffleNetV2Stride1BlurPool,
    ShuffleNetV2Stride2BlurPool,
)

EVALUATION_ID = "20260914_stem_antialiasing_multiseed_test"
OUTPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / EVALUATION_ID
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TABLES_DIR = PROJECT_ROOT / "05_结果" / "tables"

BATCH_SIZE = 256
NUM_WORKERS = 0
EXPECTED_TEST_SAMPLES = 25943
EXPECTED_PARAMETERS = 1_262_397
SEEDS = (42, 123, 2026)
SCRATCH_INDEX = WM811K_CLASS_NAMES.index("Scratch")


@dataclass(frozen=True)
class StemConfiguration:
    """One cell of the stem design space."""

    configuration_id: str
    display_name: str
    conv1_stride: int
    downsampling: str
    base_run_name: str
    model_factory: type
    stem_grid: str
    is_new: bool


CONFIGURATIONS: tuple[StemConfiguration, ...] = (
    StemConfiguration(
        "S2P", "stride 2 + max pool", 2, "maxpool",
        "shufflenet_v2_standard_ce_full",
        ShuffleNetV2Baseline, "16x16", False,
    ),
    StemConfiguration(
        "S2B", "stride 2 + blur pool", 2, "blurpool",
        "shufflenet_v2_stem_s2_blurpool_ce_full",
        ShuffleNetV2Stride2BlurPool, "16x16", True,
    ),
    StemConfiguration(
        "S2N", "stride 2 + no pool", 2, "identity",
        "shufflenet_v2_stem_s2_nopool_ce_full",
        ShuffleNetV2Stride2NoPool, "32x32", False,
    ),
    StemConfiguration(
        "S1P", "stride 1 + max pool", 1, "maxpool",
        "shufflenet_v2_stem_s1_pool_ce_full",
        ShuffleNetV2Stride1MaxPool, "32x32", False,
    ),
    StemConfiguration(
        "S1B", "stride 1 + blur pool", 1, "blurpool",
        "shufflenet_v2_stem_s1_blurpool_ce_full",
        ShuffleNetV2Stride1BlurPool, "32x32", True,
    ),
    StemConfiguration(
        "S1N", "stride 1 + no pool", 1, "identity",
        "shufflenet_v2_highres_ce_full",
        ShuffleNetV2HighRes, "64x64", False,
    ),
)

# Pre-registered decision rule (frozen 2026-09-13, before any test evaluation).
#   R = (F_S2B - F_S2P) / (F_S1N - F_S2P)
PREREGISTRATION = (
    "00_项目管理/20260913_混叠消融预注册判定规则.md"
)
DECISION_BANDS = (
    (0.25, "resolution_dominated", "分辨率主导：保留现有 early-resolution 结论"),
    (0.75, "mixed", "两者混合：必须改写为 resolution + aliasing 共同作用"),
    (float("inf"), "aliasing_dominated", "混叠主导：主结论必须改写为 aliasing"),
)


def run_name_for_seed(base_run_name: str, seed: int) -> str:
    return base_run_name if seed == 42 else f"{base_run_name}_seed{seed}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true",
                      help="校验全部输入，不做推理。")
    mode.add_argument("--execute", action="store_true",
                      help="执行测试集推理（只允许一次）。")
    mode.add_argument("--analyze-only", action="store_true",
                      help="复用已有逐样本预测重新出表，不跑推理。")
    return parser.parse_args(argv)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def checkpoint_path(configuration: StemConfiguration, seed: int) -> Path:
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    return (
        PROJECT_ROOT / "04_实验" / "checkpoints" / run_name / "best.pt"
    )


def history_path(configuration: StemConfiguration, seed: int) -> Path:
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    return PROJECT_ROOT / "04_实验" / "metrics" / f"{run_name}_history.csv"


def best_validation_epoch(configuration: StemConfiguration, seed: int) -> dict:
    """Read the frozen checkpoint-selection rule: best validation Macro-F1."""
    path = history_path(configuration, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing training history: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise ValueError(f"History is not a 30-epoch run: {path}")
    best = max(rows, key=lambda row: float(row["val_macro_f1"]))
    return {
        "epoch": int(best["epoch"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "val_accuracy": float(best["val_accuracy"]),
        "history_path": str(path.relative_to(PROJECT_ROOT)),
        "history_sha256": sha256_file(path),
    }


def validate_dataset() -> WaferMapDataset:
    dataset = WaferMapDataset("test")
    if dataset.transform is not None:
        raise ValueError("Test dataset must not apply random transforms.")
    if len(dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, got {len(dataset)}."
        )
    return dataset


def validate_split_isolation() -> dict:
    """Re-assert the lot-disjoint property of the frozen split."""
    metadata_file = (
        PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64"
        / "metadata.csv"
    )
    frame = pd.read_csv(metadata_file)
    lots = {
        split: set(frame.loc[frame["split"] == split, "lotName"])
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if not lots[left].isdisjoint(lots[right]):
            raise ValueError(f"{left} and {right} lots overlap.")
    return {split: len(values) for split, values in lots.items()}


def load_checkpoint(
    configuration: StemConfiguration,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    path = checkpoint_path(configuration, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    expected_epoch = best_validation_epoch(configuration, seed)
    if int(checkpoint["epoch"]) != expected_epoch["epoch"]:
        raise ValueError(
            f"Checkpoint epoch {checkpoint['epoch']} does not match the best "
            f"validation epoch {expected_epoch['epoch']}: {path}"
        )
    if not np.isclose(
        float(checkpoint["best_val_macro_f1"]),
        expected_epoch["val_macro_f1"],
        rtol=0.0, atol=1e-12,
    ):
        raise ValueError(f"Checkpoint validation score mismatch: {path}")

    model = configuration.model_factory(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(
            f"{configuration.configuration_id}: {parameters} parameters, "
            f"expected {EXPECTED_PARAMETERS}."
        )

    model = model.to(device).eval()
    return model, {
        "checkpoint_path": str(path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "epoch": int(checkpoint["epoch"]),
        "val_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "parameter_count": int(parameters),
        "history_path": expected_epoch["history_path"],
        "history_sha256": expected_epoch["history_sha256"],
    }


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    dataset: WaferMapDataset,
    device: torch.device,
) -> dict:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, drop_last=False,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    labels_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []

    synchronize(device)
    start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.shape[0]
        probability_chunks.append(
            torch.softmax(logits, dim=1).cpu().numpy()
        )
        labels_chunks.append(labels.cpu().numpy())

    synchronize(device)
    elapsed = time.perf_counter() - start

    y_true = np.concatenate(labels_chunks).astype(np.int64, copy=False)
    probabilities = np.concatenate(probability_chunks).astype(
        np.float32, copy=False
    )

    if not np.isfinite(total_loss):
        raise FloatingPointError("Test loss became NaN or infinite.")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Probabilities contain NaN or Inf.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise FloatingPointError("Probabilities do not sum to one.")
    if not np.array_equal(y_true, dataset.labels):
        raise RuntimeError("DataLoader label order changed unexpectedly.")

    return {
        "test_loss": total_loss / len(y_true),
        "y_true": y_true,
        "y_pred": probabilities.argmax(axis=1).astype(np.int64, copy=False),
        "probabilities": probabilities,
        "elapsed_seconds": elapsed,
    }


def metrics_for(results: dict) -> tuple[dict, list[dict]]:
    labels = np.arange(NUM_CLASSES)
    y_true, y_pred = results["y_true"], results["y_pred"]

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    summary = {
        "test_samples": int(len(y_true)),
        "test_loss": float(results["test_loss"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted",
                     zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "elapsed_seconds": float(results["elapsed_seconds"]),
        "samples_per_second": float(
            len(y_true) / results["elapsed_seconds"]
        ),
    }

    per_class = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1_score": float(class_f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id, class_name in enumerate(WM811K_CLASS_NAMES)
    ]
    return summary, per_class


def ensure_execute_allowed() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Evaluation bundle already exists: {OUTPUT_DIR}\n"
            "Refusing to re-run. Use --analyze-only to rebuild tables from "
            "the existing per-sample predictions."
        )


def save_predictions(
    path: Path,
    dataset: WaferMapDataset,
    configuration: StemConfiguration,
    seed: int,
    results: dict,
) -> None:
    """Persist one flat CSV per (configuration, seed) for paired analysis."""
    frame = dataset.metadata.copy()
    frame.insert(0, "configuration_id", configuration.configuration_id)
    frame.insert(1, "seed", seed)
    frame["true_label_id"] = results["y_true"]
    frame["predicted_label_id"] = results["y_pred"]
    frame["correct"] = results["y_true"] == results["y_pred"]
    frame["confidence"] = results["probabilities"].max(axis=1)

    if not np.array_equal(
        frame["label_id"].to_numpy(), results["y_true"]
    ):
        raise RuntimeError(
            "Metadata label_id does not match the dataset's integer labels; "
            "paired alignment across configurations cannot be guaranteed."
        )
    frame.to_csv(path, index=False)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
        "samples_per_second",
    )
    rows = []
    for (configuration_id, display_name), group in per_seed.groupby(
        ["configuration_id", "display_name"], sort=False
    ):
        row = {
            "configuration_id": configuration_id,
            "display_name": display_name,
            "seeds": len(group),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (configuration_id, class_id, class_name), group in per_class.groupby(
        ["configuration_id", "class_id", "class_name"], sort=False
    ):
        row = {
            "configuration_id": configuration_id,
            "class_id": int(class_id),
            "class_name": class_name,
            "support_per_seed": int(group["support"].iloc[0]),
        }
        for metric in ("precision", "recall", "f1_score"):
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_significance(
    predictions: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Exact McNemar of every configuration against S2P and S1N, per seed."""
    from scipy.stats import binomtest

    rows = []
    for seed in SEEDS:
        correct = {
            configuration.configuration_id: (
                predictions[(configuration.configuration_id, seed)]["correct"]
                .to_numpy(dtype=bool)
            )
            for configuration in CONFIGURATIONS
        }
        for reference_id in ("S2P", "S1N"):
            reference = correct[reference_id]
            for configuration in CONFIGURATIONS:
                if configuration.configuration_id == reference_id:
                    continue
                other = correct[configuration.configuration_id]
                if len(other) != len(reference):
                    raise RuntimeError("Prediction row counts differ.")

                other_only_right = int((other & ~reference).sum())
                reference_only_right = int((~other & reference).sum())
                discordant = other_only_right + reference_only_right
                p_value = (
                    float(binomtest(
                        other_only_right, discordant, 0.5, alternative="two-sided"
                    ).pvalue)
                    if discordant
                    else 1.0
                )
                rows.append({
                    "seed": seed,
                    "configuration_id": configuration.configuration_id,
                    "reference_id": reference_id,
                    "only_configuration_correct": other_only_right,
                    "only_reference_correct": reference_only_right,
                    "discordant": discordant,
                    "p_value": p_value,
                })

    frame = pd.DataFrame(rows)
    frame["p_value_holm"] = holm_adjust(frame["p_value"].tolist())
    frame["significant_at_0.05_holm"] = frame["p_value_holm"] < 0.05
    return frame


def factor_decomposition(
    per_seed: pd.DataFrame, metric: str = "macro_f1"
) -> pd.DataFrame:
    """Standard 2x2 factorial decomposition plus the blur factor."""
    pivot = per_seed.pivot_table(
        index="seed", columns="configuration_id", values=metric
    )
    for column in ("S2P", "S2B", "S2N", "S1P", "S1B", "S1N"):
        if column not in pivot.columns:
            raise ValueError(f"Missing configuration in results: {column}")

    rows = []
    for seed, row in pivot.iterrows():
        stride = 0.5 * ((row["S1P"] - row["S2P"]) + (row["S1N"] - row["S2N"]))
        pooling = 0.5 * ((row["S2N"] - row["S2P"]) + (row["S1N"] - row["S1P"]))
        interaction = (row["S1N"] - row["S2N"]) - (row["S1P"] - row["S2P"])
        blur = 0.5 * ((row["S2B"] - row["S2P"]) + (row["S1B"] - row["S1P"]))
        rows.append({
            "seed": int(seed),
            "stride_main_effect": float(stride),
            "pooling_main_effect": float(pooling),
            "interaction": float(interaction),
            "blur_factor_effect": float(blur),
            "aliasing_share_of_pooling": float(blur / pooling),
            # Text-book 2x2 identity: stride + pooling + interaction must
            # reproduce the S2P -> S1N total change exactly.
            "identity_residual_vs_S2P_to_S1N": float(
                stride + pooling + interaction - (row["S1N"] - row["S2P"])
            ),
        })
    return pd.DataFrame(rows)


def recovery_ratio(values: pd.Series, numerator: str) -> dict:
    """R = (F_num - F_S2P) / (F_den - F_S2P), the pre-registered statistic."""
    denominator = float(values["S1N"] - values["S2P"])
    if abs(denominator) < 1e-12:
        raise ValueError("Degenerate denominator in the recovery ratio.")
    return {
        "configuration_id": numerator,
        "value": float(values[numerator]),
        "S2P": float(values["S2P"]),
        "S1N": float(values["S1N"]),
        "recovery_ratio_R": float(
            (values[numerator] - values["S2P"]) / denominator
        ),
    }


def classify_ratio(ratio: float) -> tuple[str, str]:
    for upper, label, description in DECISION_BANDS:
        if ratio < upper:
            return label, description
    raise AssertionError("unreachable")


def scratch_decision(
    per_class: pd.DataFrame,
    per_class_seed: pd.DataFrame,
) -> dict:
    """Apply the pre-registered rule to the Scratch class."""
    means = (
        per_class[per_class["class_id"] == SCRATCH_INDEX]
        .set_index("configuration_id")["f1_score_mean"]
    )
    ratio = recovery_ratio(means, "S2B")

    by_seed = (
        per_class_seed[per_class_seed["class_id"] == SCRATCH_INDEX]
        .pivot_table(index="seed", columns="configuration_id",
                     values="f1_score")
    )
    seed_ratios = {
        str(int(seed)): recovery_ratio(row, "S2B")["recovery_ratio_R"]
        for seed, row in by_seed.iterrows()
    }

    label, description = classify_ratio(ratio["recovery_ratio_R"])

    # Pre-registered hard triggers that force the aliasing explanation.
    scratch_upper_bound = float(
        by_seed["S1N"].max()
    )
    s1n_min = float(by_seed["S1N"].min())
    hard_triggers = []
    if ratio["value"] >= s1n_min:
        hard_triggers.append(
            f"S2B Scratch F1 {ratio['value']:.4f} falls inside the S1N "
            f"three-seed range (min {s1n_min:.4f})"
        )

    return {
        "primary_metric": "test-set Scratch per-class F1 (3-seed mean)",
        "S2B_scratch_f1_mean": ratio["value"],
        "S2P_scratch_f1_mean": ratio["S2P"],
        "S1N_scratch_f1_mean": ratio["S1N"],
        "S1N_scratch_f1_seed_max": scratch_upper_bound,
        "recovery_ratio_R": ratio["recovery_ratio_R"],
        "recovery_ratio_R_per_seed": seed_ratios,
        "decision_band": label,
        "decision_text": description,
        "hard_triggers_fired": hard_triggers,
        "paper_must_be_rewritten": label in {"mixed", "aliasing_dominated"}
        or bool(hard_triggers),
    }


def write_tables(
    per_seed: pd.DataFrame,
    per_class: pd.DataFrame,
    per_class_seed: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    significance: pd.DataFrame,
    decomposition: pd.DataFrame,
    decision: dict,
    checkpoint_records: list[dict],
    device: torch.device,
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_frame.round(6).to_csv(
        TABLES_DIR / "table_stem_five_config_multiseed_test.csv", index=False
    )

    scratch = per_class[per_class["class_id"] == SCRATCH_INDEX]
    pivot_headline = per_class.pivot_table(
        index=["class_id", "class_name"],
        columns="configuration_id",
        values="f1_score_mean",
    ).reset_index()
    for reference in ("S2P", "S1N"):
        for target in ("S2B", "S1B", "S2N", "S1P"):
            if target in pivot_headline.columns and reference in pivot_headline.columns:
                pivot_headline[f"{target}_minus_{reference}_pp"] = (
                    (pivot_headline[target] - pivot_headline[reference]) * 100
                )
    pivot_headline.round(4).to_csv(
        TABLES_DIR / "table_stem_per_class_f1_five_config.csv", index=False
    )

    significance.round(8).to_csv(
        TABLES_DIR / "table_stem_paired_mcnemar_significance.csv", index=False
    )
    decomposition.round(6).to_csv(
        TABLES_DIR / "table_stem_factor_decomposition.csv", index=False
    )

    summary = {
        "evaluation_id": EVALUATION_ID,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": str(device),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "test_samples_per_seed": EXPECTED_TEST_SAMPLES,
        "parameter_count": EXPECTED_PARAMETERS,
        "seeds": list(SEEDS),
        "preregistration_document": PREREGISTRATION,
        "checkpoint_selection_rule": "best validation Macro-F1 within 30 epochs",
        "test_split_was_untouched_during_entire_project": False,
        "test_split_prior_access": (
            "Historical access by two baseline models and the frozen HighRes "
            "final test; recorded rather than denied."
        ),
        "post_test_tuning_allowed": False,
        "checkpoints": checkpoint_records,
        "decision": decision,
        "macro_f1_by_configuration": {
            row["configuration_id"]: {
                "mean": float(row["macro_f1_mean"]),
                "sample_std": float(row["macro_f1_sample_std"]),
            }
            for _, row in aggregate_frame.iterrows()
        },
        "scratch_f1_by_configuration": {
            row["configuration_id"]: {
                "mean": float(row["f1_score_mean"]),
                "sample_std": float(row["f1_score_sample_std"]),
            }
            for _, row in scratch.iterrows()
        },
    }
    (OUTPUT_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_decision(decision: dict) -> None:
    print("\n" + "=" * 78)
    print("预注册判定（主判定量：测试集 Scratch 逐类 F1，3 seed 均值）")
    print("=" * 78)
    print(f"  S2P  = {decision['S2P_scratch_f1_mean']:6.2f}")
    print(f"  S2B  = {decision['S2B_scratch_f1_mean']:6.2f}   ← blur-pool")
    print(f"  S1N  = {decision['S1N_scratch_f1_mean']:6.2f}")
    print(f"  回收率 R = {decision['recovery_ratio_R']:.3f}")
    print(f"  逐 seed R = {decision['recovery_ratio_R_per_seed']}")
    print(f"  → 判定区间：{decision['decision_band']}")
    print(f"  → {decision['decision_text']}")
    if decision["hard_triggers_fired"]:
        for trigger in decision["hard_triggers_fired"]:
            print(f"  ⚠️  硬条件触发：{trigger}")
    print(
        f"  → 论文是否需要改写："
        f"{'是' if decision['paper_must_be_rewritten'] else '否'}"
    )
    print("=" * 78)


def analyze(
    per_seed: pd.DataFrame,
    per_class_seed: pd.DataFrame,
    predictions: dict[tuple[str, int], pd.DataFrame],
    checkpoint_records: list[dict],
    device: torch.device,
) -> dict:
    per_class = aggregate_per_class(per_class_seed)
    aggregate_frame = aggregate(per_seed)
    significance = paired_significance(predictions)
    decomposition = factor_decomposition(per_seed)
    decision = scratch_decision(per_class, per_class_seed)

    write_tables(
        per_seed, per_class, per_class_seed, aggregate_frame,
        significance, decomposition, decision, checkpoint_records, device,
    )
    print_decision(decision)
    print(f"\n输出目录：{OUTPUT_DIR}")
    print(f"论文表格：{TABLES_DIR}")
    return decision


def load_existing_predictions() -> tuple[
    dict[tuple[str, int], pd.DataFrame], pd.DataFrame, pd.DataFrame
]:
    predictions = {}
    per_seed_rows = []
    per_class_rows = []

    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            path = PREDICTIONS_DIR / (
                f"predictions_{configuration.configuration_id}_seed{seed}.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing predictions: {path}")
            frame = pd.read_csv(path)
            predictions[(configuration.configuration_id, seed)] = frame

            y_true = frame["true_label_id"].to_numpy()
            y_pred = frame["predicted_label_id"].to_numpy()
            labels = np.arange(NUM_CLASSES)
            precision, recall, class_f1, support = (
                precision_recall_fscore_support(
                    y_true, y_pred, labels=labels, average=None, zero_division=0
                )
            )
            _, _, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
            per_seed_rows.append({
                "configuration_id": configuration.configuration_id,
                "display_name": configuration.display_name,
                "stem_grid": configuration.stem_grid,
                "seed": seed,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(
                    f1_score(y_true, y_pred, labels=labels,
                             average="weighted", zero_division=0)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, y_pred)
                ),
            })
            for class_id, class_name in enumerate(WM811K_CLASS_NAMES):
                per_class_rows.append({
                    "configuration_id": configuration.configuration_id,
                    "seed": seed,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": float(precision[class_id]),
                    "recall": float(recall[class_id]),
                    "f1_score": float(class_f1[class_id]),
                    "support": int(support[class_id]),
                })

    return predictions, pd.DataFrame(per_seed_rows), pd.DataFrame(per_class_rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device()

    print(f"evaluation_id={EVALUATION_ID}")
    print(f"device={device}")

    if args.analyze_only:
        if not OUTPUT_DIR.is_dir():
            raise FileNotFoundError(f"No evaluation bundle: {OUTPUT_DIR}")
        predictions, per_seed, per_class_seed = load_existing_predictions()
        summary = json.loads(
            (OUTPUT_DIR / "metrics_summary.json").read_text(encoding="utf-8")
        )
        analyze(
            per_seed, per_class_seed, predictions,
            summary.get("checkpoints", []), device,
        )
        return 0

    dataset = validate_dataset()
    lot_counts = validate_split_isolation()
    print(
        f"test_samples={len(dataset)} "
        f"class_counts={np.bincount(dataset.labels, minlength=NUM_CLASSES).tolist()}"
    )
    print(f"lot_isolation={lot_counts}")

    # Validate every checkpoint before touching the test split.
    plan = []
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            _, info = load_checkpoint(configuration, seed, torch.device("cpu"))
            plan.append((configuration, seed, info))
            print(
                f"configuration={configuration.configuration_id} seed={seed} "
                f"grid={configuration.stem_grid} epoch={info['epoch']} "
                f"val_macro_f1={info['val_macro_f1']:.4f} "
                f"sha256={info['checkpoint_sha256'][:12]}"
            )
    print(f"validated_checkpoints={len(plan)}")

    if args.preflight:
        print("预检通过：没有执行任何测试集推理。")
        return 0

    ensure_execute_allowed()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=False)

    checkpoint_records = []
    per_seed_rows = []
    per_class_rows = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}

    for index, (configuration, seed, info) in enumerate(plan, start=1):
        print(
            f"\n[{index}/{len(plan)}] "
            f"{configuration.configuration_id} seed={seed} "
            f"({configuration.display_name})",
            flush=True,
        )
        model, info = load_checkpoint(configuration, seed, device)
        results = collect_predictions(model, dataset, device)
        summary, per_class = metrics_for(results)

        record = dict(info)
        record.update({
            "configuration_id": configuration.configuration_id,
            "display_name": configuration.display_name,
            "conv1_stride": configuration.conv1_stride,
            "downsampling": configuration.downsampling,
            "stem_grid": configuration.stem_grid,
            "seed": seed,
        })
        checkpoint_records.append(record)

        row = {
            "configuration_id": configuration.configuration_id,
            "display_name": configuration.display_name,
            "stem_grid": configuration.stem_grid,
            "seed": seed,
            **summary,
        }
        per_seed_rows.append(row)
        for entry in per_class:
            per_class_rows.append({
                "configuration_id": configuration.configuration_id,
                "seed": seed,
                **entry,
            })

        save_predictions(
            PREDICTIONS_DIR / (
                f"predictions_{configuration.configuration_id}_seed{seed}.csv"
            ),
            dataset, configuration, seed, results,
        )
        predictions[(configuration.configuration_id, seed)] = pd.DataFrame({
            "true_label_id": results["y_true"],
            "predicted_label_id": results["y_pred"],
            "correct": results["y_true"] == results["y_pred"],
        })

        print(
            f"    accuracy={summary['accuracy']:.4f} "
            f"macro_f1={summary['macro_f1']:.4f} "
            f"scratch_f1="
            f"{next(e['f1_score'] for e in per_class if e['class_id'] == SCRATCH_INDEX):.4f} "
            f"({summary['samples_per_second']:.0f} samples/s)",
            flush=True,
        )

        del model, results
        if device.type == "mps":
            torch.mps.empty_cache()

    per_seed = pd.DataFrame(per_seed_rows)
    per_class_seed = pd.DataFrame(per_class_rows)
    per_seed.to_csv(OUTPUT_DIR / "test_per_seed_summary.csv", index=False)
    per_class_seed.to_csv(
        OUTPUT_DIR / "test_per_class_by_seed.csv", index=False
    )

    analyze(
        per_seed, per_class_seed, predictions, checkpoint_records, device
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
