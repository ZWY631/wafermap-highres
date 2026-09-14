#!/usr/bin/env python3
"""Synthesise the four v7 manuscript tables from frozen artifacts.

This script only *reads* frozen evidence and writes new ``table_v7_*.csv``
files. No frozen artifact is modified.

Sources
-------
``05_结果/tables/table_stem_five_config_multiseed_test.csv``
    Six-configuration stem comparison on the fixed test split.
``05_结果/tables/table_stem_per_class_f1_five_config.csv``
    Per-class F1 for the same six configurations.
``05_结果/tables/table_stem_factor_decomposition.csv``
    Per-seed 2x2 factor decomposition on the test split.
``05_结果/tables/table_five_model_multiseed_per_class.csv``
    Per-class F1 for all five generic backbones.
``05_结果/tables/table_highres_vs_standard_per_class_f1.csv``
    Parameter-matched standard-vs-HighRes classwise gains.
``04_实验/metrics/20260805_stem_2x2_validation_selection/aggregate_validation.csv``
    Validation Macro-F1 for S2P/S2N/S1P/S1N.
``04_实验/metrics/*blurpool*_history.csv``
    Validation histories for S2B/S1B (best epoch = argmax validation Macro-F1,
    which is the checkpoint-selection rule of the frozen protocol).

Outputs (new files only)
------------------------
``table_v7_stem_cost_benefit.csv``
``table_v7_five_model_per_class_selected.csv``
``table_v7_headline_gain_attribution.csv``
``table_v7_factor_decomposition_summary.csv``

Usage
-----
    python 03_代码/scripts/synthesize_v7_tables.py            # write
    python 03_代码/scripts/synthesize_v7_tables.py --check    # verify only
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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

TABLES_DIR = PROJECT_ROOT / "05_结果" / "tables"
METRICS_DIR = PROJECT_ROOT / "04_实验" / "metrics"

INPUT_SHAPE = (1, 1, 64, 64)
EXPECTED_PARAMETERS = 1_262_397
# FLOP counts already published for the four original configurations; the two
# blur-pool configurations are counted with the identical convention and are
# cross-checked against the frozen aggregate_validation.csv column.
EXPECTED_FLOPS = {
    "S2P": 22_624_960,
    "S2N": 89_117_440,
    "S1P": 90_444_544,
    "S1N": 356_414_464,
}

CONFIG_ORDER = ["S2P", "S2B", "S2N", "S1P", "S1B", "S1N"]
FACTORY = {
    "S2P": ShuffleNetV2Baseline,
    "S2N": ShuffleNetV2Stride2NoPool,
    "S1P": ShuffleNetV2Stride1MaxPool,
    "S2B": ShuffleNetV2Stride2BlurPool,
    "S1B": ShuffleNetV2Stride1BlurPool,
    "S1N": ShuffleNetV2HighRes,
}
DISPLAY = {
    "S2P": "stride 2 + max pool",
    "S2B": "stride 2 + blur pool",
    "S2N": "stride 2 + identity",
    "S1P": "stride 1 + max pool",
    "S1B": "stride 1 + blur pool",
    "S1N": "stride 1 + identity (HighRes)",
}
GRID = {"S2P": "16x16", "S2B": "16x16", "S2N": "32x32", "S1P": "32x32", "S1B": "32x32", "S1N": "64x64"}

SELECTED_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


def count_flops(model: nn.Module, label: str) -> int:
    """Conv/Linear FLOPs for one 1x1x64x64 sample, mirroring the frozen benchmark."""
    model.eval()
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            output = model(dummy)
    if tuple(output.shape) != (1, 9):
        raise ValueError(f"unexpected output shape for {label}: {tuple(output.shape)}")
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(f"parameter mismatch for {label}: {parameters}")
    return int(counter.get_total_flops())


def measure_complexity() -> pd.DataFrame:
    records = []
    for config in CONFIG_ORDER:
        model = FACTORY[config](num_classes=9)
        flops = count_flops(model, config)
        expected = EXPECTED_FLOPS.get(config)
        if expected is not None and flops != expected:
            raise ValueError(f"FLOP count mismatch for {config}: {flops} != {expected}")
        records.append(
            {
                "configuration_id": config,
                "display_name": DISPLAY[config],
                "stem_grid": GRID[config],
                "parameter_count": EXPECTED_PARAMETERS,
                "conv_linear_flops": flops,
            }
        )
    return pd.DataFrame(records)


def build_stem_cost_benefit(complexity: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_csv(TABLES_DIR / "table_stem_five_config_multiseed_test.csv")
    per_class = pd.read_csv(TABLES_DIR / "table_stem_per_class_f1_five_config.csv")

    scratch = per_class.loc[
        per_class["class_name"] == "Scratch",
        ["S2P", "S2B", "S2N", "S1P", "S1B", "S1N"],
    ].iloc[0]

    merged = test.merge(
        complexity[["configuration_id", "stem_grid", "conv_linear_flops"]],
        on="configuration_id",
        how="inner",
    )
    merged["parameter_count"] = merged["configuration_id"].map(
        dict(zip(complexity["configuration_id"], complexity["parameter_count"]))
    )
    if len(merged) != len(CONFIG_ORDER):
        raise ValueError(f"expected {len(CONFIG_ORDER)} configurations, got {len(merged)}")

    merged["scratch_f1_mean"] = merged["configuration_id"].map(scratch)
    baseline_flops = int(
        merged.loc[merged["configuration_id"] == "S2P", "conv_linear_flops"].iloc[0]
    )
    merged["flops_ratio_vs_s2p"] = merged["conv_linear_flops"] / baseline_flops
    merged["macro_f1_gain_vs_s2p_pp"] = (
        merged["macro_f1_mean"] - merged.loc[merged["configuration_id"] == "S2P", "macro_f1_mean"].iloc[0]
    ) * 100.0
    merged = merged.sort_values("conv_linear_flops")
    columns = [
        "configuration_id",
        "display_name",
        "stem_grid",
        "parameter_count",
        "conv_linear_flops",
        "flops_ratio_vs_s2p",
        "accuracy_mean_percent",
        "macro_f1_mean_percent",
        "macro_f1_gain_vs_s2p_pp",
        "scratch_f1_percent",
        "evaluation_throughput_samples_per_second",
        "evaluation_throughput_sample_std",
    ]
    merged["accuracy_mean_percent"] = merged["accuracy_mean"] * 100.0
    merged["macro_f1_mean_percent"] = merged["macro_f1_mean"] * 100.0
    merged["scratch_f1_percent"] = merged["scratch_f1_mean"] * 100.0
    merged = merged.rename(
        columns={
            "samples_per_second_mean": "evaluation_throughput_samples_per_second",
            "samples_per_second_sample_std": "evaluation_throughput_sample_std",
        }
    )
    return merged[columns]


def build_five_model_per_class() -> pd.DataFrame:
    raw = pd.read_csv(TABLES_DIR / "table_five_model_multiseed_per_class.csv")
    grouped = (
        raw.groupby(["model", "class_name"])["f1_score"]
        .agg(["mean", "std"])
        .mul(100.0)
        .reset_index()
    )
    wide_mean = grouped.pivot(index="model", columns="class_name", values="mean")
    wide_std = grouped.pivot(index="model", columns="class_name", values="std")

    overall = pd.read_csv(TABLES_DIR / "table_five_model_multiseed_test.csv")
    overall = overall.set_index("model")

    model_order = [
        "Standard ShuffleNetV2",
        "MobileNetV3-Small",
        "EfficientNet-B0",
        "ResNet18",
        "HighRes ShuffleNetV2",
    ]
    records = []
    for model in model_order:
        row = {
            "model": model,
            "parameter_count": int(overall.loc[model, "parameter_count"]),
            "macro_f1_mean_percent": round(float(overall.loc[model, "macro_f1_mean"]) * 100, 4),
            "macro_f1_sample_std_percent": round(
                float(overall.loc[model, "macro_f1_sample_std"]) * 100, 4
            ),
        }
        for name in SELECTED_CLASSES:
            key = name if name != "none" else "none"
            row[f"{key}_f1_mean_percent"] = round(float(wide_mean.loc[model, name]), 4)
            row[f"{key}_f1_sample_std_percent"] = round(float(wide_std.loc[model, name]), 4)
        records.append(row)
    return pd.DataFrame(records)


def build_headline_attribution() -> pd.DataFrame:
    raw = pd.read_csv(TABLES_DIR / "table_highres_vs_standard_per_class_f1.csv")
    total_gain = float(raw["highres_gain_pp"].sum())
    frame = raw[
        ["class_name", "support_per_seed", "standard_shufflenet_v2_f1_mean_percent",
         "highres_shufflenet_v2_f1_mean_percent", "highres_gain_pp"]
    ].copy()
    frame = frame.rename(
        columns={
            "class_name": "class",
            "support_per_seed": "support_per_seed",
            "standard_shufflenet_v2_f1_mean_percent": "standard_stem_f1_percent",
            "highres_shufflenet_v2_f1_mean_percent": "highres_f1_percent",
            "highres_gain_pp": "f1_gain_pp",
        }
    )
    frame["macro_f1_contribution_pp"] = frame["f1_gain_pp"] / len(frame)
    frame["share_of_macro_f1_gain_percent"] = frame["f1_gain_pp"] / total_gain * 100.0
    frame["macro_f1_gain_excluding_class_pp"] = [
        (total_gain - gain) / (len(frame) - 1) for gain in frame["f1_gain_pp"]
    ]
    frame["share_of_test_maps_percent"] = (
        frame["support_per_seed"] / frame["support_per_seed"].sum() * 100.0
    )
    frame = frame.sort_values("f1_gain_pp", ascending=False)
    frame["total_macro_f1_gain_pp"] = total_gain
    return frame.round(4)


def validation_macro_f1() -> dict[str, float]:
    aggregate = pd.read_csv(
        METRICS_DIR / "20260805_stem_2x2_validation_selection" / "aggregate_validation.csv"
    )
    values = {
        row["configuration_id"]: float(row["validation_macro_f1_mean"])
        for _, row in aggregate.iterrows()
    }
    for path in sorted(glob.glob(str(METRICS_DIR / "*blurpool*_history.csv"))):
        frame = pd.read_csv(path)
        column = next(c for c in frame.columns if "val" in c.lower() and "macro" in c.lower())
        config = "S2B" if "s2_blurpool" in path else "S1B"
        seed = int(re.search(r"seed(\d+)", path).group(1)) if "seed" in path else 42
        values.setdefault(config, {})
        if not isinstance(values.get(config), dict):
            values[config] = {}
        values[config][seed] = float(frame.loc[frame[column].idxmax(), column])
    for config in ("S2B", "S1B"):
        per_seed = values[config]
        values[config] = sum(per_seed.values()) / len(per_seed)
    return values


def build_factor_decomposition_summary() -> pd.DataFrame:
    def components(s2p: float, s2n: float, s1p: float, s1n: float, s2b: float, s1b: float) -> dict:
        stride = 0.5 * ((s1p - s2p) + (s1n - s2n))
        pooling = 0.5 * ((s2n - s2p) + (s1n - s1p))
        interaction = (s1n - s1p) - (s2n - s2p)
        blur = 0.5 * ((s2b - s2p) + (s1b - s1p))
        return {
            "stride_main_effect_pp": stride * 100.0,
            "pooling_main_effect_pp": pooling * 100.0,
            "interaction_pp": interaction * 100.0,
            "blur_factor_effect_pp": blur * 100.0,
            "aliasing_share_of_pooling_percent": blur / pooling * 100.0,
            "aliasing_share_min_percent": float("nan"),
            "aliasing_share_max_percent": float("nan"),
        }

    validation = validation_macro_f1()
    validation_row = components(
        validation["S2P"],
        validation["S2N"],
        validation["S1P"],
        validation["S1N"],
        validation["S2B"],
        validation["S1B"],
    )
    validation_row.update(
        {
            "split": "validation",
            "seeds": 3,
            "source": "04_实验/metrics/20260805_stem_2x2_validation_selection/aggregate_validation.csv"
            " + *blurpool*_history.csv",
        }
    )

    # The test split decomposition is read from the frozen per-seed artifact
    # rather than re-derived, so the manuscript cites the frozen numbers.
    frozen = pd.read_csv(TABLES_DIR / "table_stem_factor_decomposition.csv")
    test_row = {
        "split": "test",
        "seeds": int(len(frozen)),
        "stride_main_effect_pp": float(frozen["stride_main_effect"].mean()) * 100.0,
        "pooling_main_effect_pp": float(frozen["pooling_main_effect"].mean()) * 100.0,
        "interaction_pp": float(frozen["interaction"].mean()) * 100.0,
        "blur_factor_effect_pp": float(frozen["blur_factor_effect"].mean()) * 100.0,
        "aliasing_share_of_pooling_percent": float(
            frozen["aliasing_share_of_pooling"].mean()
        )
        * 100.0,
        "aliasing_share_min_percent": float(frozen["aliasing_share_of_pooling"].min()) * 100.0,
        "aliasing_share_max_percent": float(frozen["aliasing_share_of_pooling"].max()) * 100.0,
        "source": "05_结果/tables/table_stem_factor_decomposition.csv (per-seed mean)",
    }

    frame = pd.DataFrame([validation_row, test_row])
    return frame.round(4)


def write(frame: pd.DataFrame, name: str, check_only: bool) -> Path:
    path = TABLES_DIR / name
    if check_only:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        existing = pd.read_csv(path)
        if existing.shape != frame.shape:
            raise ValueError(f"{name} shape drift: {existing.shape} != {frame.shape}")
        print(f"OK   {name} {frame.shape}")
    else:
        frame.to_csv(path, index=False)
        print(f"WROTE {name} {frame.shape}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing outputs only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    complexity = measure_complexity()
    print(complexity.to_string(index=False))
    print()

    cost = build_stem_cost_benefit(complexity)
    per_class = build_five_model_per_class()
    attribution = build_headline_attribution()
    factors = build_factor_decomposition_summary()

    write(cost, "table_v7_stem_cost_benefit.csv", args.check)
    write(per_class, "table_v7_five_model_per_class_selected.csv", args.check)
    write(attribution, "table_v7_headline_gain_attribution.csv", args.check)
    write(factors, "table_v7_factor_decomposition_summary.csv", args.check)

    print()
    print(attribution[["class", "f1_gain_pp", "share_of_macro_f1_gain_percent",
                       "macro_f1_gain_excluding_class_pp"]].to_string(index=False))
    print()
    print(factors.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
