#!/usr/bin/env python3
"""Turn the resolution-ablation results into the paper's Table 22 rows.

Merges the measured test metrics with the pre-computed complexity of the same
cells and prints everything the manuscript section needs: the table rows, the
paired per-seed deltas that carry the argument, and the cost ratios.

Usage
-----
    python 03_代码/scripts/analyze_resolution_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES = PROJECT_ROOT / "05_结果" / "tables"
RESULTS = TABLES / "table_v7_resolution_ablation_multiseed_test.csv"
COST = TABLES / "table_v7_resolution_ablation_cost.csv"
PER_SEED = (
    PROJECT_ROOT / "04_实验" / "metrics"
    / "20260914_resolution_ablation_multiseed_test" / "per_seed_metrics.csv"
)

# The three comparisons that carry the argument.
CONTRASTS = (
    ("R128-S2P", "F64-S1N", "standard stem at 128 px vs the frozen HighRes at 64 px"),
    ("R128-S2P", "F64-S2P", "standard stem at 128 px vs the standard stem at 64 px"),
    ("R128-S1N", "F64-S1N", "HighRes at 128 px vs the frozen HighRes at 64 px"),
    ("R128-S1N", "R128-S2P", "the stem effect at 128 px"),
)


def main() -> int:
    if not RESULTS.is_file():
        raise SystemExit(f"missing {RESULTS}; run the evaluator first")
    results = pd.read_csv(RESULTS)
    cost = pd.read_csv(COST)
    results["stem"] = results["display_name"].str.contains("identity").map(
        {True: "highres", False: "standard"}
    )
    merged = results.merge(
        cost[["image_size", "stem", "flops_million", "grid_before_stage2"]],
        on=["image_size", "stem"],
        how="left",
    )

    print("=" * 92)
    print("Table 22 rows (mean ± sample SD over the available seeds)")
    print("=" * 92)
    header = (
        "| Cell | Input | Grid before stage 2 | FLOPs (M) | Accuracy (%) | "
        "Macro-F1 (%) | Scratch F1 (%) |"
    )
    print(header)
    print("|---|---:|---|---:|---:|---:|---:|")
    order = ["F64-S2P", "R64-S2P", "R128-S2P", "F64-S1N", "R64-S1N", "R128-S1N"]
    for cell_id in order:
        row = merged[merged["cell_id"] == cell_id]
        if row.empty:
            print(f"| {cell_id} | — | — | — | not evaluated | — | — |")
            continue
        row = row.iloc[0]
        name = row["display_name"].replace(" x ", " × ")
        print(
            f"| {name} | {row['image_size']} | {row['grid_before_stage2']} | "
            f"{row['flops_million']:.2f} | "
            f"{row['accuracy_mean_percent']:.4f} ± {row['accuracy_sample_std_percent']:.4f} | "
            f"**{row['macro_f1_mean_percent']:.4f} ± {row['macro_f1_sample_std_percent']:.4f}** | "
            f"{row['scratch_f1_mean_percent']:.4f} ± {row['scratch_f1_sample_std_percent']:.4f} |"
        )

    print()
    print("=" * 92)
    print("Contrasts")
    print("=" * 92)
    for left, right, label in CONTRASTS:
        a = merged[merged["cell_id"] == left]
        b = merged[merged["cell_id"] == right]
        if a.empty or b.empty:
            print(f"\n{label}: incomplete ({left} or {right} not evaluated)")
            continue
        a, b = a.iloc[0], b.iloc[0]
        d_macro = a["macro_f1_mean_percent"] - b["macro_f1_mean_percent"]
        d_scratch = a["scratch_f1_mean_percent"] - b["scratch_f1_mean_percent"]
        d_acc = a["accuracy_mean_percent"] - b["accuracy_mean_percent"]
        ratio = a["flops_million"] / b["flops_million"]
        print(f"\n{label}")
        print(
            f"  Macro-F1 {a['macro_f1_mean_percent']:.4f} vs {b['macro_f1_mean_percent']:.4f} "
            f"= {d_macro:+.4f} pp    Scratch {d_scratch:+.4f} pp    Accuracy {d_acc:+.4f} pp"
        )
        print(f"  compute ratio {ratio:.2f}x")

    if PER_SEED.is_file():
        per_seed = pd.read_csv(PER_SEED)
        print()
        print("=" * 92)
        print("Per-seed Macro-F1 (needed for any paired claim)")
        print("=" * 92)
        pivot = per_seed.pivot_table(
            index="seed", columns="cell_id", values="macro_f1"
        ).mul(100).round(4)
        print(pivot.to_string())
        for left, right, label in CONTRASTS:
            if left in pivot.columns and right in pivot.columns:
                delta = (pivot[left] - pivot[right]).round(4)
                print(f"\n{left} minus {right} per seed ({label}):")
                print(delta.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
