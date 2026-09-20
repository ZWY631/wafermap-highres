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

# Table 6 (as numbered in the manuscript) order and labels. The two 64 x 64 rows are the frozen three-seed
# configurations: the re-trained control cells exist only for seed 42 (their
# role is to validate the new code path, reported in the text), so the table
# uses the frozen values as the reference corners of the 2 x 2 design.
TABLE_ORDER = (
    ("F64-S2P", "Standard stem, 64 × 64 (frozen)"),
    ("R128-S2P", "Standard stem, 128 × 128"),
    ("F64-S1N", "HighRes stem, 64 × 64 (frozen)"),
    ("R128-S1N", "HighRes stem, 128 × 128"),
)

# The 32 x 32 configurations of the stem ablation form the reference band for
# the "internal grid, not input pixels" branch of the argument. They are read
# from the frozen table rather than hardcoded.
STEM_TABLE = TABLES / "table_stem_five_config_multiseed_test.csv"
BAND_IDS = ("S2N", "S1P", "S1B")
FROZEN_HIGHRES_64 = 90.2029
FROZEN_STANDARD_64 = 80.4947


def band_32() -> dict[str, float]:
    frame = pd.read_csv(STEM_TABLE).set_index("configuration_id")
    return {key: float(frame.loc[key, "macro_f1_mean"]) * 100.0 for key in BAND_IDS}


def verdict(merged: pd.DataFrame) -> list[str]:
    """Report which branch of the argument the measured cells support."""
    rows = merged.set_index("cell_id")
    if "R128-S2P" not in rows.index or "F64-S1N" not in rows.index:
        return ["Not decidable yet: the 128 px standard cell has not been evaluated."]
    s128 = float(rows.loc["R128-S2P", "macro_f1_mean_percent"])
    gap_to_highres = FROZEN_HIGHRES_64 - s128
    band = band_32()
    band_lo, band_hi = min(band.values()), max(band.values())
    in_band = band_lo - 0.25 <= s128 <= band_hi + 0.25
    below_band = s128 < band_lo - 0.25

    lines = [
        f"standard@128 Macro-F1 = {s128:.4f}",
        f"  vs frozen HighRes@64 ({FROZEN_HIGHRES_64:.4f}): {s128 - FROZEN_HIGHRES_64:+.4f} pp",
        f"  vs frozen standard@64 ({FROZEN_STANDARD_64:.4f}): "
        f"{s128 - FROZEN_STANDARD_64:+.4f} pp",
        f"  vs the 32 x 32 band: {s128 - band_hi:+.4f} to {s128 - band_lo:+.4f} pp",
        "  frozen 32 x 32 band "
        + ", ".join(f"{key} {value:.4f}" for key, value in band.items()),
    ]
    if (in_band or below_band) and gap_to_highres > 0.5:
        if below_band:
            lines.append(
                "BRANCH A (strengthened) - internal grid dominates. The standard stem "
                "at 128 px does not even reach the 32 x 32 band: it lands below every "
                "configuration that attains the same pre-stage-2 grid by other means, "
                "while staying far below HighRes@64 at 3.94 times less compute. The "
                "samples have to survive the stem, not merely enter the network."
            )
        else:
            lines.append(
                "BRANCH A - internal grid dominates. Feeding the standard stem a finer "
                "input buys little: it lands in the 32 x 32 band and stays well below "
                "HighRes@64 despite costing only 1/3.94 of its compute. Keep the "
                "early-resolution claim and report the cost saving."
            )
    elif abs(gap_to_highres) <= 0.5:
        lines.append(
            "BRANCH B - input sampling dominates. The standard stem catches up with "
            "HighRes@64 once the input is fine enough, so the section must be "
            "rewritten around the input grid rather than the stem operator."
        )
    else:
        lines.append(
            "BRANCH C - intermediate. Interpret against the per-seed deltas below "
            "and against the 32 x 32 band before writing the section."
        )
    if "R128-S1N" in rows.index:
        h128 = float(rows.loc["R128-S1N", "macro_f1_mean_percent"])
        lines.append(
            f"HighRes@128 = {h128:.4f} ({h128 - FROZEN_HIGHRES_64:+.4f} pp vs "
            f"HighRes@64), i.e. the ceiling check."
        )
    return lines


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--cost", type=Path, default=COST)
    parser.add_argument("--per-seed", type=Path, default=PER_SEED)
    args = parser.parse_args()

    results_path: Path = args.results
    if not results_path.is_file():
        raise SystemExit(f"missing {results_path}; run the evaluator first")
    results = pd.read_csv(results_path)
    cost = pd.read_csv(args.cost)
    results["stem"] = results["display_name"].str.contains("identity").map(
        {True: "highres", False: "standard"}
    )
    merged = results.merge(
        cost[["image_size", "stem", "flops_million", "grid_before_stage2"]],
        on=["image_size", "stem"],
        how="left",
    )

    print("=" * 92)
    print("Table 6 (paste into the manuscript; cells in paper order)")
    print("=" * 92)
    print(
        "| Configuration | Input | Grid before stage 2 | FLOPs (M) | Accuracy (%) "
        "| Macro-F1 (%) | Scratch F1 (%) |"
    )
    print("|---|---:|---|---:|---:|---:|---:|")
    for cell_id, label in TABLE_ORDER:
        row = merged[merged["cell_id"] == cell_id]
        if row.empty:
            print(f"| {label} | — | — | — | not evaluated | — | — |")
            continue
        row = row.iloc[0]
        bold = "**" if cell_id == "R128-S2P" else ""
        print(
            f"| {label} | {row['image_size']} × {row['image_size']} | "
            f"{row['grid_before_stage2']} | {row['flops_million']:.2f} | "
            f"{bold}{row['accuracy_mean_percent']:.4f} ± "
            f"{row['accuracy_sample_std_percent']:.4f}{bold} | "
            f"{bold}{row['macro_f1_mean_percent']:.4f} ± "
            f"{row['macro_f1_sample_std_percent']:.4f}{bold} | "
            f"{row['scratch_f1_mean_percent']:.4f} ± "
            f"{row['scratch_f1_sample_std_percent']:.4f} |"
        )

    print()
    print("=" * 92)
    print("Verdict")
    print("=" * 92)
    for line in verdict(merged):
        print("  " + line)

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

    if Path(args.per_seed).is_file():
        per_seed = pd.read_csv(args.per_seed)
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
