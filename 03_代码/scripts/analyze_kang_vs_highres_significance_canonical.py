#!/usr/bin/env python3
"""Run the corrected Kang/HighRes analysis against canonical HighRes files.

The published copy of HighRes seed42 contains a one-character label typo and
was edited after the original analysis.  This driver keeps the corrected
implementation unchanged while selecting the canonical copy under
``04_实验/metrics/20260729_highres_ce_multiseed_final_test`` and writing a
separate, non-overwriting evidence namespace.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).with_name(
    "analyze_kang_vs_highres_significance_corrected.py"
)
CANONICAL_HIGHRES_DIR = (
    PROJECT_ROOT / "04_实验/metrics/20260729_highres_ce_multiseed_final_test/predictions"
)
ANALYSIS_ID = "20260817_kang_vs_highres_significance_canonical"
OUTPUT_DIR = PROJECT_ROOT / f"04_实验/metrics/{ANALYSIS_ID}"
PAPER_TABLE = PROJECT_ROOT / (
    "05_结果/tables/table_kang_vs_highres_paired_significance_canonical.csv"
)


def load_implementation():
    spec = importlib.util.spec_from_file_location(
        "kang_highres_significance_corrected_canonical_driver", IMPLEMENTATION
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load implementation: {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HIGHRES_DIR = CANONICAL_HIGHRES_DIR
    module.ANALYSIS_ID = ANALYSIS_ID
    module.OUTPUT_DIR = OUTPUT_DIR
    module.PAPER_TABLE = PAPER_TABLE
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected paired tests using canonical HighRes predictions."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--replicates", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation = load_implementation()
    if args.check_inputs:
        helpers = implementation.load_helpers()
        observed, _ = implementation.validate_inputs(helpers)
        print("CANONICAL_INPUT_VALIDATION_OK")
        print(observed.to_string(index=False))
        return
    implementation.run(args.replicates)
    manifest_path = OUTPUT_DIR / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["driver_script"] = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    manifest["canonical_highres_source"] = str(
        CANONICAL_HIGHRES_DIR.relative_to(PROJECT_ROOT)
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
