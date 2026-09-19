#!/usr/bin/env python3
"""Diagnose how a raw WM-811K pickle lines up with the frozen split manifest.

The frozen split is keyed by ``source_index``, a positional index into the
pickle released on Kaggle. An alternative distribution of the same records (for
example the MIRLab release) may hold them in a different order, in which case
positional alignment fails and the records have to be matched on
``(lotName, waferIndex)`` instead.

This script is read-only: it loads the pickle, compares it with the manifest,
and reports a verdict. When positional alignment fails but key alignment
succeeds, it writes an alignment map that converts each manifest row into a raw
pickle row, so the preprocessing can be re-run without changing the frozen
split.

Usage
-----
    python 03_代码/scripts/check_raw_alignment.py
    python 03_代码/scripts/check_raw_alignment.py --raw 02_数据/raw/mirlab_2022/LSWMD.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
SCRIPTS_DIR = CODE_DIR / "scripts"
SRC_DIR = CODE_DIR / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inspect_raw_wm811k import load_legacy_pandas_pickle, normalize_nested_value  # noqa: E402
from wafermap.paths import SPLITS_DIR  # noqa: E402

SPLIT_FILE = SPLITS_DIR / "wm811k_labeled_lot_disjoint.csv"
ALIGNMENT_OUT = (
    PROJECT_ROOT / "02_数据" / "interim" / "raw_alignment_for_manifest.csv"
)
CHECK_COLUMNS = ("lotName", "waferIndex", "failureType")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "02_数据" / "raw" / "mirlab_2022" / "LSWMD.pkl",
    )
    parser.add_argument("--write-alignment", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.raw.is_file():
        raise SystemExit(f"raw pickle not found: {args.raw}")

    manifest = pd.read_csv(SPLIT_FILE)
    print(f"manifest rows      : {len(manifest)}")
    print(f"manifest columns   : {list(manifest.columns)}")

    raw = load_legacy_pandas_pickle(args.raw)
    print(f"raw pickle rows    : {len(raw)}")
    print(f"raw pickle columns : {list(raw.columns)}")

    missing = [c for c in CHECK_COLUMNS if c not in raw.columns]
    if missing:
        print(f"\nVERDICT: the pickle lacks {missing}; key alignment is impossible.")
        return 2

    indices = manifest["source_index"].to_numpy(dtype="int64")
    if indices.max() >= len(raw):
        print("\nVERDICT: source_index exceeds the pickle length; different release.")
        return 2

    selected = raw.iloc[indices]
    positional_ok = True
    for column in CHECK_COLUMNS:
        expected = manifest[
            "label" if column == "failureType" else column
        ].map(normalize_nested_value).to_numpy()
        observed = selected[column].map(normalize_nested_value).to_numpy()
        matches = int((expected == observed).sum())
        print(f"positional {column:12s}: {matches}/{len(manifest)} rows agree")
        positional_ok = positional_ok and matches == len(manifest)

    if positional_ok:
        print("\nVERDICT: POSITIONAL ALIGNMENT OK - the frozen source_index is valid")
        print("         for this release; the standard preprocessing path applies.")
        return 0

    print("\npositional alignment failed; trying (lotName, waferIndex) keys")
    raw_keys = pd.MultiIndex.from_arrays(
        [
            raw["lotName"].map(normalize_nested_value),
            raw["waferIndex"].map(normalize_nested_value),
        ]
    )
    if raw_keys.has_duplicates:
        print(f"         note: {int(raw_keys.duplicated().sum())} duplicate keys in the pickle")
    key_to_position = {}
    for position, key in enumerate(raw_keys):
        key_to_position.setdefault(key, position)

    positions = []
    unresolved = 0
    label_mismatch = 0
    for _, row in manifest.iterrows():
        key = (normalize_nested_value(row["lotName"]), normalize_nested_value(row["waferIndex"]))
        position = key_to_position.get(key)
        if position is None:
            unresolved += 1
            positions.append(-1)
            continue
        positions.append(position)
        if normalize_nested_value(raw.iloc[position]["failureType"]) != normalize_nested_value(
            row["label"]
        ):
            label_mismatch += 1

    resolved = len(manifest) - unresolved
    print(f"key alignment     : {resolved}/{len(manifest)} rows resolved")
    print(f"label mismatches  : {label_mismatch}")

    if unresolved == 0 and label_mismatch == 0:
        print("\nVERDICT: KEY ALIGNMENT OK - rerun preprocessing with this alignment map.")
        if args.write_alignment:
            ALIGNMENT_OUT.parent.mkdir(parents=True, exist_ok=True)
            out = manifest[["source_index", "lotName", "waferIndex", "label", "split"]].copy()
            out["raw_position"] = positions
            out.to_csv(ALIGNMENT_OUT, index=False)
            print(f"         wrote {ALIGNMENT_OUT}")
        else:
            print("         pass --write-alignment to emit the map")
        return 0

    print("\nVERDICT: NEITHER alignment works - the release does not contain the same records.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
