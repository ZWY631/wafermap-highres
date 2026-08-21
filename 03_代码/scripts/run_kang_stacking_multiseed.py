#!/usr/bin/env python3
"""Preflight or execute the three frozen Kang stacking seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "03_代码/scripts/train_kang_stacking_seed.py"
SEEDS = (42, 123, 2026)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    return parser.parse_args(argv)


def planned_commands(python_executable=sys.executable, seeds=SEEDS, execute=False):
    mode = "--execute" if execute else "--preflight"
    return [
        [python_executable, str(TRAIN_SCRIPT), "--seed", str(seed), mode]
        for seed in seeds
    ]


def main(argv=None):
    args = parse_args(argv)
    commands = planned_commands(seeds=tuple(args.seeds), execute=args.execute)
    for command in commands:
        print("COMMAND", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.execute:
        print("ALL_REQUESTED_STACKING_SEEDS_COMPLETE")
    else:
        print("STACKING_MULTISEED_PREFLIGHT_COMPLETE; no training started")


if __name__ == "__main__":
    main()

