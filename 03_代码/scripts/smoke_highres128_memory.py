#!/usr/bin/env python3
"""Smoke-test whether HighRes at 128 px fits this machine's memory.

The frozen protocol trains at batch 128, which thrashes a 16 GB machine once the
high-resolution stem keeps 128 x 128 feature maps. This wrapper shrinks the
batch and caps the number of training batches per epoch so one epoch can be
timed cheaply, without touching the frozen training module.

Usage
-----
    python 03_代码/scripts/smoke_highres128_memory.py --batch-size 32 --batches 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
SRC_DIR = CODE_DIR / "src"
for path in (SRC_DIR, CODE_DIR / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import wafermap.controlled_ce_training as training  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--stem", default="highres")
    return parser.parse_args()


def swap_used_mb() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    for token in out.split():
        if token.endswith("M") and token[:-1].replace(".", "").isdigit():
            continue
    parts = out.split("used =")[-1].split("M")[0].strip()
    try:
        return float(parts)
    except ValueError:
        return float("nan")


def main() -> int:
    args = parse_args()
    training.TRAIN_BATCH_SIZE = args.batch_size
    training.EVAL_BATCH_SIZE = args.eval_batch_size
    training.MAX_TRAIN_BATCHES = args.batches

    print(f"train batch size : {training.TRAIN_BATCH_SIZE}")
    print(f"eval batch size  : {training.EVAL_BATCH_SIZE}")
    print(f"batches per epoch: {training.MAX_TRAIN_BATCHES} (capped for the test)")
    print(f"swap used before : {swap_used_mb():.0f} MB")

    sys.argv = [
        sys.argv[0],
        "--image-size", str(args.image_size),
        "--stem", args.stem,
        "--seed", "42",
    ]
    import train_resolution_ablation as ablation

    started = time.time()
    ablation.main()
    elapsed = time.time() - started
    print(f"\nelapsed {elapsed:.1f} s for the capped run")
    print(f"swap used after  : {swap_used_mb():.0f} MB")
    per_batch = elapsed / max(args.batches, 1)
    full_epoch = per_batch * (121069 / args.batch_size)
    print(f"~{per_batch:.2f} s per {args.batch_size}-sample batch")
    print(f"projected full epoch: {full_epoch/60:.1f} min -> "
          f"30 epochs = {full_epoch*30/3600:.1f} h per run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
