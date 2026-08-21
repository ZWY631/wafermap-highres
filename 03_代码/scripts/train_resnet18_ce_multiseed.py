#!/usr/bin/env python3
"""Run the paired-seed ResNet18 + CE baseline protocol."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.models import ResNet18Baseline


EXPERIMENT = ExperimentSpec(
    base_run_name="resnet18_baseline_full",
    model_name="ResNet18Baseline",
    log_title="WM-811K ResNet18 baseline training",
    model_factory=ResNet18Baseline,
)


if __name__ == "__main__":
    main(EXPERIMENT)
