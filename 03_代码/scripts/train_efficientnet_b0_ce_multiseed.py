#!/usr/bin/env python3
"""Train the EfficientNet-B0 + CE comparison baseline."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.lightweight_baselines import EfficientNetB0Baseline


EXPERIMENT = ExperimentSpec(
    base_run_name="efficientnet_b0_ce_full",
    model_name="EfficientNetB0Baseline",
    log_title="WM-811K EfficientNet-B0 Cross Entropy baseline",
    model_factory=EfficientNetB0Baseline,
)


if __name__ == "__main__":
    main(EXPERIMENT)
