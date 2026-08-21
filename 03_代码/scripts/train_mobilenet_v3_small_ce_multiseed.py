#!/usr/bin/env python3
"""Train the MobileNetV3-Small + CE comparison baseline."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.lightweight_baselines import MobileNetV3SmallBaseline


EXPERIMENT = ExperimentSpec(
    base_run_name="mobilenet_v3_small_ce_full",
    model_name="MobileNetV3SmallBaseline",
    log_title="WM-811K MobileNetV3-Small Cross Entropy baseline",
    model_factory=MobileNetV3SmallBaseline,
)


if __name__ == "__main__":
    main(EXPERIMENT)
