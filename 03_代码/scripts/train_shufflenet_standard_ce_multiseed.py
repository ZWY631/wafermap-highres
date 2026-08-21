#!/usr/bin/env python3
"""Train the standard-downsampling ShuffleNetV2 + CE control."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.models import ShuffleNetV2Baseline


EXPERIMENT = ExperimentSpec(
    base_run_name="shufflenet_v2_standard_ce_full",
    model_name="ShuffleNetV2Baseline",
    log_title=(
        "WM-811K Standard ShuffleNetV2 Cross Entropy control training"
    ),
    model_factory=ShuffleNetV2Baseline,
)


if __name__ == "__main__":
    main(EXPERIMENT)
