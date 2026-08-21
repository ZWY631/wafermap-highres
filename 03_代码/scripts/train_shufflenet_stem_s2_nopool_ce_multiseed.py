#!/usr/bin/env python3
"""Train the stride-two, no-pool ShuffleNetV2 stem ablation."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.stem_ablation_models import ShuffleNetV2Stride2NoPool


EXPERIMENT = ExperimentSpec(
    base_run_name="shufflenet_v2_stem_s2_nopool_ce_full",
    model_name="ShuffleNetV2Stride2NoPool",
    log_title=(
        "WM-811K ShuffleNetV2 stride-2 no-pool stem ablation training"
    ),
    model_factory=ShuffleNetV2Stride2NoPool,
)


if __name__ == "__main__":
    main(EXPERIMENT)
