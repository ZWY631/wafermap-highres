#!/usr/bin/env python3
"""Train the stride-one anti-aliased (blur-pool) ShuffleNetV2 stem (S1B).

Intermediate-resolution control: the same 32 x 32 grid before stage 2 as S2N
and S1P, but reached with a low-pass filter. Together with S2B this
triangulates whether the stem effect tracks resolution or aliasing.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.stem_antialiasing_models import ShuffleNetV2Stride1BlurPool


EXPERIMENT = ExperimentSpec(
    base_run_name="shufflenet_v2_stem_s1_blurpool_ce_full",
    model_name="ShuffleNetV2Stride1BlurPool",
    log_title=(
        "WM-811K ShuffleNetV2 stride-1 anti-aliased (blur-pool) stem training"
    ),
    model_factory=ShuffleNetV2Stride1BlurPool,
)


if __name__ == "__main__":
    main(EXPERIMENT)
