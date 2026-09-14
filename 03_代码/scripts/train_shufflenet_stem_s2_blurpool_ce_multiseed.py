#!/usr/bin/env python3
"""Train the stride-two anti-aliased (blur-pool) ShuffleNetV2 stem (S2B).

Resolution-matched control for the frozen standard stem (S2P): identical
parameter count and identical 16 x 16 grid before stage 2, but the initial
max pooling is replaced by a fixed binomial low-pass filter.

Interpretation
--------------
* S2B close to S2P  -> the Standard/HighRes gap is driven by **resolution**.
* S2B close to S1N  -> the gap is driven by **aliasing**.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import ExperimentSpec, main
from wafermap.stem_antialiasing_models import ShuffleNetV2Stride2BlurPool


EXPERIMENT = ExperimentSpec(
    base_run_name="shufflenet_v2_stem_s2_blurpool_ce_full",
    model_name="ShuffleNetV2Stride2BlurPool",
    log_title=(
        "WM-811K ShuffleNetV2 stride-2 anti-aliased (blur-pool) stem training"
    ),
    model_factory=ShuffleNetV2Stride2BlurPool,
)


if __name__ == "__main__":
    main(EXPERIMENT)
