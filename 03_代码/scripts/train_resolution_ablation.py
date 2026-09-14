#!/usr/bin/env python3
"""Train one cell of the input-resolution x stem ablation.

Answers the reviewer question "why 64 x 64?" by holding the stem configuration
fixed and varying only the input resolution:

=========================================  =======  ======  ===============
run name suffix                            input    stem    grid before s2
=========================================  =======  ======  ===============
``..._standard_res64_ce_full``             64       S2P     16 x 16
``..._highres_res64_ce_full``              64       S1N     64 x 64
``..._standard_res128_ce_full``            128      S2P     32 x 32
``..._highres_res128_ce_full``             128      S1N     128 x 128
=========================================  =======  ======  ===============

The two 64 x 64 cells re-run the frozen configurations through this new code
path and therefore serve as a paired control: if they reproduce the frozen
numbers, the 128 x 128 cells are trustworthy.

Prerequisite
------------
The 128 x 128 processed arrays must exist. They are produced from the raw
pickle, which is *not* on disk as of this writing:

    python scripts/preprocess_wm811k_at_size.py --image-size 128
    python scripts/train_resolution_ablation.py --image-size 128 \
        --stem highres --seed 42

Split identity
--------------
Every resolution reuses ``wm811k_labeled_lot_disjoint.csv`` unchanged, so the
train/val/test membership of each wafer is identical across resolutions and
the comparison is paired at the wafer level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import wafermap.controlled_ce_training as training  # noqa: E402
import wafermap.dataset as dataset_module  # noqa: E402
from wafermap.constants import WM811K_CLASS_NAMES  # noqa: E402
from wafermap.models import ShuffleNetV2Baseline  # noqa: E402
from wafermap.models_improved import ShuffleNetV2HighRes  # noqa: E402
from wafermap.paths import PROCESSED_DATA_DIR  # noqa: E402


STEMS = {
    "standard": ("s2p", ShuffleNetV2Baseline),
    "highres": ("s1n", ShuffleNetV2HighRes),
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train one resolution x stem ablation cell."
    )
    parser.add_argument(
        "--image-size", type=int, required=True,
        help="Input resolution; a matching processed directory must exist.",
    )
    parser.add_argument(
        "--stem", choices=sorted(STEMS), required=True,
        help="standard = stride-2 + maxpool (S2P); highres = stride-1, no pool (S1N).",
    )
    parser.add_argument(
        "--seed", type=int, choices=list(training.SUPPORTED_SEEDS), default=None,
        help="Seed forwarded to the frozen training protocol.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate wiring and print the resolved run name without training.",
    )
    return parser.parse_args(argv)


def processed_dir_for(image_size: int) -> Path:
    return PROCESSED_DATA_DIR / f"wm811k_labeled_{image_size}x{image_size}"


def activate_resolution(image_size: int, verbose: bool = True) -> Path:
    """Point the frozen dataset loader at another resolution.

    ``wafermap.dataset`` resolves ``IMAGE_FILE``, ``METADATA_FILE`` and
    ``IMAGE_SIZE`` from module globals at call time, and
    ``controlled_ce_training`` records ``IMAGE_SIZE`` into its metrics file.
    Rebinding those globals reuses the frozen training loop unchanged, which
    is what keeps the cells comparable.
    """
    directory = processed_dir_for(image_size)
    image_file = directory / "images_uint8.npy"
    metadata_file = directory / "metadata.csv"

    if not image_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"Processed data for {image_size} x {image_size} not found in "
            f"{directory}.\n"
            "Build it first (requires the raw LSWMD.pkl):\n"
            f"  python scripts/preprocess_wm811k_at_size.py "
            f"--image-size {image_size}"
        )

    dataset_module.IMAGE_FILE = image_file
    dataset_module.METADATA_FILE = metadata_file
    dataset_module.IMAGE_SIZE = image_size
    training.IMAGE_SIZE = image_size

    if verbose:
        print(f"activated resolution={image_size} dir={directory}")
    return directory


def build_experiment(image_size: int, stem: str) -> training.ExperimentSpec:
    stem_tag, model_factory = STEMS[stem]
    return training.ExperimentSpec(
        base_run_name=(
            f"shufflenet_v2_{stem}_res{image_size}_ce_full"
        ),
        model_name=f"{model_factory.__name__}@{image_size}px({stem_tag})",
        log_title=(
            f"WM-811K resolution ablation {image_size}x{image_size} "
            f"stem={stem_tag}"
        ),
        model_factory=model_factory,
    )


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)

    if args.image_size not in (64, 128):
        print(
            f"warning: image_size={args.image_size} is outside the planned "
            "64/128 grid; results are still valid but not pre-registered.",
        )

    activate_resolution(args.image_size)
    experiment = build_experiment(args.image_size, args.stem)

    print(f"run_name={experiment.base_run_name}")
    print(f"model={experiment.model_name}")

    if args.dry_run:
        # Touch the dataset once to prove the wiring really resolves.
        dataset = dataset_module.WaferMapDataset("val")
        images, _ = dataset[0]
        expected = (1, args.image_size, args.image_size)
        assert tuple(images.shape) == expected, (
            f"dataset returned {tuple(images.shape)}, expected {expected}"
        )
        print(
            f"dry-run OK: val maps={len(dataset)} "
            f"first_tensor_shape={tuple(images.shape)} "
            f"classes={len(WM811K_CLASS_NAMES)}"
        )
        return

    forwarded = []
    if args.seed is not None:
        forwarded = ["--seed", str(args.seed)]
    sys.argv = [sys.argv[0], *forwarded]
    training.main(experiment)


if __name__ == "__main__":
    main()
