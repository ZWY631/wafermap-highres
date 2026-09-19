#!/usr/bin/env python3
"""Evaluate the input-resolution x stem ablation on the fixed test split.

This mirrors ``evaluate_stem_antialiasing_multiseed.py``: one evaluator, one
frozen split, one checkpoint-selection rule, and paired comparisons at the
level of individual wafer maps. It differs in one respect only -- the input
resolution varies between cells, so the dataset is re-pointed at the matching
``02_数据/processed/wm811k_labeled_{size}x{size}`` directory before each block.

Two of the four cells are *controls*: they re-train the already published
64 x 64 configurations through the resolution-ablation code path. A third pair
of anchor cells re-evaluates the **frozen** 64 x 64 checkpoints through this
evaluator. If the anchors do not reproduce the published values
(S2P Macro-F1 80.4947, S1N Macro-F1 90.2029), the evaluator is not consistent
with the frozen one and the run aborts before any 128 x 128 number is
produced.

Usage
-----
    python 03_代码/scripts/evaluate_resolution_ablation_multiseed.py --anchor
    python 03_代码/scripts/evaluate_resolution_ablation_multiseed.py --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import wafermap.dataset as dataset_module  # noqa: E402
from wafermap.constants import WM811K_CLASS_NAMES  # noqa: E402
from wafermap.models import ShuffleNetV2Baseline  # noqa: E402
from wafermap.models_improved import ShuffleNetV2HighRes  # noqa: E402
from wafermap.paths import PROCESSED_DATA_DIR  # noqa: E402

SEEDS = (42, 123, 2026)
EXPECTED_TEST_SAMPLES = 25943
EXPECTED_PARAMETERS = 1_262_397
BATCH_SIZE = 256
OUTPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / "20260914_resolution_ablation_multiseed_test"
TABLE_PATH = PROJECT_ROOT / "05_结果" / "tables" / "table_v7_resolution_ablation_multiseed_test.csv"

# Published anchors. The frozen checkpoints must reproduce these through this
# evaluator before any new number is trusted.
ANCHOR_MACRO_F1 = {"S2P": 0.804947, "S1N": 0.902029}
ANCHOR_TOLERANCE = 0.0005


@dataclass(frozen=True)
class ResolutionCell:
    cell_id: str
    display_name: str
    image_size: int
    stem: str
    base_run_name: str
    model_factory: type
    role: str  # "anchor" | "control" | "new"


CELLS: tuple[ResolutionCell, ...] = (
    ResolutionCell(
        "F64-S2P", "frozen 64 x 64, stride 2 + max pool", 64, "standard",
        "shufflenet_v2_standard_ce_full", ShuffleNetV2Baseline, "anchor",
    ),
    ResolutionCell(
        "F64-S1N", "frozen 64 x 64, stride 1 + identity (HighRes)", 64, "highres",
        "shufflenet_v2_highres_ce_full", ShuffleNetV2HighRes, "anchor",
    ),
    ResolutionCell(
        "R64-S2P", "control 64 x 64, stride 2 + max pool", 64, "standard",
        "shufflenet_v2_standard_res64_ce_full", ShuffleNetV2Baseline, "control",
    ),
    ResolutionCell(
        "R64-S1N", "control 64 x 64, stride 1 + identity", 64, "highres",
        "shufflenet_v2_highres_res64_ce_full", ShuffleNetV2HighRes, "control",
    ),
    ResolutionCell(
        "R128-S2P", "128 x 128, stride 2 + max pool", 128, "standard",
        "shufflenet_v2_standard_res128_ce_full", ShuffleNetV2Baseline, "new",
    ),
    ResolutionCell(
        "R128-S1N", "128 x 128, stride 1 + identity", 128, "highres",
        "shufflenet_v2_highres_res128_ce_full", ShuffleNetV2HighRes, "new",
    ),
)


def run_name_for_seed(base_run_name: str, seed: int) -> str:
    return base_run_name if seed == 42 else f"{base_run_name}_seed{seed}"


def activate_resolution(image_size: int) -> Path:
    directory = PROCESSED_DATA_DIR / f"wm811k_labeled_{image_size}x{image_size}"
    image_file = directory / "images_uint8.npy"
    metadata_file = directory / "metadata.csv"
    for path in (image_file, metadata_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Processed data for {image_size} x {image_size} is missing: {path}\n"
                f"Run: python 03_代码/scripts/preprocess_wm811k_at_size.py "
                f"--image-size {image_size}"
            )
    dataset_module.IMAGE_FILE = image_file
    dataset_module.METADATA_FILE = metadata_file
    dataset_module.IMAGE_SIZE = image_size
    return directory


def checkpoint_path(cell: ResolutionCell, seed: int) -> Path:
    run_name = run_name_for_seed(cell.base_run_name, seed)
    return PROJECT_ROOT / "04_实验" / "checkpoints" / run_name / "best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device() -> torch.device:
    """Match the frozen protocol: Apple MPS when available, else CPU."""
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_cell(cell: ResolutionCell, seed: int, dataset, device: torch.device) -> dict:
    path = checkpoint_path(cell, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = cell.model_factory(num_classes=len(WM811K_CLASS_NAMES))
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise ValueError(f"{cell.cell_id}: unexpected parameter count {parameters}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(targets.numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(labels)
    per_class = f1_score(
        y_true, y_pred, labels=list(range(len(WM811K_CLASS_NAMES))),
        average=None, zero_division=0,
    )
    scratch_index = WM811K_CLASS_NAMES.index("Scratch")
    return {
        "cell_id": cell.cell_id,
        "display_name": cell.display_name,
        "image_size": cell.image_size,
        "stem": cell.stem,
        "role": cell.role,
        "seed": seed,
        "parameters": parameters,
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "val_macro_f1": float(checkpoint.get("best_val_macro_f1", float("nan"))),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "scratch_f1": float(per_class[scratch_index]),
        "test_samples": int(len(y_true)),
    }


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "scratch_f1"]
    rows = []
    for (cell_id, display_name), group in per_seed.groupby(
        ["cell_id", "display_name"], sort=False
    ):
        row = {
            "cell_id": cell_id,
            "display_name": display_name,
            "image_size": int(group["image_size"].iloc[0]),
            "stem": group["stem"].iloc[0],
            "role": group["role"].iloc[0],
            "seeds": int(len(group)),
        }
        for metric in metrics:
            row[f"{metric}_mean_percent"] = round(float(group[metric].mean()) * 100, 4)
            row[f"{metric}_sample_std_percent"] = round(
                float(group[metric].std(ddof=1)) * 100, 4
            )
        rows.append(row)
    return pd.DataFrame(rows)


def check_anchors(per_seed: pd.DataFrame) -> dict:
    """Abort unless the frozen checkpoints reproduce the published values."""
    findings = {}
    for cell_id, published in ANCHOR_MACRO_F1.items():
        rows = per_seed[per_seed["cell_id"] == f"F64-{cell_id}"]
        if rows.empty:
            raise RuntimeError(f"anchor cell F64-{cell_id} was not evaluated")
        observed = float(rows["macro_f1"].mean())
        delta = abs(observed - published)
        findings[cell_id] = {"observed": observed, "published": published, "delta": delta}
        if delta > ANCHOR_TOLERANCE:
            raise RuntimeError(
                f"anchor mismatch for {cell_id}: observed {observed:.6f} vs "
                f"published {published:.6f} (delta {delta:.6f} > "
                f"{ANCHOR_TOLERANCE}). The evaluator is not consistent with the "
                f"frozen one; refusing to report new numbers."
            )
    return findings


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--anchor", action="store_true",
        help="evaluate only the two frozen 64x64 anchor cells.",
    )
    mode.add_argument(
        "--execute", action="store_true",
        help="evaluate every cell whose checkpoints exist.",
    )
    mode.add_argument(
        "--preflight", action="store_true",
        help="report which checkpoints are present without evaluating.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.preflight:
        for cell in CELLS:
            present = [s for s in SEEDS if checkpoint_path(cell, s).is_file()]
            print(f"{cell.cell_id:9s} {cell.image_size:>3}px {cell.role:8s} seeds={present}")
        return 0

    cells = [c for c in CELLS if c.role == "anchor"] if args.anchor else list(CELLS)

    device = select_device()
    print(f"device={device.type}")

    records: list[dict] = []
    current_size: int | None = None
    dataset = None
    for cell in cells:
        missing = [s for s in SEEDS if not checkpoint_path(cell, s).is_file()]
        if missing:
            print(f"skip {cell.cell_id}: missing checkpoints for seeds {missing}")
            continue
        if cell.image_size != current_size:
            directory = activate_resolution(cell.image_size)
            dataset = dataset_module.WaferMapDataset("test")
            if len(dataset) != EXPECTED_TEST_SAMPLES:
                raise ValueError(
                    f"{cell.image_size}px: expected {EXPECTED_TEST_SAMPLES} test "
                    f"samples, got {len(dataset)}"
                )
            current_size = cell.image_size
            print(f"activated {cell.image_size}x{cell.image_size} ({directory.name})")
        for seed in SEEDS:
            record = evaluate_cell(cell, seed, dataset, device)
            records.append(record)
            print(
                f"  {cell.cell_id} seed{seed}: acc={record['accuracy']*100:.4f} "
                f"macroF1={record['macro_f1']*100:.4f} "
                f"scratchF1={record['scratch_f1']*100:.4f}"
            )

    if not records:
        print("no checkpoints available; nothing to evaluate")
        return 1

    per_seed = pd.DataFrame(records)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    anchor_findings = {}
    if {"F64-S2P", "F64-S1N"}.issubset(set(per_seed["cell_id"])):
        anchor_findings = check_anchors(per_seed)
        print("\nanchor check passed:")
        for cell_id, values in anchor_findings.items():
            print(
                f"  {cell_id}: {values['observed']*100:.4f} vs published "
                f"{values['published']*100:.4f} (delta {values['delta']*100:.4f} pp)"
            )

    per_seed.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)
    summary = summarise(per_seed)
    summary.to_csv(OUTPUT_DIR / "aggregate_metrics.csv", index=False)
    summary.to_csv(TABLE_PATH, index=False)
    (OUTPUT_DIR / "anchor_check.json").write_text(
        json.dumps(anchor_findings, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {TABLE_PATH}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
