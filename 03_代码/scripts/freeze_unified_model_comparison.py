#!/usr/bin/env python3
"""Freeze all five model families before unified benchmark evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.lightweight_baselines import (
    EfficientNetB0Baseline,
    MobileNetV3SmallBaseline,
)
from wafermap.models import ResNet18Baseline, ShuffleNetV2Baseline
from wafermap.models_improved import ShuffleNetV2HighRes


OUTPUT = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260801_五模型统一评估冻结清单.json"
)
EVALUATION_ID = "20260801_five_model_multiseed_unified_test"
SEEDS = (42, 123, 2026)

MODEL_SPECS = (
    {
        "model_id": "highres_shufflenet_v2",
        "model_name": "ShuffleNetV2HighRes",
        "display_name": "HighRes ShuffleNetV2",
        "base_run_name": "shufflenet_v2_highres_ce_full",
        "parameter_count": 1262397,
    },
    {
        "model_id": "standard_shufflenet_v2",
        "model_name": "ShuffleNetV2Baseline",
        "display_name": "Standard ShuffleNetV2",
        "base_run_name": "shufflenet_v2_standard_ce_full",
        "parameter_count": 1262397,
    },
    {
        "model_id": "resnet18",
        "model_name": "ResNet18Baseline",
        "display_name": "ResNet18",
        "base_run_name": "resnet18_baseline_full",
        "parameter_count": 11174857,
    },
    {
        "model_id": "mobilenet_v3_small",
        "model_name": "MobileNetV3SmallBaseline",
        "display_name": "MobileNetV3-Small",
        "base_run_name": "mobilenet_v3_small_ce_full",
        "parameter_count": 1526793,
    },
    {
        "model_id": "efficientnet_b0",
        "model_name": "EfficientNetB0Baseline",
        "display_name": "EfficientNet-B0",
        "base_run_name": "efficientnet_b0_ce_full",
        "parameter_count": 4018501,
    },
)

MODEL_FACTORIES = {
    "highres_shufflenet_v2": ShuffleNetV2HighRes,
    "standard_shufflenet_v2": ShuffleNetV2Baseline,
    "resnet18": ResNet18Baseline,
    "mobilenet_v3_small": MobileNetV3SmallBaseline,
    "efficientnet_b0": EfficientNetB0Baseline,
}

CODE_FILES = (
    "03_代码/src/wafermap/models.py",
    "03_代码/src/wafermap/models_improved.py",
    "03_代码/src/wafermap/lightweight_baselines.py",
    "03_代码/src/wafermap/dataset.py",
    "03_代码/src/wafermap/constants.py",
    "03_代码/src/wafermap/paths.py",
)

DATA_FILES = (
    "02_数据/processed/wm811k_labeled_64x64/images_uint8.npy",
    "02_数据/processed/wm811k_labeled_64x64/metadata.csv",
    "02_数据/splits/wm811k_labeled_lot_disjoint.csv",
)

REUSED_PREDICTIONS = {
    ("highres_shufflenet_v2", 42): (
        "04_实验/metrics/20260729_highres_ce_multiseed_final_test/"
        "predictions/predictions_seed42.csv"
    ),
    ("highres_shufflenet_v2", 123): (
        "04_实验/metrics/20260729_highres_ce_multiseed_final_test/"
        "predictions/predictions_seed123.csv"
    ),
    ("highres_shufflenet_v2", 2026): (
        "04_实验/metrics/20260729_highres_ce_multiseed_final_test/"
        "predictions/predictions_seed2026.csv"
    ),
    ("resnet18", 42): (
        "04_实验/metrics/resnet18_baseline_full_predictions.csv"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_name(base_run_name: str, seed: int) -> str:
    return base_run_name if seed == 42 else f"{base_run_name}_seed{seed}"


def checkpoint_record(spec: dict, seed: int) -> dict:
    name = run_name(spec["base_run_name"], seed)
    relative_checkpoint = f"04_实验/checkpoints/{name}/best.pt"
    relative_history = f"04_实验/metrics/{name}_history.csv"
    checkpoint_file = PROJECT_ROOT / relative_checkpoint
    history_file = PROJECT_ROOT / relative_history

    if not checkpoint_file.is_file() or not history_file.is_file():
        raise FileNotFoundError(f"Incomplete run: {name}")

    history = pd.read_csv(history_file)
    if len(history) != 30 or history["epoch"].tolist() != list(range(1, 31)):
        raise ValueError(f"Training history is not 30 complete epochs: {name}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = MODEL_FACTORIES[spec["model_id"]]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actual_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if actual_parameter_count != spec["parameter_count"]:
        raise ValueError(f"Parameter count mismatch: {name}")

    expected = {
        "run_name": name,
        "model_name": spec["model_name"],
        "loss_name": "CrossEntropyLoss",
        "parameter_count": spec["parameter_count"],
        "random_seed": seed,
    }
    legacy_resnet_seed42 = spec["model_id"] == "resnet18" and seed == 42
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if legacy_resnet_seed42 and key in {
            "run_name",
            "model_name",
            "loss_name",
            "parameter_count",
        } and actual is None:
            continue
        if actual != value:
            raise ValueError(
                f"Checkpoint mismatch for {name}: {key}={actual!r}"
            )

    best_row = history.loc[history["val_macro_f1"].idxmax()]
    if int(best_row["epoch"]) != int(checkpoint["epoch"]):
        raise ValueError(f"Best epoch mismatch: {name}")
    if not np.isclose(
        float(best_row["val_macro_f1"]),
        float(checkpoint["best_val_macro_f1"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Best validation Macro-F1 mismatch: {name}")

    reused = REUSED_PREDICTIONS.get((spec["model_id"], seed))
    if reused is not None and not (PROJECT_ROOT / reused).is_file():
        raise FileNotFoundError(f"Missing reusable predictions: {reused}")

    return {
        "seed": seed,
        "run_name": name,
        "checkpoint_path": relative_checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "history_path": relative_history,
        "history_sha256": sha256_file(history_file),
        "best_epoch": int(checkpoint["epoch"]),
        "validation_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "evaluation_action": "reuse_frozen_predictions" if reused else "infer_once",
        "reused_predictions_path": reused,
        "reused_predictions_sha256": (
            sha256_file(PROJECT_ROOT / reused) if reused else None
        ),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite freeze manifest: {OUTPUT}")

    models = []
    for spec in MODEL_SPECS:
        record = dict(spec)
        record["checkpoints"] = [
            checkpoint_record(spec, seed) for seed in SEEDS
        ]
        models.append(record)

    manifest = {
        "schema_version": 1,
        "freeze_date": str(date.today()),
        "status": "frozen_before_unified_test",
        "evaluation_id": EVALUATION_ID,
        "task": "WM-811K lot-disjoint nine-class wafer bin map classification",
        "selection_split": "validation",
        "primary_metric": "Macro-F1",
        "seeds": list(SEEDS),
        "test_split_role": (
            "fixed benchmark test split with documented historical access; "
            "not an untouched independent holdout"
        ),
        "test_tuning_allowed": False,
        "models": models,
        "frozen_code": [
            {"path": path, "sha256": sha256_file(PROJECT_ROOT / path)}
            for path in CODE_FILES
        ],
        "dataset": [
            {"path": path, "sha256": sha256_file(PROJECT_ROOT / path)}
            for path in DATA_FILES
        ],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": {
            "test_samples": 25943,
            "batch_size": 256,
            "num_workers": 0,
            "inference_mode": True,
            "new_checkpoint_inference_count": 11,
            "reused_prediction_count": 4,
            "post_test_tuning_allowed": False,
            "sample_standard_deviation_ddof": 1,
        },
    }

    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Freeze manifest written: {OUTPUT}")
    print(f"Freeze manifest SHA-256: {sha256_file(OUTPUT)}")
    print("Models: 5; checkpoints: 15; infer once: 11; reuse: 4")


if __name__ == "__main__":
    main()
