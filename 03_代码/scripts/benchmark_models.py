#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.dataset import WaferMapDataset
from wafermap.models import ResNet18Baseline
from wafermap.models_improved import (
    ShuffleNetV2ECA,
    ShuffleNetV2HighResECA,
)
from wafermap.paths import EXPERIMENT_DIR


RESULT_FILE = (
    EXPERIMENT_DIR
    / "metrics"
    / "model_inference_benchmark.csv"
)

BATCH_SIZES = (1, 128)
WARMUP_STEPS = 10
MEASURE_STEPS = 30

MODEL_CONFIGS = [
    (
        "resnet18_baseline_full",
        ResNet18Baseline,
        EXPERIMENT_DIR
        / "checkpoints"
        / "resnet18_baseline_full"
        / "best.pt",
    ),
    (
        "shufflenet_v2_eca_focal_full",
        ShuffleNetV2ECA,
        EXPERIMENT_DIR
        / "checkpoints"
        / "shufflenet_v2_eca_focal_full"
        / "best.pt",
    ),
    (
        "shufflenet_v2_highres_eca_focal_full",
        ShuffleNetV2HighResECA,
        EXPERIMENT_DIR
        / "checkpoints"
        / "shufflenet_v2_highres_eca_focal_full"
        / "best.pt",
    ),
]


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_model(model_class, checkpoint_file, device):
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
    )

    model = model_class()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(device)
    model.eval()

    return model


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


@torch.inference_mode()
def measure_model(
    model,
    dataset,
    batch_size,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    images, _ = next(iter(loader))
    images = images.to(device)

    for _ in range(WARMUP_STEPS):
        model(images)

    synchronize(device)

    start_time = time.perf_counter()

    for _ in range(MEASURE_STEPS):
        model(images)

    synchronize(device)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    mean_batch_seconds = (
        elapsed_seconds / MEASURE_STEPS
    )

    samples_per_second = (
        batch_size / mean_batch_seconds
    )

    return (
        mean_batch_seconds,
        samples_per_second,
    )


def main():
    device = select_device()
    print(f"Benchmark device: {device}")

    dataset = WaferMapDataset("val")
    rows = []

    for (
        run_name,
        model_class,
        checkpoint_file,
    ) in MODEL_CONFIGS:
        model = load_model(
            model_class,
            checkpoint_file,
            device,
        )

        parameter_count = count_parameters(model)
        checkpoint_size_mb = (
            checkpoint_file.stat().st_size
            / (1024 * 1024)
        )

        for batch_size in BATCH_SIZES:
            mean_seconds, samples_per_second = (
                measure_model(
                    model,
                    dataset,
                    batch_size,
                    device,
                )
            )

            row = {
                "run_name": run_name,
                "model_name": model_class.__name__,
                "device": str(device),
                "batch_size": batch_size,
                "parameter_count": parameter_count,
                "checkpoint_size_mb": checkpoint_size_mb,
                "mean_batch_seconds": mean_seconds,
                "samples_per_second": samples_per_second,
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
            }

            rows.append(row)

            print(
                f"{run_name} "
                f"batch={batch_size} "
                f"params={parameter_count:,} "
                f"samples/s="
                f"{samples_per_second:.1f}"
            )

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with RESULT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()