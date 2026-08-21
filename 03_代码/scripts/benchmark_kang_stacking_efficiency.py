#!/usr/bin/env python3
"""Benchmark frozen Kang stacking branches on synthetic inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from wafermap.stacking_reproduction import HandcraftedFNN, VGG16WaferCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "04_实验/checkpoints/kang_kang_stacking_lot_reproduction"
OUTPUT_DIR = PROJECT_ROOT / "04_实验/benchmarks/20260810_kang_stacking_efficiency"
PAPER_TABLE = PROJECT_ROOT / "05_结果/tables/table_kang_stacking_efficiency.csv"
BATCH_SIZES = (1, 128)
WARMUP = 10
MEASURE = 30
ROUNDS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_stage(seed, stage, model):
    path = RUN_ROOT / f"seed{seed}" / "stages" / stage / "best.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, path


def static_complexity(model, sample):
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            output = model(sample)
    return int(sum(p.numel() for p in model.parameters())), int(counter.get_total_flops()), tuple(output.shape)


@torch.inference_mode()
def benchmark(model, device, sample_shape, branch):
    model = model.to(device)
    rows = []
    for batch_size in BATCH_SIZES:
        inputs = torch.randn((batch_size, *sample_shape), device=device)
        for _ in range(WARMUP):
            model(inputs)
        sync(device)
        for round_index in range(1, ROUNDS + 1):
            sync(device)
            start = time.perf_counter()
            for _ in range(MEASURE):
                output = model(inputs)
            sync(device)
            seconds = (time.perf_counter() - start) / MEASURE
            rows.append({
                "branch": branch,
                "device": str(device),
                "batch_size": batch_size,
                "round": round_index,
                "batch_latency_ms": seconds * 1000,
                "latency_ms_per_sample": seconds * 1000 / batch_size,
                "samples_per_second": batch_size / seconds,
            })
    model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    return rows


def main():
    args = parse_args()
    if OUTPUT_DIR.exists() or PAPER_TABLE.exists():
        raise FileExistsError("Refusing to overwrite Kang efficiency outputs.")
    seeds = (42, 123, 2026)
    for seed in seeds:
        manifest = RUN_ROOT / f"seed{seed}" / "run_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "training_complete_test_not_accessed":
            raise ValueError(f"Incomplete seed {seed}")
    device = select_device()
    complexity = []
    rounds = []
    for branch, model_class, shape, stage in (
        ("mfe", HandcraftedFNN, (59,), "final_mfe"),
        ("cnn", VGG16WaferCNN, (1, 64, 64), "final_cnn"),
    ):
        model, checkpoint = load_stage(42, stage, model_class())
        sample = torch.zeros((1, *shape), dtype=torch.float32)
        params, flops, output_shape = static_complexity(model, sample)
        complexity.append({
            "branch": branch,
            "parameter_count": params,
            "parameters_million": params / 1_000_000,
            "flops": flops,
            "flops_million": flops / 1_000_000,
            "checkpoint_mib": checkpoint.stat().st_size / (1024**2),
            "output_shape": str(output_shape),
        })
        if args.run:
            rounds.extend(benchmark(model, device, shape, branch))
    complexity_frame = pd.DataFrame(complexity)
    if not args.run:
        print(complexity_frame.to_string(index=False))
        return
    rounds_frame = pd.DataFrame(rounds)
    aggregate = rounds_frame.groupby(["branch", "device", "batch_size"], as_index=False).agg(
        rounds=("round", "count"),
        batch_latency_ms_mean=("batch_latency_ms", "mean"),
        batch_latency_ms_sample_std=("batch_latency_ms", "std"),
        latency_ms_per_sample_mean=("latency_ms_per_sample", "mean"),
        samples_per_second_mean=("samples_per_second", "mean"),
    )
    OUTPUT_DIR.mkdir(parents=True)
    complexity_frame.to_csv(OUTPUT_DIR / "complexity.csv", index=False)
    rounds_frame.to_csv(OUTPUT_DIR / "benchmark_rounds.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    complexity_frame.to_csv(PAPER_TABLE, index=False)
    (OUTPUT_DIR / "analysis_manifest.json").write_text(json.dumps({
        "analysis_id": "20260810_kang_stacking_efficiency",
        "completed_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "synthetic_input_only": True,
        "seeds_checkpointed": list(seeds),
        "checkpoint_used_for_complexity": 42,
        "warmup_steps": WARMUP,
        "measure_steps": MEASURE,
        "rounds": ROUNDS,
        "complexity": complexity,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
