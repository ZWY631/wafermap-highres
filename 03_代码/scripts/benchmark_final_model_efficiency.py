#!/usr/bin/env python3
"""Measure final-model complexity and MPS forward-pass speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "03_代码"
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.models import ResNet18Baseline
from wafermap.models_improved import ShuffleNetV2HighRes


ANALYSIS_ID = "20260729_final_model_complexity_mps_benchmark"
FREEZE_MANIFEST = (
    PROJECT_ROOT / "00_项目管理" / "20260729_最终模型冻结清单.json"
)
EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "c682922b1c291f03a41a81e5d15ebfe08e4456e77c816f7943041fc11fcef8db"
)
EXPECTED_BASELINE_SOURCE_SHA256 = (
    "01d1af8ebaa12b5b390742a9e56bd81902b55ba2653a4f80f1f586b63c22ba35"
)

INPUT_SHAPE = (1, 1, 64, 64)
BATCH_SIZES = (1, 128)
WARMUP_STEPS = 10
MEASURE_STEPS = 30
ROUNDS = 3

MODEL_CONFIGS = (
    {
        "model_id": "resnet18_baseline",
        "display_name": "ResNet18 Baseline",
        "model_class": ResNet18Baseline,
        "checkpoint": (
            PROJECT_ROOT
            / "04_实验"
            / "checkpoints"
            / "resnet18_baseline_full"
            / "best.pt"
        ),
        "checkpoint_sha256": (
            "9cb3f6317272b65a0ef21d66505230a567ef7e7074d820477f63df5c70982b83"
        ),
        "expected_parameters": 11174857,
        "expected_flops": 283255808,
    },
    {
        "model_id": "highres_shufflenetv2_ce",
        "display_name": "Final HighRes ShuffleNetV2",
        "model_class": ShuffleNetV2HighRes,
        "checkpoint": (
            PROJECT_ROOT
            / "04_实验"
            / "checkpoints"
            / "shufflenet_v2_highres_ce_full"
            / "best.pt"
        ),
        "checkpoint_sha256": (
            "cafebb6a76ef566ea05fa47e7745677c35cb3abbec87b1de468c0898b02c67bc"
        ),
        "expected_parameters": 1262397,
        "expected_flops": 356414464,
    },
)

METRIC_FILENAMES = (
    "model_complexity.csv",
    "mps_benchmark_rounds.csv",
    "mps_benchmark_summary.csv",
    "efficiency_summary.json",
    "analysis_manifest.json",
)
FIGURE_STEM = "final_model_complexity_mps_comparison"
PAPER_TABLE_NAME = "table_final_model_complexity_mps.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare ResNet18 with the frozen final HighRes ShuffleNetV2. "
            "The benchmark uses synthetic 64x64 inputs and never reads a dataset."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate checkpoints and calculate static complexity only.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Calculate complexity and run the MPS forward-pass benchmark.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "metrics_dir": output_root / "04_实验" / "benchmarks" / ANALYSIS_ID,
        "figure_dir": output_root / "05_结果" / "figures" / "model_results",
        "table_dir": output_root / "05_结果" / "tables",
    }


def expected_output_files(output_root: Path) -> list[Path]:
    paths = output_paths(output_root)
    files = [paths["metrics_dir"] / name for name in METRIC_FILENAMES]
    files.extend(
        [
            paths["figure_dir"] / f"{FIGURE_STEM}.png",
            paths["figure_dir"] / f"{FIGURE_STEM}.pdf",
            paths["table_dir"] / PAPER_TABLE_NAME,
        ]
    )
    return files


def ensure_outputs_are_available(output_root: Path):
    paths = output_paths(output_root)
    existing = []
    if paths["metrics_dir"].exists():
        existing.append(paths["metrics_dir"])
    existing.extend(path for path in expected_output_files(output_root) if path.exists())
    if existing:
        formatted = "\n".join(f"- {path}" for path in sorted(set(existing)))
        raise FileExistsError(
            "Refusing to overwrite existing efficiency outputs:\n"
            f"{formatted}"
        )


def load_freeze_manifest() -> dict:
    if not FREEZE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing freeze manifest: {FREEZE_MANIFEST}")
    if sha256_file(FREEZE_MANIFEST) != EXPECTED_FREEZE_MANIFEST_SHA256:
        raise ValueError("Final model freeze manifest hash has changed.")
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("selected_architecture_id") != "highres_ce_no_eca":
        raise ValueError("Frozen architecture is not the expected final model.")
    if int(manifest.get("parameter_count", -1)) != 1262397:
        raise ValueError("Frozen final parameter count is incorrect.")
    return manifest


def validate_frozen_environment(freeze_manifest: dict):
    expected = freeze_manifest.get("frozen_environment", {})
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    for key, actual_value in actual.items():
        expected_value = expected.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Frozen environment mismatch: {key}={actual_value!r}, "
                f"expected {expected_value!r}."
            )


def validate_source_files(freeze_manifest: dict) -> dict[str, str]:
    frozen_code = {
        item["path"]: item["sha256"]
        for item in freeze_manifest.get("frozen_code", [])
    }
    expected_hashes = {
        "03_代码/src/wafermap/models.py": EXPECTED_BASELINE_SOURCE_SHA256,
        "03_代码/src/wafermap/models_improved.py": frozen_code.get(
            "03_代码/src/wafermap/models_improved.py"
        ),
    }
    actual_hashes = {}
    for relative_path, expected_hash in expected_hashes.items():
        if not expected_hash:
            raise ValueError(f"Missing frozen source hash: {relative_path}")
        source_file = PROJECT_ROOT / relative_path
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing model source: {source_file}")
        actual_hash = sha256_file(source_file)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen model source hash mismatch: {relative_path}")
        actual_hashes[relative_path] = actual_hash
    return actual_hashes


def validate_checkpoint(config: dict, freeze_manifest: dict) -> dict:
    checkpoint_file = config["checkpoint"]
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")
    if sha256_file(checkpoint_file) != config["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_file}")

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_file}")

    if config["model_id"] == "highres_shufflenetv2_ce":
        expected_fields = {
            "run_name": "shufflenet_v2_highres_ce_full",
            "model_name": "ShuffleNetV2HighRes",
            "loss_name": "CrossEntropyLoss",
            "parameter_count": freeze_manifest["parameter_count"],
            "image_size": 64,
            "random_seed": 42,
        }
        for key, expected_value in expected_fields.items():
            if checkpoint.get(key) != expected_value:
                raise ValueError(
                    f"Final checkpoint mismatch: {key}={checkpoint.get(key)!r}, "
                    f"expected {expected_value!r}."
                )
    return checkpoint


def load_model(config: dict, checkpoint: dict) -> nn.Module:
    model = config["model_class"]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != config["expected_parameters"]:
        raise ValueError(
            f"Parameter count mismatch for {config['display_name']}: "
            f"{parameter_count}"
        )
    return model


def state_dict_size_bytes(model: nn.Module) -> int:
    total = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total


def calculate_complexity(config: dict, model: nn.Module) -> dict:
    dummy_input = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            output = model(dummy_input)
    if tuple(output.shape) != (1, 9):
        raise ValueError(
            f"Unexpected output shape for {config['display_name']}: {output.shape}"
        )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    flops = int(counter.get_total_flops())
    if flops <= 0:
        raise ValueError(f"FLOP counter returned zero for {config['display_name']}")
    if flops != config["expected_flops"]:
        raise ValueError(
            f"FLOP count mismatch for {config['display_name']}: {flops}, "
            f"expected {config['expected_flops']}"
        )

    return {
        "model_id": config["model_id"],
        "model_name": config["display_name"],
        "input_shape": "1x1x64x64",
        "parameter_count": int(total_parameters),
        "trainable_parameter_count": int(trainable_parameters),
        "parameters_million": float(total_parameters / 1_000_000),
        "parameter_memory_fp32_mib": float(
            total_parameters * 4 / (1024**2)
        ),
        "state_dict_memory_mib": float(state_dict_size_bytes(model) / (1024**2)),
        "training_checkpoint_mib": float(
            config["checkpoint"].stat().st_size / (1024**2)
        ),
        "conv_linear_flops": flops,
        "conv_linear_flops_million": float(flops / 1_000_000),
        "estimated_macs": int(flops // 2),
        "estimated_macs_million": float(flops / 2 / 1_000_000),
    }


def select_mps_device() -> torch.device:
    if not torch.backends.mps.is_built():
        raise RuntimeError("This PyTorch build does not include MPS support.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this Mac.")
    return torch.device("mps")


def synchronize(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark_model(
    config: dict,
    model: nn.Module,
    device: torch.device,
) -> list[dict]:
    model = model.to(device)
    rows = []

    for batch_size in BATCH_SIZES:
        torch.manual_seed(20260729 + batch_size)
        inputs = torch.randn(
            batch_size,
            INPUT_SHAPE[1],
            INPUT_SHAPE[2],
            INPUT_SHAPE[3],
            dtype=torch.float32,
            device=device,
        )

        for _ in range(WARMUP_STEPS):
            model(inputs)
        synchronize(device)

        for round_index in range(1, ROUNDS + 1):
            synchronize(device)
            start_time = time.perf_counter()
            for _ in range(MEASURE_STEPS):
                output = model(inputs)
            synchronize(device)
            elapsed_seconds = time.perf_counter() - start_time

            if tuple(output.shape) != (batch_size, 9):
                raise ValueError(
                    f"Unexpected benchmark output shape: {tuple(output.shape)}"
                )
            mean_batch_seconds = elapsed_seconds / MEASURE_STEPS
            rows.append(
                {
                    "model_id": config["model_id"],
                    "model_name": config["display_name"],
                    "device": str(device),
                    "batch_size": batch_size,
                    "round": round_index,
                    "warmup_steps": WARMUP_STEPS,
                    "measure_steps": MEASURE_STEPS,
                    "mean_batch_seconds": mean_batch_seconds,
                    "batch_latency_ms": mean_batch_seconds * 1000,
                    "latency_ms_per_sample": (
                        mean_batch_seconds / batch_size * 1000
                    ),
                    "samples_per_second": batch_size / mean_batch_seconds,
                }
            )

        del inputs

    model = model.to("cpu")
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return rows


def aggregate_benchmark(rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = (
        "mean_batch_seconds",
        "batch_latency_ms",
        "latency_ms_per_sample",
        "samples_per_second",
    )
    for (model_id, model_name, device, batch_size), group in rounds.groupby(
        ["model_id", "model_name", "device", "batch_size"],
        sort=False,
    ):
        row = {
            "model_id": model_id,
            "model_name": model_name,
            "device": device,
            "batch_size": int(batch_size),
            "rounds": len(group),
            "warmup_steps": WARMUP_STEPS,
            "measure_steps_per_round": MEASURE_STEPS,
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def add_relative_complexity(complexity: pd.DataFrame) -> pd.DataFrame:
    result = complexity.copy()
    baseline = result.loc[result["model_id"] == "resnet18_baseline"].iloc[0]
    result["parameter_change_percent_vs_baseline"] = (
        result["parameter_count"] / baseline["parameter_count"] - 1.0
    ) * 100
    result["flops_change_percent_vs_baseline"] = (
        result["conv_linear_flops"] / baseline["conv_linear_flops"] - 1.0
    ) * 100
    result["state_dict_size_change_percent_vs_baseline"] = (
        result["state_dict_memory_mib"] / baseline["state_dict_memory_mib"] - 1.0
    ) * 100
    return result


def format_mean_sd(mean: float, sample_std: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} +/- {sample_std:.{decimals}f}"


def build_paper_table(
    complexity: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, model_row in complexity.iterrows():
        model_id = model_row["model_id"]
        batch1 = benchmark.loc[
            benchmark["model_id"].eq(model_id)
            & benchmark["batch_size"].eq(1)
        ].iloc[0]
        batch128 = benchmark.loc[
            benchmark["model_id"].eq(model_id)
            & benchmark["batch_size"].eq(128)
        ].iloc[0]
        rows.append(
            {
                "model": model_row["model_name"],
                "parameters_million": model_row["parameters_million"],
                "parameter_change_percent_vs_resnet18": model_row[
                    "parameter_change_percent_vs_baseline"
                ],
                "estimated_macs_million": model_row["estimated_macs_million"],
                "conv_linear_flops_million": model_row[
                    "conv_linear_flops_million"
                ],
                "state_dict_memory_mib": model_row["state_dict_memory_mib"],
                "mps_batch1_latency_ms_mean_sd": format_mean_sd(
                    batch1["batch_latency_ms_mean"],
                    batch1["batch_latency_ms_sample_std"],
                    3,
                ),
                "mps_batch1_samples_per_second_mean_sd": format_mean_sd(
                    batch1["samples_per_second_mean"],
                    batch1["samples_per_second_sample_std"],
                    1,
                ),
                "mps_batch128_samples_per_second_mean_sd": format_mean_sd(
                    batch128["samples_per_second_mean"],
                    batch128["samples_per_second_sample_std"],
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def annotate_bars(
    axis: plt.Axes,
    bars,
    decimals: int = 2,
    errors=None,
):
    heights = np.asarray([float(bar.get_height()) for bar in bars])
    error_values = (
        np.zeros_like(heights)
        if errors is None
        else np.asarray(errors, dtype=np.float64)
    )
    if error_values.shape != heights.shape:
        raise ValueError("Bar heights and error values must have the same shape.")
    maximum = float(np.max(heights + error_values))
    axis.set_ylim(0, maximum * 1.22 if maximum > 0 else 1)
    for bar, error in zip(bars, error_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + error + maximum * 0.025,
            f"{bar.get_height():.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_efficiency(
    complexity: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
):
    configure_plot_style()
    model_order = [config["model_id"] for config in MODEL_CONFIGS]
    labels = [config["display_name"] for config in MODEL_CONFIGS]
    colors = ["#7F8C8D", "#2A9D8F"]
    complexity_indexed = complexity.set_index("model_id").loc[model_order]

    batch1 = benchmark.loc[benchmark["batch_size"].eq(1)].set_index("model_id")
    batch1 = batch1.loc[model_order]
    batch128 = benchmark.loc[benchmark["batch_size"].eq(128)].set_index("model_id")
    batch128 = batch128.loc[model_order]

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.3))

    bars = axes[0, 0].bar(
        labels,
        complexity_indexed["parameters_million"],
        color=colors,
    )
    axes[0, 0].set_title("Model Parameters")
    axes[0, 0].set_ylabel("Parameters (million)")
    annotate_bars(axes[0, 0], bars, decimals=2)

    bars = axes[0, 1].bar(
        labels,
        complexity_indexed["conv_linear_flops_million"],
        color=colors,
    )
    axes[0, 1].set_title("Conv/Linear FLOPs")
    axes[0, 1].set_ylabel("FLOPs per sample (million)")
    annotate_bars(axes[0, 1], bars, decimals=1)

    bars = axes[1, 0].bar(
        labels,
        batch1["batch_latency_ms_mean"],
        yerr=batch1["batch_latency_ms_sample_std"],
        capsize=4,
        color=colors,
    )
    axes[1, 0].set_title("MPS Latency, Batch Size 1")
    axes[1, 0].set_ylabel("Milliseconds per sample")
    annotate_bars(
        axes[1, 0],
        bars,
        decimals=2,
        errors=batch1["batch_latency_ms_sample_std"].to_numpy(),
    )

    bars = axes[1, 1].bar(
        labels,
        batch128["samples_per_second_mean"],
        yerr=batch128["samples_per_second_sample_std"],
        capsize=4,
        color=colors,
    )
    axes[1, 1].set_title("MPS Throughput, Batch Size 128")
    axes[1, 1].set_ylabel("Samples per second")
    annotate_bars(
        axes[1, 1],
        bars,
        decimals=1,
        errors=batch128["samples_per_second_sample_std"].to_numpy(),
    )

    for axis in axes.flat:
        axis.grid(axis="y", linestyle="--", alpha=0.28)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=8)

    figure.suptitle(
        "Model Complexity and M1/MPS Forward-Pass Benchmark",
        fontsize=14,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Input: 1x1x64x64. Timing excludes data loading; error bars show "
            "+/- 1 sample SD across 3 rounds."
        ),
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=[0, 0.04, 1, 0.97])
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def hardware_info() -> dict:
    chip = ""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        chip = result.stdout.strip()
    except OSError:
        chip = ""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "chip": chip,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": "mps",
    }


def build_summary(
    complexity: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> dict:
    final_row = complexity.loc[
        complexity["model_id"].eq("highres_shufflenetv2_ce")
    ].iloc[0]
    baseline_row = complexity.loc[
        complexity["model_id"].eq("resnet18_baseline")
    ].iloc[0]
    final_batch1 = benchmark.loc[
        benchmark["model_id"].eq("highres_shufflenetv2_ce")
        & benchmark["batch_size"].eq(1)
    ].iloc[0]
    baseline_batch1 = benchmark.loc[
        benchmark["model_id"].eq("resnet18_baseline")
        & benchmark["batch_size"].eq(1)
    ].iloc[0]

    return {
        "analysis_id": ANALYSIS_ID,
        "analysis_type": "static_complexity_and_synthetic_mps_forward_benchmark",
        "dataset_read": False,
        "test_inference_performed": False,
        "input_shape": list(INPUT_SHAPE),
        "flop_definition": (
            "PyTorch FlopCounterMode convolution and linear FLOPs; multiply "
            "and add are counted as two FLOPs. MACs are reported as FLOPs/2."
        ),
        "timing_scope": "model forward pass only; data loading excluded",
        "benchmark_protocol": {
            "batch_sizes": list(BATCH_SIZES),
            "warmup_steps": WARMUP_STEPS,
            "measure_steps_per_round": MEASURE_STEPS,
            "rounds": ROUNDS,
            "aggregation": (
                "arithmetic mean and sample standard deviation of each "
                "per-round metric"
            ),
        },
        "hardware": hardware_info(),
        "final_vs_resnet18": {
            "parameter_reduction_percent": float(
                100
                * (
                    1
                    - final_row["parameter_count"]
                    / baseline_row["parameter_count"]
                )
            ),
            "flops_change_percent": float(
                100
                * (
                    final_row["conv_linear_flops"]
                    / baseline_row["conv_linear_flops"]
                    - 1
                )
            ),
            "batch1_latency_change_percent": float(
                100
                * (
                    final_batch1["batch_latency_ms_mean"]
                    / baseline_batch1["batch_latency_ms_mean"]
                    - 1
                )
            ),
        },
    }


def prepare_models() -> tuple[list[dict], dict[str, str], dict]:
    freeze_manifest = load_freeze_manifest()
    validate_frozen_environment(freeze_manifest)
    prepared = []
    input_hashes = {
        str(FREEZE_MANIFEST.relative_to(PROJECT_ROOT)): (
            EXPECTED_FREEZE_MANIFEST_SHA256
        )
    }
    input_hashes.update(validate_source_files(freeze_manifest))
    for config in MODEL_CONFIGS:
        checkpoint = validate_checkpoint(config, freeze_manifest)
        model = load_model(config, checkpoint)
        prepared.append({"config": config, "model": model})
        input_hashes[str(config["checkpoint"].relative_to(PROJECT_ROOT))] = config[
            "checkpoint_sha256"
        ]

    return prepared, input_hashes, freeze_manifest


def write_outputs(
    output_root: Path,
    complexity: pd.DataFrame,
    rounds: pd.DataFrame,
    benchmark: pd.DataFrame,
    input_hashes: dict[str, str],
):
    paths = output_paths(output_root)
    metrics_dir = paths["metrics_dir"]
    figure_dir = paths["figure_dir"]
    table_dir = paths["table_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    complexity.to_csv(
        metrics_dir / "model_complexity.csv", index=False, float_format="%.8f"
    )
    rounds.to_csv(
        metrics_dir / "mps_benchmark_rounds.csv", index=False, float_format="%.8f"
    )
    benchmark.to_csv(
        metrics_dir / "mps_benchmark_summary.csv", index=False, float_format="%.8f"
    )

    summary = build_summary(complexity, benchmark)
    (metrics_dir / "efficiency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paper_table = build_paper_table(complexity, benchmark)
    paper_table.to_csv(
        table_dir / PAPER_TABLE_NAME,
        index=False,
        float_format="%.4f",
    )

    output_png = figure_dir / f"{FIGURE_STEM}.png"
    output_pdf = figure_dir / f"{FIGURE_STEM}.pdf"
    plot_efficiency(complexity, benchmark, output_png, output_pdf)

    output_hashes = {}
    for path in expected_output_files(output_root):
        if path.name == "analysis_manifest.json":
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Expected output was not created: {path}")
        output_hashes[str(path.relative_to(output_root))] = sha256_file(path)

    manifest = {
        "analysis_id": ANALYSIS_ID,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_read": False,
        "test_inference_performed": False,
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
    }
    (metrics_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_complexity(complexity: pd.DataFrame):
    print("Input validation: PASS")
    print("No dataset was read and no test inference was performed.")
    for _, row in complexity.iterrows():
        print(
            f"{row['model_name']}: {int(row['parameter_count']):,} params, "
            f"{row['conv_linear_flops_million']:.1f}M FLOPs, "
            f"{row['estimated_macs_million']:.1f}M estimated MACs"
        )


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.run:
        ensure_outputs_are_available(output_root)

    prepared, input_hashes, _ = prepare_models()
    complexity_rows = [
        calculate_complexity(item["config"], item["model"])
        for item in prepared
    ]
    complexity = add_relative_complexity(pd.DataFrame(complexity_rows))
    print_complexity(complexity)

    if args.check_inputs:
        print("Preflight complete. No benchmark was run and no files were created.")
        return

    device = select_mps_device()
    print(f"Benchmark device: {device}")
    round_rows = []
    for item in prepared:
        print(f"Benchmarking {item['config']['display_name']}...")
        round_rows.extend(
            benchmark_model(item["config"], item["model"], device)
        )
    rounds = pd.DataFrame(round_rows)
    benchmark = aggregate_benchmark(rounds)

    for _, row in benchmark.iterrows():
        print(
            f"{row['model_name']} batch={int(row['batch_size'])}: "
            f"{row['batch_latency_ms_mean']:.3f} ms/batch, "
            f"{row['samples_per_second_mean']:.1f} samples/s"
        )

    write_outputs(
        output_root,
        complexity,
        rounds,
        benchmark,
        input_hashes,
    )
    paths = output_paths(output_root)
    print(f"Metrics saved to: {paths['metrics_dir']}")
    print(f"Figure saved to: {paths['figure_dir'] / (FIGURE_STEM + '.png')}")
    print(f"Paper table saved to: {paths['table_dir'] / PAPER_TABLE_NAME}")


if __name__ == "__main__":
    main()
