"""Verify PyTorch MPS availability, correctness, training, and basic speed."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = PROJECT_ROOT / "04_实验" / "benchmarks" / "mps_environment_check.txt"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def save(self) -> None:
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_matmul(device: torch.device, size: int, repeats: int) -> float:
    generator = torch.Generator(device="cpu").manual_seed(20260724 + size)
    left_cpu = torch.randn(size, size, generator=generator)
    right_cpu = torch.randn(size, size, generator=generator)
    left = left_cpu.to(device)
    right = right_cpu.to(device)

    for _ in range(3):
        _ = left @ right
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = left @ right
    synchronize(device)
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> int:
    report = Report()
    passed = False
    try:
        report.add("=== PyTorch MPS environment check ===")
        report.add(f"macOS: {platform.mac_ver()[0]}")
        report.add(f"machine: {platform.machine()}")
        report.add(f"PyTorch: {torch.__version__}")
        report.add(
            "PYTORCH_ENABLE_MPS_FALLBACK: "
            + os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "<unset>")
        )

        is_built = torch.backends.mps.is_built()
        is_available = torch.backends.mps.is_available()
        report.add(f"MPS built: {is_built}")
        report.add(f"MPS available: {is_available}")
        if not is_built or not is_available:
            raise RuntimeError("MPS is not available in this PyTorch/macOS setup")

        cpu = torch.device("cpu")
        mps = torch.device("mps")

        report.add("\n[1/4] Device and matrix multiplication")
        device_tensor = torch.randn(1024, 1024, device=mps)
        product = device_tensor @ device_tensor.T
        synchronize(mps)
        report.add(f"tensor device: {device_tensor.device}")
        report.add(f"matrix result finite: {torch.isfinite(product).all().item()}")

        report.add("\n[2/4] CPU/MPS numerical agreement")
        torch.manual_seed(42)
        left = torch.randn(256, 256)
        right = torch.randn(256, 256)
        expected = left @ right
        actual = (left.to(mps) @ right.to(mps)).cpu()
        max_abs_error = (expected - actual).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        report.add(f"maximum absolute error: {max_abs_error:.8f}")
        report.add("numerical comparison: PASS")

        report.add("\n[3/4] Forward, backward, and optimizer step")
        torch.manual_seed(7)
        model = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 9),
        ).to(mps)
        inputs = torch.randn(16, 2, 64, 64, device=mps)
        targets = torch.randint(0, 9, (16,), device=mps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        before = model[0].weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        optimizer.step()
        synchronize(mps)
        parameter_change = (model[0].weight.detach() - before).abs().max().item()
        report.add(f"model output device: {logits.device}")
        report.add(f"loss: {loss.item():.6f}")
        report.add(f"gradients finite: {gradients_finite}")
        report.add(f"maximum parameter change: {parameter_change:.8f}")
        if not gradients_finite or parameter_change <= 0:
            raise RuntimeError("Training step did not produce a valid parameter update")
        report.add("training step: PASS")

        report.add("\n[4/4] Basic matrix multiplication timing")
        report.add("Times are diagnostics only, not paper benchmark results.")
        for size, repeats in ((256, 10), (1024, 5), (2048, 3)):
            cpu_ms = benchmark_matmul(cpu, size, repeats)
            mps_ms = benchmark_matmul(mps, size, repeats)
            report.add(
                f"{size}x{size}: CPU {cpu_ms:.3f} ms | "
                f"MPS {mps_ms:.3f} ms | speedup {cpu_ms / mps_ms:.2f}x"
            )

        if hasattr(torch.mps, "current_allocated_memory"):
            allocated_mb = torch.mps.current_allocated_memory() / (1024**2)
            report.add(f"MPS allocated memory after test: {allocated_mb:.2f} MiB")

        passed = True
        report.add("\nOVERALL RESULT: PASS")
        return 0
    except Exception as error:
        report.add(f"\nOVERALL RESULT: FAIL ({type(error).__name__}: {error})")
        return 1
    finally:
        report.save()
        print(f"Result saved to: {RESULT_PATH}", flush=True)
        if not passed:
            print("MPS verification did not pass; review the saved report.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
