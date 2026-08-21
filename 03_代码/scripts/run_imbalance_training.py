#!/usr/bin/env python3
"""Preflight or sequentially execute the six frozen imbalance runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
PROJECT_ROOT = CODE_DIR.parent
SCRIPTS_DIR = CODE_DIR / "scripts"
PROTOCOL_FILE = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260805_类别不平衡消融冻结协议.md"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import build_run_paths
from wafermap.imbalance_training import (
    BASE_RUN_NAMES,
    SUPPORTED_SEEDS,
    SUPPORTED_STRATEGIES,
    run_name_for_seed,
    validate_selected_architecture,
)


@dataclass(frozen=True)
class PlannedStrategy:
    strategy: str
    display_name: str


PLANNED_STRATEGIES = (
    PlannedStrategy("weighted_ce", "inverse-frequency weighted CE"),
    PlannedStrategy("balanced_sampler", "weighted random sampler"),
)

FROZEN_SOURCE_SHA256 = {
    "03_代码/src/wafermap/imbalance_training.py": (
        "3d0e5333d6196ebac647439dfeb1fdc92bc788d26a85511548c4c55c86a9106b"
    ),
    "03_代码/scripts/train_shufflenet_highres_imbalance.py": (
        "7f20d6b14f3a7f05c3406d35803eed84e2781a2cf397a4aa820daa2e8d7b1546"
    ),
    "03_代码/src/wafermap/models_improved.py": (
        "9f42de30839646d2ae7c5b82190d97b29dd6fb54cd47f8c574c4bfd1fadde392"
    ),
    "03_代码/src/wafermap/controlled_ce_training.py": (
        "ccc5ae077f6b5808f4f77dd93b11a0a9ad4e3a29760a76caa91c07a09cf7c82f"
    ),
    "03_代码/src/wafermap/dataset.py": (
        "79a264db96f05fc2d108ea69c421101c06e5ec127c33b415aa373033e40442ce"
    ),
    "03_代码/src/wafermap/transforms.py": (
        "2c5dd9622cbf55600522dc554fa611cb7c1cd27ac167641da321ac1838472459"
    ),
    "03_代码/src/wafermap/constants.py": (
        "d2c4f4055a6bdae3bd0130859abbefcc20f267746fd2275f78b367bb8dd4c552"
    ),
    "03_代码/src/wafermap/paths.py": (
        "f0e85b19fb2cb685b70c7f90d3a94ebc7ba80235aa5e267b3b4851dad49800c0"
    ),
    "00_项目管理/20260805_stem_2x2_架构选择冻结清单.json": (
        "26e9f93dfdae615fbdf91da612cb877637cb13b18e6572b934b9d3690d705ead"
    ),
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the frozen six-run WM-811K imbalance plan."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start training; without it, only preflight runs.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        default=list(SUPPORTED_STRATEGIES),
        help="Subset of strategies to run in this session.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SUPPORTED_SEEDS,
        default=list(SUPPORTED_SEEDS),
        help="Subset of seeds to run in this session.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources():
    for relative_path, expected_hash in FROZEN_SOURCE_SHA256.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen source: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen source hash mismatch for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError(f"Missing imbalance protocol: {PROTOCOL_FILE}")
    validate_selected_architecture()


def planned_commands(
    python_executable: str = sys.executable,
    strategies: Sequence[str] = SUPPORTED_STRATEGIES,
    seeds: Sequence[int] = SUPPORTED_SEEDS,
):
    script = SCRIPTS_DIR / "train_shufflenet_highres_imbalance.py"
    commands = []
    for planned in PLANNED_STRATEGIES:
        if planned.strategy not in strategies:
            continue
        for seed in seeds:
            commands.append(
                (
                    planned,
                    seed,
                    [
                        python_executable,
                        str(script),
                        "--strategy",
                        planned.strategy,
                        "--seed",
                        str(seed),
                    ],
                )
            )
    return commands


def run_paths(strategy: str, seed: int):
    return build_run_paths(
        run_name_for_seed(strategy, seed),
    )


def validate_complete_run_manifest(
    planned: PlannedStrategy,
    seed: int,
    manifest_file: Path,
    artifacts: Sequence[Path],
):
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    run_name = run_name_for_seed(planned.strategy, seed)
    if payload.get("strategy") != planned.strategy:
        raise ValueError(f"Strategy mismatch in {manifest_file}")
    if payload.get("run_name") != run_name:
        raise ValueError(f"Run name mismatch in {manifest_file}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {manifest_file}")
    recorded = payload.get("artifact_sha256", {})
    for path in artifacts:
        relative_path = str(path.relative_to(PROJECT_ROOT))
        expected_hash = recorded.get(relative_path)
        if expected_hash is None:
            raise ValueError(f"Artifact hash missing: {relative_path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {path}")


def classify_run_state(planned: PlannedStrategy, seed: int) -> str:
    paths = run_paths(planned.strategy, seed)
    artifacts = (
        paths.best_checkpoint,
        paths.last_checkpoint,
        paths.metrics_file,
        paths.log_file,
    )
    manifest = paths.checkpoint_dir / "run_manifest.json"
    expected = artifacts + (manifest,)
    present = [path.exists() for path in expected]
    if all(present):
        validate_complete_run_manifest(planned, seed, manifest, artifacts)
        return "complete"
    if any(present) or paths.checkpoint_dir.exists():
        return "partial"
    return "missing"


def finalize_run_manifest(
    planned: PlannedStrategy,
    seed: int,
    command: Sequence[str],
):
    paths = run_paths(planned.strategy, seed)
    manifest_file = paths.checkpoint_dir / "run_manifest.json"
    if manifest_file.exists():
        raise FileExistsError(f"Run manifest already exists: {manifest_file}")
    artifacts = (
        paths.best_checkpoint,
        paths.last_checkpoint,
        paths.metrics_file,
        paths.log_file,
    )
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Training returned without all required artifacts:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    with paths.metrics_file.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if tuple(int(row["epoch"]) for row in rows) != tuple(range(1, 31)):
        raise ValueError(f"History is not a complete 30-epoch run: {paths.metrics_file}")
    source_hashes = {
        relative_path: sha256_file(PROJECT_ROOT / relative_path)
        for relative_path in FROZEN_SOURCE_SHA256
    }
    source_hashes[str(Path(__file__).resolve().relative_to(PROJECT_ROOT))] = (
        sha256_file(Path(__file__).resolve())
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": "20260805_class_imbalance_ablation",
        "strategy": planned.strategy,
        "run_name": run_name_for_seed(planned.strategy, seed),
        "seed": seed,
        "command": list(command),
        "selected_architecture": "S1N",
        "architecture_freeze_sha256": FROZEN_SOURCE_SHA256[
            "00_项目管理/20260805_stem_2x2_架构选择冻结清单.json"
        ],
        "protocol_file": str(PROTOCOL_FILE.relative_to(PROJECT_ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "source_sha256": source_hashes,
        "artifact_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in artifacts
        },
    }
    manifest_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def preflight(
    strategies: Sequence[str] = SUPPORTED_STRATEGIES,
    seeds: Sequence[int] = SUPPORTED_SEEDS,
):
    validate_frozen_sources()
    plan = planned_commands(strategies=strategies, seeds=seeds)
    if not plan:
        raise ValueError("At least one strategy and one seed are required.")
    states = []
    for planned, seed, command in plan:
        state = classify_run_state(planned, seed)
        states.append((planned, seed, command, state))
        print(
            f"strategy={planned.strategy} seed={seed} state={state}"
        )
    partial = [item for item in states if item[3] == "partial"]
    if partial:
        names = ", ".join(
            f"{item[0].strategy}/seed{item[1]}" for item in partial
        )
        raise RuntimeError("Partial imbalance runs require audit: " + names)
    missing = [item for item in states if item[3] == "missing"]
    print(f"planned_runs={len(states)}")
    print(f"complete_runs={len(states) - len(missing)}")
    print(f"missing_runs={len(missing)}")
    return missing


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    missing = preflight(
        strategies=args.strategies,
        seeds=args.seeds,
    )
    if not args.execute:
        print("Preflight passed; no training was started.")
        return
    if not missing:
        print("All six imbalance runs are already complete.")
        return
    for index, (planned, seed, command, _) in enumerate(missing, start=1):
        print(
            f"Starting imbalance run {index}/{len(missing)}: "
            f"{planned.strategy}, seed={seed}"
        )
        subprocess.run(command, check=True)
        finalize_run_manifest(planned, seed, command)
    print("All missing imbalance runs completed successfully.")


if __name__ == "__main__":
    main()
