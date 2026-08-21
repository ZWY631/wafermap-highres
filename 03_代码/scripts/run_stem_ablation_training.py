#!/usr/bin/env python3
"""Preflight or sequentially execute the six planned stem runs."""

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
SCRIPTS_DIR = CODE_DIR / "scripts"
SRC_DIR = CODE_DIR / "src"
PROJECT_ROOT = CODE_DIR.parent
PROTOCOL_FILE = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260805_stem_2x2_补充实验冻结协议.md"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import (
    SUPPORTED_SEEDS,
    build_run_paths,
    run_name_for_seed,
)


@dataclass(frozen=True)
class PlannedConfiguration:
    configuration_id: str
    script_name: str
    base_run_name: str


CONFIGURATIONS = (
    PlannedConfiguration(
        configuration_id="s2_nopool",
        script_name="train_shufflenet_stem_s2_nopool_ce_multiseed.py",
        base_run_name="shufflenet_v2_stem_s2_nopool_ce_full",
    ),
    PlannedConfiguration(
        configuration_id="s1_pool",
        script_name="train_shufflenet_stem_s1_pool_ce_multiseed.py",
        base_run_name="shufflenet_v2_stem_s1_pool_ce_full",
    ),
)

FROZEN_SOURCE_SHA256 = {
    "03_代码/src/wafermap/stem_ablation_models.py": (
        "230e637b19f8dd94be56d498e4d7c097972c3e0006129bcfea9661333b0ee073"
    ),
    "03_代码/scripts/train_shufflenet_stem_s2_nopool_ce_multiseed.py": (
        "158bd202399f86f282896aceab46cd9dfdb8873376df556148f0e0ace8386a84"
    ),
    "03_代码/scripts/train_shufflenet_stem_s1_pool_ce_multiseed.py": (
        "d7af2e5d2384b92b6d6af0075eacb59e7538cd682dc4b9d0af67cc964c10e052"
    ),
    "03_代码/src/wafermap/controlled_ce_training.py": (
        "ccc5ae077f6b5808f4f77dd93b11a0a9ad4e3a29760a76caa91c07a09cf7c82f"
    ),
    "03_代码/src/wafermap/transforms.py": (
        "2c5dd9622cbf55600522dc554fa611cb7c1cd27ac167641da321ac1838472459"
    ),
    "03_代码/src/wafermap/dataset.py": (
        "79a264db96f05fc2d108ea69c421101c06e5ec127c33b415aa373033e40442ce"
    ),
    "03_代码/src/wafermap/constants.py": (
        "d2c4f4055a6bdae3bd0130859abbefcc20f267746fd2275f78b367bb8dd4c552"
    ),
    "03_代码/src/wafermap/paths.py": (
        "f0e85b19fb2cb685b70c7f90d3a94ebc7ba80235aa5e267b3b4851dad49800c0"
    ),
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the frozen six-run WM-811K stem ablation plan."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start training. Without this flag, only preflight is run.",
    )
    return parser.parse_args(argv)


def planned_commands(python_executable: str = sys.executable):
    commands = []
    for configuration in CONFIGURATIONS:
        script = SCRIPTS_DIR / configuration.script_name
        for seed in SUPPORTED_SEEDS:
            commands.append(
                (
                    configuration,
                    seed,
                    [python_executable, str(script), "--seed", str(seed)],
                )
            )
    return commands


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
            raise FileNotFoundError(f"Missing frozen source file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen source hash mismatch for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )
    if not PROTOCOL_FILE.is_file():
        raise FileNotFoundError(f"Missing frozen protocol: {PROTOCOL_FILE}")


def classify_run_state(configuration: PlannedConfiguration, seed: int) -> str:
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    paths = build_run_paths(run_name)
    artifacts = (
        paths.best_checkpoint,
        paths.last_checkpoint,
        paths.metrics_file,
        paths.log_file,
    )
    manifest_file = paths.checkpoint_dir / "run_manifest.json"
    expected_files = artifacts + (manifest_file,)
    present = [path.exists() for path in expected_files]
    if all(present):
        validate_complete_run_manifest(
            configuration,
            seed,
            manifest_file,
            artifacts,
        )
        return "complete"
    if any(present) or paths.checkpoint_dir.exists():
        return "partial"
    return "missing"


def validate_complete_run_manifest(
    configuration: PlannedConfiguration,
    seed: int,
    manifest_file: Path,
    artifacts: Sequence[Path],
):
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    if payload.get("configuration_id") != configuration.configuration_id:
        raise ValueError(f"Configuration mismatch in {manifest_file}")
    if payload.get("run_name") != run_name:
        raise ValueError(f"Run name mismatch in {manifest_file}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {manifest_file}")

    recorded_hashes = payload.get("artifact_sha256", {})
    for path in artifacts:
        relative_path = str(path.relative_to(PROJECT_ROOT))
        expected_hash = recorded_hashes.get(relative_path)
        if expected_hash is None:
            raise ValueError(
                f"Artifact hash missing from {manifest_file}: {relative_path}"
            )
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {path}")


def finalize_run_manifest(
    configuration: PlannedConfiguration,
    seed: int,
    command: Sequence[str],
):
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    paths = build_run_paths(run_name)
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
    epochs = tuple(int(row["epoch"]) for row in rows)
    if epochs != tuple(range(1, 31)):
        raise ValueError(
            f"History is not a complete 30-epoch run: {paths.metrics_file}"
        )

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
        "protocol_id": "20260805_stem_2x2_ablation",
        "configuration_id": configuration.configuration_id,
        "run_name": run_name,
        "seed": seed,
        "command": list(command),
        "epochs": 30,
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


def preflight():
    validate_frozen_sources()
    plan = planned_commands()
    states = []
    for configuration, seed, command in plan:
        script = Path(command[1])
        if not script.is_file():
            raise FileNotFoundError(f"Missing training script: {script}")
        state = classify_run_state(configuration, seed)
        states.append((configuration, seed, command, state))
        print(
            f"configuration={configuration.configuration_id} "
            f"seed={seed} state={state}"
        )

    partial = [item for item in states if item[3] == "partial"]
    if partial:
        names = ", ".join(
            f"{item[0].configuration_id}/seed{item[1]}" for item in partial
        )
        raise RuntimeError(
            "Partial runs require manual audit before continuing: " + names
        )

    missing = [item for item in states if item[3] == "missing"]
    print(f"planned_runs={len(states)}")
    print(f"complete_runs={len(states) - len(missing)}")
    print(f"missing_runs={len(missing)}")
    return missing


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    missing = preflight()

    if not args.execute:
        print("Preflight passed; no training was started.")
        return

    if not missing:
        print("All six planned runs are already complete.")
        return

    for index, (configuration, seed, command, _) in enumerate(missing, start=1):
        print(
            f"Starting run {index}/{len(missing)}: "
            f"{configuration.configuration_id}, seed={seed}"
        )
        subprocess.run(command, check=True)
        finalize_run_manifest(configuration, seed, command)

    print("All missing stem ablation runs completed successfully.")


if __name__ == "__main__":
    main()
