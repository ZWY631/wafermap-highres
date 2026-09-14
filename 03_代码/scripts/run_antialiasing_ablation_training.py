#!/usr/bin/env python3
"""Run the anti-aliasing (blur-pool) stem ablation: S2B and S1B, 3 seeds each.

Why this experiment exists
--------------------------
The frozen 2x2 ablation (S2P / S2N / S1P / S1N) shows both stem factors matter,
but it cannot say whether the gain comes from **keeping more spatial samples**
or from **not aliasing the samples that are kept**. Those two explanations are
distinguishable, and this runner produces the runs that distinguish them:

  S2P  stride 2 + maxpool     16 x 16 grid   (frozen, 1,262,397 params)
  S2B  stride 2 + blurpool    16 x 16 grid   (NEW,    1,262,397 params)
  S1N  stride 1 + identity    64 x 64 grid   (frozen, 1,262,397 params)

S2B is resolution-matched to S2P and aliasing-matched to S1N.

Protocol identity guarantee
---------------------------
The data pipeline, transforms, dataset, splits, optimizer, schedule, epoch
budget, and checkpoint rule are the *frozen* ones. This runner verifies the
SHA-256 of every shared source file against the hashes recorded for the
original ablation before it starts, so a run that completes here is directly
comparable to the already-published S2P/S2N/S1P/S1N numbers. If any shared
file has drifted, the run is refused rather than silently producing an
incomparable result.

Usage
-----
    # integrity + path check only, no training
    python scripts/run_antialiasing_ablation_training.py

    # actually train all six runs (~3.5 h per run on M1 Pro/MPS, ~21 h total)
    python scripts/run_antialiasing_ablation_training.py --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import ast
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

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.controlled_ce_training import (  # noqa: E402
    SUPPORTED_SEEDS,
    build_run_paths,
    run_name_for_seed,
)

PROTOCOL_ID = "20260805_stem_2x2_ablation+antialiasing_extension"


def _frozen_source_hashes() -> dict:
    """Read FROZEN_SOURCE_SHA256 from the original runner without executing it.

    The original runner is a script, not an importable module, so executing it
    would drag in its argparse setup and side effects. Parsing the literal
    assignment with ``ast`` keeps the pinned hashes as the single source of
    truth while staying side-effect free. If the frozen runner ever drops the
    constant, this raises rather than silently skipping the integrity check.
    """
    path = SCRIPTS_DIR / "run_stem_ablation_training.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen runner: {path}")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "FROZEN_SOURCE_SHA256"
            ):
                hashes = ast.literal_eval(node.value)
                if not isinstance(hashes, dict) or not hashes:
                    raise ValueError(
                        f"FROZEN_SOURCE_SHA256 in {path} is not a non-empty dict"
                    )
                return hashes
    raise ValueError(
        f"FROZEN_SOURCE_SHA256 not found in {path}; refusing to run without "
        "a protocol-identity check."
    )


@dataclass(frozen=True)
class PlannedConfiguration:
    configuration_id: str
    script_name: str
    base_run_name: str
    stem_grid: str
    role: str


CONFIGURATIONS = (
    PlannedConfiguration(
        configuration_id="s2_blurpool",
        script_name="train_shufflenet_stem_s2_blurpool_ce_multiseed.py",
        base_run_name="shufflenet_v2_stem_s2_blurpool_ce_full",
        stem_grid="16x16",
        role="resolution-matched anti-aliasing control for S2P",
    ),
    PlannedConfiguration(
        configuration_id="s1_blurpool",
        script_name="train_shufflenet_stem_s1_blurpool_ce_multiseed.py",
        base_run_name="shufflenet_v2_stem_s1_blurpool_ce_full",
        stem_grid="32x32",
        role="intermediate-resolution anti-aliasing control for S2N/S1P",
    ),
)

# The new model module is deliberately not in the frozen list: it is an
# addition, and the frozen files must remain byte-identical for the original
# runs to keep their audit trail.
EXTENSION_SOURCES = ("03_代码/src/wafermap/stem_antialiasing_models.py",)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the blur-pool (anti-aliasing) stem ablation."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start training. Without this flag, only preflight is run.",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        default=[c.configuration_id for c in CONFIGURATIONS],
        choices=[c.configuration_id for c in CONFIGURATIONS],
        help="Subset of configurations to run (default: all).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SUPPORTED_SEEDS),
        choices=list(SUPPORTED_SEEDS),
        help="Subset of seeds to run (default: all three).",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources() -> None:
    """Refuse to run if any shared pipeline file differs from the frozen hash."""
    frozen = _frozen_source_hashes()
    for relative_path, expected_hash in frozen.items():
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen source file: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                "Frozen source hash mismatch, refusing to run an "
                f"incomparable experiment.\n  file:     {relative_path}\n"
                f"  expected: {expected_hash}\n  actual:   {actual_hash}"
            )
    print(f"frozen_sources_verified={len(frozen)}")


def validate_extension_sources() -> None:
    for relative_path in EXTENSION_SOURCES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing extension source: {path}")
    print(f"extension_sources_present={len(EXTENSION_SOURCES)}")


def planned_commands(
    python_executable: str,
    configurations: Sequence[str],
    seeds: Sequence[int],
):
    commands = []
    for configuration in CONFIGURATIONS:
        if configuration.configuration_id not in configurations:
            continue
        script = SCRIPTS_DIR / configuration.script_name
        for seed in seeds:
            commands.append(
                (
                    configuration,
                    seed,
                    [python_executable, str(script), "--seed", str(seed)],
                )
            )
    return commands


def classify_run_state(configuration: PlannedConfiguration, seed: int) -> str:
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    paths = build_run_paths(run_name)
    artifacts = (
        paths.best_checkpoint,
        paths.last_checkpoint,
        paths.metrics_file,
        paths.log_file,
    )
    present = [path.exists() for path in artifacts]
    if all(present):
        return "complete"
    if any(present) or paths.checkpoint_dir.exists():
        return "partial"
    return "missing"


def finalize_run_manifest(
    configuration: PlannedConfiguration,
    seed: int,
    command: Sequence[str],
) -> None:
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

    frozen = _frozen_source_hashes()
    source_hashes = {
        relative_path: sha256_file(PROJECT_ROOT / relative_path)
        for relative_path in frozen
    }
    for relative_path in EXTENSION_SOURCES:
        source_hashes[relative_path] = sha256_file(PROJECT_ROOT / relative_path)
    source_hashes[
        str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    ] = sha256_file(Path(__file__).resolve())

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": PROTOCOL_ID,
        "configuration_id": configuration.configuration_id,
        "configuration_role": configuration.role,
        "stem_grid": configuration.stem_grid,
        "run_name": run_name,
        "seed": seed,
        "command": list(command),
        "epochs": 30,
        "parameter_count": 1_262_397,
        "shared_protocol_with_frozen_ablation": True,
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


def preflight(configurations: Sequence[str], seeds: Sequence[int]):
    validate_frozen_sources()
    validate_extension_sources()
    plan = planned_commands(sys.executable, configurations, seeds)

    states = []
    for configuration, seed, command in plan:
        script = Path(command[1])
        if not script.is_file():
            raise FileNotFoundError(f"Missing training script: {script}")
        state = classify_run_state(configuration, seed)
        states.append((configuration, seed, command, state))
        print(
            f"configuration={configuration.configuration_id} "
            f"seed={seed} stem_grid={configuration.stem_grid} state={state}"
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
    missing = preflight(args.configurations, args.seeds)

    if not args.execute:
        print("Preflight passed; no training was started.")
        return

    if not missing:
        print("All planned anti-aliasing runs are already complete.")
        return

    for index, (configuration, seed, command, _) in enumerate(missing, start=1):
        print(
            f"Starting run {index}/{len(missing)}: "
            f"{configuration.configuration_id}, seed={seed}"
        )
        subprocess.run(command, check=True)
        finalize_run_manifest(configuration, seed, command)

    print("All missing anti-aliasing stem runs completed successfully.")


if __name__ == "__main__":
    main()
