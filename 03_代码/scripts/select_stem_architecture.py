#!/usr/bin/env python3
"""Select a stem using only the frozen three-seed validation histories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "04_实验" / "metrics"
OUTPUT_DIR = METRICS_DIR / "20260805_stem_2x2_validation_selection"
FREEZE_FILE = (
    PROJECT_ROOT
    / "00_项目管理"
    / "20260805_stem_2x2_架构选择冻结清单.json"
)

SEEDS = (42, 123, 2026)
EXPECTED_EPOCHS = tuple(range(1, 31))
PRACTICAL_EQUIVALENCE_MARGIN = 0.005
ONE_SIDED_T_CRITICAL_DF2_95 = 2.91998558
NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class StemConfiguration:
    configuration_id: str
    display_name: str
    conv1_stride: int
    keep_max_pool: bool
    flops: int
    parameter_count: int
    base_run_name: str
    fixed_rank: int


CONFIGURATIONS = (
    StemConfiguration(
        configuration_id="S2P",
        display_name="stride 2 + max pool",
        conv1_stride=2,
        keep_max_pool=True,
        flops=22_624_960,
        parameter_count=1_262_397,
        base_run_name="shufflenet_v2_standard_ce_full",
        fixed_rank=0,
    ),
    StemConfiguration(
        configuration_id="S2N",
        display_name="stride 2 + no pool",
        conv1_stride=2,
        keep_max_pool=False,
        flops=89_117_440,
        parameter_count=1_262_397,
        base_run_name="shufflenet_v2_stem_s2_nopool_ce_full",
        fixed_rank=1,
    ),
    StemConfiguration(
        configuration_id="S1P",
        display_name="stride 1 + max pool",
        conv1_stride=1,
        keep_max_pool=True,
        flops=90_444_544,
        parameter_count=1_262_397,
        base_run_name="shufflenet_v2_stem_s1_pool_ce_full",
        fixed_rank=2,
    ),
    StemConfiguration(
        configuration_id="S1N",
        display_name="stride 1 + no pool",
        conv1_stride=1,
        keep_max_pool=False,
        flops=356_414_464,
        parameter_count=1_262_397,
        base_run_name="shufflenet_v2_highres_ce_full",
        fixed_rank=3,
    ),
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or freeze the validation-only 2x2 stem decision."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate all histories and planned outputs without writing.",
    )
    mode.add_argument(
        "--select",
        action="store_true",
        help="Apply the frozen rule and write non-overwriting outputs.",
    )
    return parser.parse_args(argv)


def run_name_for_seed(base_run_name: str, seed: int) -> str:
    if seed == 42:
        return base_run_name
    return f"{base_run_name}_seed{seed}"


def history_path(configuration: StemConfiguration, seed: int) -> Path:
    run_name = run_name_for_seed(configuration.base_run_name, seed)
    return METRICS_DIR / f"{run_name}_history.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_best_validation_row(path: Path) -> dict:
    if "test" in path.name.lower() or "prediction" in path.name.lower():
        raise ValueError(f"Test-derived input is forbidden: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing history file: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != len(EXPECTED_EPOCHS):
        raise ValueError(
            f"Expected 30 epochs in {path}, found {len(rows)}"
        )
    epochs = tuple(int(row["epoch"]) for row in rows)
    if epochs != EXPECTED_EPOCHS:
        raise ValueError(f"Epoch sequence is incomplete in {path}")

    required = (
        "val_macro_f1",
        "val_accuracy",
        "val_balanced_accuracy",
    )
    for row in rows:
        for field in required:
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Invalid {field} in {path}: {value}")

    return max(rows, key=lambda row: float(row["val_macro_f1"]))


def load_validation_scores():
    records = []
    scores = {}
    input_hashes = {}
    for configuration in CONFIGURATIONS:
        configuration_scores = {}
        for seed in SEEDS:
            path = history_path(configuration, seed)
            best = read_best_validation_row(path)
            score = float(best["val_macro_f1"])
            configuration_scores[seed] = score
            relative_path = str(path.relative_to(PROJECT_ROOT))
            input_hashes[relative_path] = sha256_file(path)
            records.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "display_name": configuration.display_name,
                    "seed": seed,
                    "run_name": run_name_for_seed(
                        configuration.base_run_name,
                        seed,
                    ),
                    "best_epoch": int(best["epoch"]),
                    "val_accuracy": float(best["val_accuracy"]),
                    "val_macro_f1": score,
                    "val_balanced_accuracy": float(
                        best["val_balanced_accuracy"]
                    ),
                    "history_path": relative_path,
                    "history_sha256": input_hashes[relative_path],
                }
            )
        scores[configuration.configuration_id] = configuration_scores
    return records, scores, input_hashes


def sample_standard_deviation(values: Sequence[float]) -> float:
    if len(values) < 2 or max(values) - min(values) <= NUMERICAL_TOLERANCE:
        return 0.0
    return statistics.stdev(values)


def choose_architecture(
    scores: Mapping[str, Mapping[int, float]],
):
    configurations = {
        item.configuration_id: item for item in CONFIGURATIONS
    }
    expected_ids = set(configurations)
    if set(scores) != expected_ids:
        raise ValueError("Scores must contain all four frozen configurations.")

    aggregates = {}
    for configuration_id, seed_scores in scores.items():
        if set(seed_scores) != set(SEEDS):
            raise ValueError(
                f"Scores for {configuration_id} must contain all three seeds."
            )
        values = [float(seed_scores[seed]) for seed in SEEDS]
        aggregates[configuration_id] = {
            "mean": statistics.mean(values),
            "sample_std": sample_standard_deviation(values),
        }

    highest_mean = max(item["mean"] for item in aggregates.values())
    leader_candidates = [
        configuration_id
        for configuration_id, item in aggregates.items()
        if highest_mean - item["mean"] <= NUMERICAL_TOLERANCE
    ]
    leader = min(
        leader_candidates,
        key=lambda configuration_id: (
            configurations[configuration_id].flops,
            aggregates[configuration_id]["sample_std"],
            configurations[configuration_id].fixed_rank,
        ),
    )

    comparisons = {}
    eligible = []
    for configuration_id in configurations:
        differences = [
            scores[leader][seed] - scores[configuration_id][seed]
            for seed in SEEDS
        ]
        difference_mean = statistics.mean(differences)
        difference_std = sample_standard_deviation(differences)
        upper_confidence_bound = difference_mean + (
            ONE_SIDED_T_CRITICAL_DF2_95
            * difference_std
            / math.sqrt(len(SEEDS))
        )
        is_equivalent = (
            upper_confidence_bound
            <= PRACTICAL_EQUIVALENCE_MARGIN + NUMERICAL_TOLERANCE
        )
        comparisons[configuration_id] = {
            "paired_differences_by_seed": {
                str(seed): differences[index]
                for index, seed in enumerate(SEEDS)
            },
            "paired_difference_mean": difference_mean,
            "paired_difference_sample_std": difference_std,
            "one_sided_95_percent_upper_bound": upper_confidence_bound,
            "practically_equivalent": is_equivalent,
        }
        if is_equivalent:
            eligible.append(configuration_id)

    winner = min(
        eligible,
        key=lambda configuration_id: (
            configurations[configuration_id].flops,
            -aggregates[configuration_id]["mean"],
            aggregates[configuration_id]["sample_std"],
            configurations[configuration_id].fixed_rank,
        ),
    )
    return leader, winner, aggregates, comparisons, eligible


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_payload(records, scores, input_hashes):
    leader, winner, aggregates, comparisons, eligible = choose_architecture(
        scores
    )
    aggregate_rows = []
    for configuration in CONFIGURATIONS:
        configuration_id = configuration.configuration_id
        aggregate_rows.append(
            {
                "configuration_id": configuration_id,
                "display_name": configuration.display_name,
                "conv1_stride": configuration.conv1_stride,
                "keep_max_pool": configuration.keep_max_pool,
                "parameter_count": configuration.parameter_count,
                "conv_linear_flops": configuration.flops,
                "validation_macro_f1_mean": aggregates[configuration_id][
                    "mean"
                ],
                "validation_macro_f1_sample_std": aggregates[
                    configuration_id
                ]["sample_std"],
                "upper_bound_vs_accuracy_leader": comparisons[
                    configuration_id
                ]["one_sided_95_percent_upper_bound"],
                "practically_equivalent": comparisons[configuration_id][
                    "practically_equivalent"
                ],
                "selected": configuration_id == winner,
            }
        )

    source_files = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "03_代码/src/wafermap/stem_ablation_models.py",
        PROJECT_ROOT / "03_代码/src/wafermap/controlled_ce_training.py",
        PROJECT_ROOT / "03_代码/src/wafermap/transforms.py",
        PROJECT_ROOT / "03_代码/src/wafermap/dataset.py",
    )
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in source_files
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "frozen_after_validation_selection",
        "protocol_id": "20260805_stem_2x2_validation_selection",
        "selection_split": "validation only",
        "test_inputs_used": False,
        "seeds": list(SEEDS),
        "epochs_per_seed": len(EXPECTED_EPOCHS),
        "primary_metric": "best validation Macro-F1 per seed",
        "practical_equivalence_margin": PRACTICAL_EQUIVALENCE_MARGIN,
        "one_sided_t_critical_df2_95": ONE_SIDED_T_CRITICAL_DF2_95,
        "sample_standard_deviation_ddof": 1,
        "configurations": [asdict(item) for item in CONFIGURATIONS],
        "validation_records": records,
        "aggregate_statistics": aggregates,
        "accuracy_leader": leader,
        "comparisons_to_accuracy_leader": comparisons,
        "eligible_configurations": eligible,
        "selected_architecture_id": winner,
        "selection_rule": (
            "Among configurations whose paired one-sided 95% upper bound "
            "is <= 0.005 Macro-F1 below the validation leader, select the "
            "lowest-FLOP configuration; then higher mean, lower SD, fixed rank."
        ),
        "input_history_sha256": input_hashes,
        "source_sha256": source_hashes,
    }
    return payload, aggregate_rows


def ensure_outputs_are_available():
    existing = [path for path in (OUTPUT_DIR, FREEZE_FILE) if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite frozen selection outputs:\n" + formatted
        )


def write_outputs(payload, records, aggregate_rows):
    ensure_outputs_are_available()
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="stem_selection_",
        dir=OUTPUT_DIR.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        write_csv(temporary / "per_seed_validation.csv", records)
        write_csv(temporary / "aggregate_validation.csv", aggregate_rows)
        decision_file = temporary / "architecture_decision.json"
        decision_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "protocol_id": payload["protocol_id"],
                    "files_sha256": artifacts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_freeze = FREEZE_FILE.with_suffix(".json.tmp")
        if temporary_freeze.exists():
            raise FileExistsError(f"Temporary freeze file exists: {temporary_freeze}")
        temporary_freeze.write_text(
            decision_file.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        shutil.copytree(temporary, OUTPUT_DIR)
        temporary_freeze.replace(FREEZE_FILE)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    ensure_outputs_are_available()
    records, scores, input_hashes = load_validation_scores()
    payload, aggregate_rows = build_payload(records, scores, input_hashes)

    print("Validation-only stem selection preflight passed")
    print(f"accuracy_leader={payload['accuracy_leader']}")
    print(f"selected_architecture={payload['selected_architecture_id']}")
    print(
        "eligible_configurations="
        + ",".join(payload["eligible_configurations"])
    )

    if args.preflight:
        print("No files were written.")
        return

    write_outputs(payload, records, aggregate_rows)
    print(f"Frozen decision: {FREEZE_FILE}")
    print(f"Validation outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
