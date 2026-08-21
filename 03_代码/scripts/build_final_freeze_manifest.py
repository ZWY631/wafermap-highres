#!/usr/bin/env python3
"""Build the pre-submission consolidated final-results freeze manifest.

Hashes every artifact that the v4/v5 manuscript and the submission package
depend on: final paper tables, figures, canonical test predictions, analysis
manifests, checkpoint run manifests, the lot-disjoint split manifest, the
manuscript sources, and the historical freeze lists in 00_项目管理.

Conventions:
- Canonical predictions live in 04_实验/metrics/... and are the paper evidence.
  The copy under 05_结果/predictions/... is a working copy; the seed-42 file in
  that copy is known to contain one textually corrupted row and is therefore
  excluded from the submission package (still hashed for the audit trail).
- Output refuses to overwrite an existing manifest.

Usage:
    PYTHONPATH=03_代码/src wafer-sci-python 03_代码/scripts/build_final_freeze_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = PROJECT_ROOT / "00_项目管理" / "20260821_投稿前最终结果冻结清单.json"
OUTPUT_MD = PROJECT_ROOT / "00_项目管理" / "20260821_投稿前最终结果冻结清单.md"

SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(rel_dir: str, pattern: str) -> list[dict]:
    entries = []
    base = PROJECT_ROOT / rel_dir
    for path in sorted(base.glob(pattern)):
        if path.is_file():
            entries.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return entries


def hash_files(paths: list[str]) -> list[dict]:
    entries = []
    for rel in paths:
        path = PROJECT_ROOT / rel
        if path.is_file():
            entries.append(
                {
                    "path": rel,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return entries


def main() -> None:
    if OUTPUT_JSON.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {OUTPUT_JSON}")

    # 1. Paper tables: all tables in 05_结果/tables.
    tables = hash_tree("05_结果/tables", "*.csv")

    # 2. Figures: PNG and vector PDF variants.
    figures = hash_tree("05_结果/figures", "**/*.png") + hash_tree(
        "05_结果/figures", "**/*.pdf"
    )

    # 3. Canonical test predictions used as paper evidence.
    canonical_predictions = hash_files(
        [
            "04_实验/metrics/20260729_highres_ce_multiseed_final_test/predictions/predictions_seed42.csv",
            "04_实验/metrics/20260729_highres_ce_multiseed_final_test/predictions/predictions_seed123.csv",
            "04_实验/metrics/20260729_highres_ce_multiseed_final_test/predictions/predictions_seed2026.csv",
            "04_实验/metrics/20260807_kang_kang_stacking_lot_reproduction/predictions_seed42.csv",
            "04_实验/metrics/20260807_kang_kang_stacking_lot_reproduction/predictions_seed123.csv",
            "04_实验/metrics/20260807_kang_kang_stacking_lot_reproduction/predictions_seed2026.csv",
        ]
    )

    # Working copies under 05_结果 (audit only; seed-42 is known non-canonical).
    working_predictions = hash_tree(
        "05_结果/predictions/20260729_highres_ce_multiseed_final_test", "*.csv"
    )
    for entry in working_predictions:
        entry["role"] = (
            "working copy - NOT for submission (seed-42 row 25832 textually "
            "corrupted: 'nne' label / '0.o99968696' confidence)"
            if "seed42" in entry["path"]
            else "working copy - NOT for submission"
        )

    # 4. Analysis manifests (exclude _invalid_runs).
    manifests = []
    for manifest in sorted((PROJECT_ROOT / "04_实验/metrics").glob("*/analysis_manifest.json")):
        if "_invalid_runs" in manifest.parts:
            continue
        manifests.append(
            {
                "path": str(manifest.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(manifest),
                "size_bytes": manifest.stat().st_size,
            }
        )

    # 5. Checkpoint run manifests (they record checkpoint SHA-256 internally).
    checkpoint_manifests = hash_tree("04_实验/checkpoints", "*/run_manifest.json")

    # 6. Fixed lot-disjoint split manifest.
    split = hash_files(["02_数据/splits/wm811k_labeled_lot_disjoint.csv"])

    # 7. Manuscript sources and references (v4 audited baseline + v5 submission draft).
    manuscript = hash_files(
        [
            "06_论文/manuscript/05_英文论文领域对照增强版_v4.md",
            "06_论文/manuscript/05_英文论文领域对照增强版_v4.docx",
            "06_论文/manuscript/06_英文论文投稿定稿版_v5.md",
            "06_论文/manuscript/06_英文论文投稿定稿版_v5.docx",
            "06_论文/manuscript/07_中文翻译版_v5.md",
            "06_论文/manuscript/07_中文翻译版_v5.docx",
            "06_论文/manuscript/00_论文写作数据底稿.md",
            "06_论文/references/01_英文论文初稿_v1_references.bib",
        ]
    )

    # 8. Submission package materials.
    submission = hash_tree("06_论文/submission", "*.md")

    # 9. Historical freeze lists (00_项目管理 JSON + protocols).
    freeze_lists = hash_tree("00_项目管理", "*.json")

    # 10. Statistical-analysis and build/audit driver scripts referenced by the paper.
    drivers = hash_files(
        [
            "03_代码/scripts/analyze_kang_vs_highres_significance_canonical.py",
            "03_代码/scripts/analyze_kang_vs_highres_significance_corrected.py",
            "03_代码/scripts/build_manuscript_v5.py",
            "06_论文/manuscript/generate_manuscript_v5_docx.py",
            "03_代码/scripts/audit_presubmission.py",
        ]
    )

    manifest = {
        "manifest_id": "20260821_presubmission_final_freeze",
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "builder_script": "03_代码/scripts/build_final_freeze_manifest.py",
        "builder_script_sha256": SCRIPT_SHA256,
        "purpose": (
            "Consolidated pre-submission freeze of every result artifact cited "
            "by the manuscript, with SHA-256 for traceability and audit."
        ),
        "canonical_predictions_note": (
            "Paper evidence uses the prediction files under "
            "04_实验/metrics/... (hash-locked by the 17 August 2026 canonical "
            "paired-significance manifest). The 05_结果/predictions copies are "
            "working copies and are excluded from the submission package."
        ),
        "sections": {
            "paper_tables": tables,
            "figures": figures,
            "canonical_predictions": canonical_predictions,
            "working_predictions_audit_only": working_predictions,
            "analysis_manifests": manifests,
            "checkpoint_run_manifests": checkpoint_manifests,
            "split_manifest": split,
            "manuscript_and_references": manuscript,
            "submission_package": submission,
            "historical_freeze_lists": freeze_lists,
            "driver_scripts": drivers,
        },
        "authoritative_tables": {
            "kang_vs_highres_paired_significance": (
                "05_结果/tables/table_kang_vs_highres_paired_significance_canonical.csv"
            ),
            "note": (
                "table_kang_vs_highres_paired_significance.csv (original) and "
                "_corrected/_invalid_unit_bug variants are superseded; do not "
                "place them in the submission package."
            ),
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    total_files = sum(len(v) for v in manifest["sections"].values())
    lines = [
        "# 20260821 投稿前最终结果冻结清单",
        "",
        f"- 清单 ID：{manifest['manifest_id']}",
        f"- 生成时间：{manifest['generated_at']}",
        f"- 生成脚本：`{manifest['builder_script']}`（SHA-256 `{SCRIPT_SHA256}`）",
        f"- 哈希文件总数：{total_files}",
        "",
        "## 说明",
        "",
        "- 论文证据以 `04_实验/metrics/` 下哈希锁定的 canonical 预测文件为准。",
        "- `05_结果/predictions/` 下的副本为工作副本，seed-42 文件第 25832 行存在文本损坏（`nne`/`0.o99968696`），**不得**放入投稿包。",
        "- Kang vs HighRes 配对显著性以 `_canonical.csv` 为准，其余变体（原始/`_corrected`/`_invalid_unit_bug`）已废弃。",
        "",
        "## 分节文件数",
        "",
        "| 分节 | 文件数 |",
        "|---:|---:|",
    ]
    for name, entries in manifest["sections"].items():
        lines.append(f"| {name} | {len(entries)} |")
    lines += ["", "完整哈希见 `20260821_投稿前最终结果冻结清单.json`。", ""]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUTPUT_JSON} ({total_files} files)")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
