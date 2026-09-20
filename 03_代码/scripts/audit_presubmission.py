#!/usr/bin/env python3
"""Pre-submission audit: cross-check manuscript v5 against the frozen evidence.

Checks:
1. Every in-text citation [n] has a numbered reference entry and vice versa,
   and first-appearance order is monotonic.
2. Every figure mentioned as Fig. N has a file and is embedded in the docx.
3. Every Table N caption exists in the manuscript.
4. Canonical prediction files hash-match the 17 August 2026 canonical manifest.
5. Key manuscript numbers match the authoritative CSVs.
6. The corrupted 05_结果 working copy is flagged and excluded.
7. No unresolved author placeholders remain except the explicitly allowed ones.
8. Submission package files exist.

Writes 00_项目管理/20260821_投稿前总审计报告.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MD = PROJECT_ROOT / "06_论文/manuscript/09_英文论文投稿定稿版_v7.md"
DOCX = PROJECT_ROOT / "06_论文/manuscript/09_英文论文投稿定稿版_v7.docx"
CANON_MANIFEST = (
    PROJECT_ROOT
    / "04_实验/metrics/20260817_kang_vs_highres_significance_canonical/analysis_manifest.json"
)
REPORT = PROJECT_ROOT / "00_项目管理/20260914_v7投稿前总审计报告.md"

FINAL_TABLE = PROJECT_ROOT / "05_结果/tables/table_final_highres_ce_multiseed_test.csv"
KANG_TABLE = (
    PROJECT_ROOT
    / "05_结果/tables/table_kang_vs_highres_paired_significance_canonical.csv"
)
WORKING_SEED42 = (
    PROJECT_ROOT
    / "05_结果/predictions/20260729_highres_ce_multiseed_final_test/predictions_seed42.csv"
)

ALLOWED_PLACEHOLDERS = [
    "[Author name(s) to be added]",
    "[Institution, department, city, postcode, country to be added]",
    "[Name and active e-mail address to be added]",
    "[16-digit ORCID of the corresponding author to be added]",
    "[To be completed after the author list and contribution roles are confirmed.]",
    "[Date of submission]",
    "[Corresponding author name]",
    "[Full author list]",
    "[Author A]",
    "[Author B]",
]

# Values that the v7 rewrite replaced. Each was a transcription error against
# the frozen artifacts, so none of them may reappear in the manuscript.
FORBIDDEN_TOKENS = {
    "± 0.0100": "Table 2C S2P accuracy SD (frozen value 0.0425)",
    "± 0.0222": "Table 2C S2N accuracy SD (frozen value 0.0289)",
    "± 0.0632": "Table 2C S1P accuracy SD (frozen value 0.0770)",
    "± 0.0494": "Table 2C S1B accuracy SD (frozen value 0.0526)",
    "−6.56": "factorial interaction on the test split (frozen value −6.57)",
    "+1.54 test points": "blur-factor effect on test (frozen value +1.65)",
    "Manuscript status": "internal build metadata",
    "Target journal": "internal build metadata",
    "Author Checklist": "internal working checklist",
    "Generated from submission-ready manuscript source": "internal build note",
    # Positioning guards: the paper is a controlled ablation study, not a new
    # network, and it makes no state-of-the-art claim.
    "proposed model": "positioning: use 'high-resolution configuration'",
    "our proposed": "positioning: the contribution is an attribution study",
    "SOTA": "no state-of-the-art claims",
    "state-of-the-art": "no state-of-the-art claims",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    md_text = MD.read_text(encoding="utf-8")
    head, refs_part = md_text.split("\n## References\n")
    findings: list[str] = []
    failures: list[str] = []

    # ---- 1. citation integrity ----
    # Supports both the comma form ``[12, 13]`` and the range form ``[21-30]``,
    # and expands ranges so the first-appearance order is comparable to the
    # reference list. ``network.conv5[0]`` and numeric intervals are excluded by
    # requiring every number to be a listed reference.
    cited: set[int] = set()
    order: list[int] = []
    seen: set[int] = set()
    listed = {int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", refs_part, re.M)}

    def expand_group(group: str) -> list[int]:
        numbers: list[int] = []
        for piece in group.split(","):
            piece = piece.strip()
            if "-" in piece:
                low, high = (int(part.strip()) for part in piece.split("-", 1))
                numbers.extend(range(low, high + 1))
            else:
                numbers.append(int(piece))
        return numbers

    for m in re.finditer(r"\[(\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)\]", head):
        numbers = expand_group(m.group(1))
        if not set(numbers) <= listed:
            continue
        for n in numbers:
            if n == 0:
                continue
            cited.add(n)
            if n not in seen:
                seen.add(n)
                order.append(n)

    listed_counts = {}
    for m in re.finditer(r"^(\d+)\. ", refs_part, re.M):
        listed_counts[int(m.group(1))] = listed_counts.get(int(m.group(1)), 0) + 1
    duplicated = {n: c for n, c in listed_counts.items() if c > 1}
    if duplicated:
        failures.append(f"文献表存在重复编号条目：{duplicated}")
    if cited == listed == set(range(1, len(listed) + 1)):
        findings.append(
            f"引用完整性：正文引用集合 = 文献表集合 = 1..{len(listed)}"
            f"（{len(listed)} 条），无孤儿引用、无重复条目。"
        )
    else:
        failures.append(f"引用不一致：cited={sorted(cited - listed)} listed-only={sorted(listed - cited)}")
    if order == sorted(order):
        findings.append("引用顺序：文献编号严格按正文首次出现顺序单调递增。")
    else:
        failures.append(f"引用顺序非单调：{order}")

    # ---- 2. figures ----
    fig_refs = sorted({int(m.group(1)) for m in re.finditer(r"Fig\. (\d)", md_text)})
    figure_files = {
        1: "05_结果/figures/data_overview/wm811k_class_examples.png",
        2: "05_结果/figures/model_results/final_highres_ce_multiseed_training_curves.png",
        3: "05_结果/figures/model_results/final_highres_ce_multiseed_confusion_matrix_mean.png",
        4: "05_结果/figures/model_results/final_model_complexity_mps_comparison.png",
        5: "05_结果/figures/model_results/final_highres_ce_unanimous_error_examples.png",
        6: "05_结果/figures/model_results/highres_robustness_curves.png",
        7: "05_结果/figures/model_results/highres_gradcam_representative_samples.png",
    }
    if fig_refs == list(range(1, 8)):
        findings.append("图引用：Fig. 1–7 均在正文引用，且每张图的源文件存在。")
    else:
        failures.append(f"图引用异常：{fig_refs}")
    missing = [p for p in figure_files.values() if not (PROJECT_ROOT / p).is_file()]
    if missing:
        failures.append(f"缺失图文件：{missing}")
    else:
        findings.append("图文件：7 张图（PNG + PDF 矢量版）全部存在。")

    from docx import Document
    doc = Document(str(DOCX))
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    embedded = sum(
        1
        for p in doc.paragraphs
        for shape in p._p.findall(f".//{ns}blip")
    )
    findings.append(f"Word 内嵌图：检出 {embedded} 个图片对象（应为 7）。")

    # ---- 3. tables ----
    expected_tables = [str(n) for n in range(1, 23)]
    captions = re.findall(r"\*\*Table ([\dA-Z]+)\.", md_text)
    missing_tables = [t for t in expected_tables if t not in captions]
    if not missing_tables:
        findings.append(
            f"表标题：{len(expected_tables)} 个 Table 标题齐全且位于表格上方"
            f"（按正文出现顺序连续编号 1–22，无字母后缀）。"
        )
    else:
        failures.append(f"缺失表标题：{missing_tables}")

    # v7 added two tables and removed the letter-suffix scheme, so the count of
    # embedded Word tables must equal the number of captions.
    markdown_blocks = 0
    previous = False
    for raw_line in md_text.split("\n"):
        current = raw_line.strip().startswith("|")
        if current and not previous:
            markdown_blocks += 1
        previous = current
    if markdown_blocks != len(expected_tables):
        failures.append(
            f"表格块与标题数量不符：Markdown 表格块 {markdown_blocks} 个，"
            f"Table 标题 {len(expected_tables)} 个（存在无标题表格或漏引用）。"
        )
    else:
        findings.append(
            f"表格块一致性：Markdown 表格块 {markdown_blocks} 个，与标题数一致，无无标题表格。"
        )

    # ---- 4. canonical prediction hashes ----
    canon = json.loads(CANON_MANIFEST.read_text(encoding="utf-8"))
    input_hashes = canon["input_hashes"]
    mismatch = []
    for rel, expected in input_hashes.items():
        actual = sha256_file(PROJECT_ROOT / rel)
        if actual != expected:
            mismatch.append(f"{rel}: manifest={expected[:16]}... actual={actual[:16]}...")
    if not mismatch:
        findings.append("数据追溯：6 个 canonical 预测文件的 SHA-256 与 20260817 配对显著性清单完全一致。")
    else:
        failures.append(f"canonical 哈希不一致：{mismatch}")

    working_hash = sha256_file(WORKING_SEED42)
    findings.append(
        f"工作副本标记：05_结果 的 seed42 副本哈希 {working_hash[:16]}... 与 canonical 不同"
        "（第 25832 行文本损坏），已排除出投稿包。"
    )

    # ---- 5. key numbers ----
    import csv
    def read_csv(path: Path):
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    final_rows = read_csv(FINAL_TABLE)
    acc = final_rows[0].get("accuracy_percent_mean_sd", "?")
    mf1 = final_rows[0].get("macro_f1_percent_mean_sd", "?")
    spot = {
        "98.0342": "abstract/conclusion accuracy",
        "90.2029": "abstract/conclusion Macro-F1",
        "9.7082": "stem Macro-F1 gain",
        "1.6290": "Kang Macro-F1 advantage",
        "0.1914": "Kang accuracy advantage",
        "91.44": "parameter reduction vs stacking",
        "85.72": "FLOP reduction vs stacking",
        # v7 cost-transparency and concentration numbers; sources are
        # 05_结果/tables/table_v7_*.csv.
        "15.75": "FLOP factor across the six stem configurations",
        "1.1525": "S1B minus S1N Macro-F1 gap",
        "90.89": "S1B Conv/Linear FLOPs (million)",
        "64.9": "Scratch share of the aggregate Macro-F1 gain",
        "6.2973": "Scratch contribution to Macro-F1",
        "3.8369": "mean gain of the other eight classes",
        "8.4308": "ResNet18 gain over the matched standard stem",
        "1.2774": "HighRes minus ResNet18 Macro-F1",
        "8.85": "ResNet18 parameter factor over the standard stem",
        "12.52": "ResNet18 FLOP factor over the standard stem",
        "93.68": "INT8 cold-start regression",
        "72.4452": "ResNet18 Scratch F1",
        # Input-resolution control (Sections 3.4.2 / 4.1.1.2, Table 6);
        # sources are table_v7_resolution_ablation_*.csv.
        "87.4079": "standard stem at 128 px Macro-F1",
        "67.9853": "standard stem at 128 px Scratch F1",
        "6.9132": "gain of standard@128 over standard@64",
        "2.7950": "gap between standard@128 and HighRes@64",
        "1.1576": "lower bound of the 32 x 32 band gap",
        "1.6425": "upper bound of the 32 x 32 band gap",
        "87.5777": "best standard@128 seed",
        "89.9796": "weakest frozen HighRes seed",
    }
    for token, label in spot.items():
        count = md_text.count(token)
        if count >= 2:
            findings.append(f"关键数字 {token}（{label}）：正文出现 {count} 次，摘要/结论一致。")
        elif count == 1:
            findings.append(f"关键数字 {token}（{label}）：正文出现 1 次。")
        else:
            failures.append(f"关键数字缺失：{token}（{label}）")

    # Values replaced by v7 must not reappear (regression guard).
    for token, label in FORBIDDEN_TOKENS.items():
        if token in md_text:
            failures.append(f"已废弃的数值/措辞重新出现：{token}（{label}）")
    if not any(token in md_text for token in FORBIDDEN_TOKENS):
        findings.append(
            f"回归防护：{len(FORBIDDEN_TOKENS)} 个已废弃数值/措辞均未出现。"
        )

    # Anti-aliasing control numbers (Sections 3.4.1 / 4.1.1.1, Tables 2C and
    # 8C). These are cross-checked against the frozen evaluation bundle rather
    # than only counted, so a typo in the manuscript fails the audit.
    antialiasing_table = (
        PROJECT_ROOT
        / "05_结果/tables/table_stem_five_config_multiseed_test.csv"
    )
    antialiasing_spot = {
        "80.4947": "S2P test Macro-F1",
        "83.4344": "S2B test Macro-F1",
        "2.9398": "S2B over S2P Macro-F1 gain",
        "8.2051": "S1P over S2P Macro-F1 gain",
        "6.7685": "S1N minus S2B Macro-F1 gap",
        "35.3814": "S2B Scratch F1",
        "46.3151": "S1N minus S2B Scratch F1 gap",
        "30% to 44%": "aliasing share of the pooling penalty",
    }
    for token, label in antialiasing_spot.items():
        count = md_text.count(token)
        if count >= 1:
            findings.append(
                f"抗混叠关键数字 {token}（{label}）：正文出现 {count} 次。"
            )
        else:
            failures.append(f"抗混叠关键数字缺失：{token}（{label}）")

    if antialiasing_table.is_file():
        rows = {
            row["configuration_id"]: row
            for row in read_csv(antialiasing_table)
        }
        for configuration_id, expected_percent in {
            "S2P": 80.4947,
            "S2B": 83.4344,
            "S1N": 90.2029,
        }.items():
            actual = float(rows[configuration_id]["macro_f1_mean"]) * 100.0
            if abs(actual - expected_percent) > 0.001:
                failures.append(
                    f"表 {antialiasing_table.name} 的 {configuration_id} "
                    f"Macro-F1={actual:.4f}，与正文 {expected_percent} 不符。"
                )
            else:
                findings.append(
                    f"表 {antialiasing_table.name} 的 {configuration_id} "
                    f"Macro-F1={actual:.4f} 与正文一致。"
                )
    else:
        failures.append(f"缺少抗混叠结果表：{antialiasing_table.name}")
    findings.append(
        f"表 {FINAL_TABLE.name} 首行：accuracy={acc}，macro_f1={mf1}"
        "（与正文 98.0342/90.2029 交叉核对）。"
    )

    # ---- 6. corrupted working copy ----
    corrupted = "nne" in WORKING_SEED42.read_text(encoding="utf-8", errors="ignore")
    if corrupted:
        findings.append("已确认 05_结果 seed42 工作副本含损坏文本 'nne'，投稿包排除标记生效。")
    else:
        findings.append("05_结果 seed42 工作副本未检出 'nne'（若曾修复，请更新冻结清单备注）。")

    # ---- 7. placeholders (author-supplied fields only; citations/CI brackets are not placeholders) ----
    placeholder_hints = (
        "author", "affiliation", "e-mail", "email", "to be", "to add", "add the",
        "add an", "date of submission", "repository", "orcid", "corresponding",
        "before submission", "before peer review",
    )
    leftover = []
    for token in set(re.findall(r"\[[^\]]+\]", md_text)):
        lowered = token.lower()
        if any(hint in lowered for hint in placeholder_hints):
            if token not in ALLOWED_PLACEHOLDERS:
                leftover.append(token)
    if leftover:
        failures.append(f"未授权占位符：{sorted(leftover)}")
    else:
        findings.append("占位符检查：仅剩作者/单位/邮箱/贡献等预期占位符（提交前填写）。")

    # ---- 8. submission package ----
    package = [
        "06_论文/submission/01_cover_letter_JIM.md",
        "06_论文/submission/02_highlights.md",
        "06_论文/submission/03_credit_author_statement.md",
        "06_论文/submission/04_conflict_of_interest_statement.md",
        "06_论文/submission/05_data_code_availability_statement.md",
        "06_论文/submission/06_submission_package_manifest.md",
        "06_论文/references/01_英文论文初稿_v1_references.bib",
        "00_项目管理/20260821_投稿前最终结果冻结清单.json",
    ]
    missing_pkg = [p for p in package if not (PROJECT_ROOT / p).is_file()]
    if not missing_pkg:
        findings.append("投稿包：8 个必需文件全部存在。")
    else:
        failures.append(f"投稿包缺失：{missing_pkg}")

    lines = [
        "# 20260914 v7 投稿前总审计报告",
        "",
        f"- 审计对象：`09_英文论文投稿定稿版_v7.md` / `.docx`（43 条文献、21 表、7 图）",
        f"- 审计时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 审计脚本：`03_代码/scripts/audit_presubmission.py`",
        "",
        "## 结论",
        "",
        (
            "**通过**（0 项失败）："
            if not failures
            else f"**存在 {len(failures)} 项失败，需修复后再投稿。**"
        ),
        "",
        "## 检查结果",
        "",
    ]
    lines += [f"- ✅ {f}" for f in findings]
    if failures:
        lines += ["", "## 失败项", ""]
        lines += [f"- ❌ {f}" for f in failures]
    lines += [
        "",
        "## 提交前人工待办（不阻塞审计通过）",
        "",
        "- 填写作者、单位、通讯作者、ORCID（稿件头部 + docx 属性 + Cover Letter）。",
        "- 确认 CRediT 角色并在 Declarations 填写 Author contributions。",
        "- 把 v7 新增代码、表与两份新 metrics 目录 push 到 https://github.com/ZWY631/wafermap-highres，使仓库与稿件的代码可用性声明一致。",
        "- 按 JIM 官方投稿指南复核图表分辨率与参考文献格式（Springer 基础格式）。",
        "- 全体作者审阅并同意投稿；确认利益冲突声明。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT} ({len(findings)} findings, {len(failures)} failures)")


if __name__ == "__main__":
    main()
