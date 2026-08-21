#!/usr/bin/env python3
"""Smoke-test every entry script in 03_代码/scripts without doing real work.

Strategy (safe by construction):
1. Scripts WITHOUT argparse would execute real work if run -> skipped, reported.
2. Scripts WITH argparse: run `--help` first (always safe).
3. If `--help` output advertises a safe flag (--preflight / --check-inputs /
   --smoke / --dry-run), additionally run that flag to exercise input/output
   path validation without training or inference.

Writes 00_项目管理/20260821_冒烟测试报告.md
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "03_代码" / "scripts"
REPORT = PROJECT_ROOT / "00_项目管理" / "20260821_冒烟测试报告.md"
PYTHON = sys.executable

SAFE_FLAGS = ("--preflight", "--check-inputs", "--smoke", "--dry-run")
SKIP_REASONS = {
    "preprocess_wm811k.py": "无 CLI：直接执行会重跑 17 万图预处理（确定性，勿随意运行）",
    "make_wm811k_splits.py": "无 CLI：直接执行会重写划分文件",
    "check_mps.py": "无 CLI：真实执行 MPS 环境检查并写报告文件",
    "build_final_freeze_manifest.py": "构建类：会写冻结清单（存在即拒绝覆盖）",
    "build_manuscript_v5.py": "构建类：会重写 v5 稿件",
    "audit_presubmission.py": "构建类：会重写审计报告",
    "run_smoke_tests.py": "本冒烟脚本自身（源码含 argparse 字符串导致误判，显式跳过）",
}

# 需要额外必需参数才能执行安全标志的脚本
EXTRA_ARGS = {
    "train_kang_stacking_seed.py": ["--seed", "42"],
}

# safe_rc == 1 但属于“输出已存在、拒绝覆盖”的预期防护行为（本地已完成全部实验，
# 全新 clone 上 preflight 会通过），不视为异常。
EXPECTED_REJECT_MARKERS = (
    "Refusing to overwrite",
    "Refusing to repeat",
    "refusing to overwrite",
    "Choose a seed that has not been run",
    "output_exists=True",
    "already exists",
)


def has_argparse(source: str) -> bool:
    return "import argparse" in source or "from argparse import" in source


def run(python: str, script: Path, *args: str, timeout: int) -> tuple[int, str]:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "03_代码" / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        result = subprocess.run(
            [python, str(script), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
            env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -99, f"TIMEOUT after {timeout}s"
    except Exception as error:  # noqa: BLE001
        return -98, f"EXCEPTION: {error}"


def main() -> None:
    rows = []
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        source = script.read_text(encoding="utf-8")
        name = script.name

        if name in SKIP_REASONS or not has_argparse(source):
            rows.append(
                {
                    "script": name,
                    "help_rc": "-",
                    "safe_flag": "-",
                    "safe_rc": "-",
                    "summary": SKIP_REASONS.get(name, "无 argparse：跳过（直接运行会执行真实工作）"),
                }
            )
            continue

        help_rc, help_out = run(PYTHON, script, "--help", timeout=60)
        advertised = [f for f in SAFE_FLAGS if f in help_out]
        safe_rc, safe_out, flag_used = "-", "-", "-"
        if help_rc == 0 and advertised:
            flag_used = advertised[0]
            extra = EXTRA_ARGS.get(name, [])
            safe_rc, safe_out = run(PYTHON, script, flag_used, *extra, timeout=180)
        tail = " | ".join(
            line.strip() for line in (safe_out if safe_rc != "-" else help_out).splitlines()[-3:] if line.strip()
        )[:220]
        if str(safe_rc) == "1" and any(marker in safe_out for marker in EXPECTED_REJECT_MARKERS):
            status = f"预期拒绝（输出已存在）→ {tail}"
        else:
            status = tail
        rows.append(
            {
                "script": name,
                "help_rc": str(help_rc),
                "safe_flag": flag_used,
                "safe_rc": str(safe_rc),
                "summary": status,
            }
        )

    lines = [
        "# 20260821 入口脚本冒烟测试报告",
        "",
        f"- 执行时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Python：{PYTHON}",
        f"- 范围：`03_代码/scripts/` 全部 {len(rows)} 个脚本",
        "- 安全策略：无 argparse 或构建类脚本一律跳过（不执行真实工作）；有 argparse 的脚本先 `--help`，若提供安全标志（--preflight/--check-inputs/--smoke/--dry-run）再执行该标志。",
        "",
        "## 结果汇总",
        "",
    ]
    passed = [r for r in rows if r["safe_rc"] == "0" or (r["safe_rc"] == "-" and r["help_rc"] == "0")]
    lines.append(f"- `--help`/安全标志执行通过：**{len(passed)}/{len(rows)}**")
    rejected = [r for r in rows if r["safe_rc"] == "1" and "预期拒绝" in r["summary"]]
    if rejected:
        lines.append(f"- 预期拒绝（本地输出已存在，防护生效）：**{len(rejected)}**")
    failed = [r for r in rows if r["safe_rc"] not in ("0", "-", "1") or (r["safe_rc"] == "1" and "预期拒绝" not in r["summary"]) or r["help_rc"] not in ("0", "-")]
    if failed:
        lines.append(f"- 真异常（需人工查看）：**{len(failed)}**")
    else:
        lines.append("- 真异常：**0**")
    lines += ["", "## 明细", "", "| 脚本 | --help | 安全标志 | 安全执行 | 输出摘要 |", "|---|---:|---|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r['script']} | {r['help_rc']} | {r['safe_flag']} | {r['safe_rc']} | {r['summary']} |"
        )
    lines += ["", "## 说明", "", "- `-` 表示不适用（跳过或无需执行）。", "- 预检失败通常表示『本地未准备对应输入』（如 checkpoint 缺失），不一定是脚本缺陷；复现前按 README 顺序准备即可。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT} ({len(rows)} scripts)")


if __name__ == "__main__":
    main()
