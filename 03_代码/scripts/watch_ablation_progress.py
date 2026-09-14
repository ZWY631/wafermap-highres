#!/usr/bin/env python3
"""实时查看 blur-pool 消融（S2B/S1B）训练进度与验证精度。

用法
----
    python 03_代码/scripts/watch_ablation_progress.py            # 看一次
    python 03_代码/scripts/watch_ablation_progress.py --watch    # 每 30 秒刷新
    python 03_代码/scripts/watch_ablation_progress.py --watch --interval 10

看什么
------
* **验证最优 Macro-F1**：论文口径的模型选择指标（best validation Macro-F1）。
  一个 run 的最终成绩 = 它跑完 30 epoch 后的这一列。
* **验证 Accuracy**：总体正确率，被 none 大类主导，参考价值低于 Macro-F1。
* **预注册判定**要等 6 个 run 全部完成、再做一次性**测试集**评估后才能算，
  本脚本只看训练期的验证集，不构成结论。

数据来源：`04_实验/metrics/<run_name>_history.csv`，每个 epoch 追加一行。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "04_实验" / "metrics"
CHECKPOINT_DIR = PROJECT_ROOT / "04_实验" / "checkpoints"

EPOCHS = 30

# (显示名, run_name 前缀, 分组)
RUNS: tuple[tuple[str, str, str], ...] = (
    ("S2P 标准  stride2+maxpool", "shufflenet_v2_standard_ce_full", "参照(冻结)"),
    ("S2N       stride2+identity", "shufflenet_v2_stem_s2_nopool_ce_full", "参照(冻结)"),
    ("S1P       stride1+maxpool", "shufflenet_v2_stem_s1_pool_ce_full", "参照(冻结)"),
    ("S1N 主模型 stride1+identity", "shufflenet_v2_highres_ce_full", "参照(冻结)"),
    ("S2B 新    stride2+blurpool", "shufflenet_v2_stem_s2_blurpool_ce_full", "本次实验"),
    ("S1B 新    stride1+blurpool", "shufflenet_v2_stem_s1_blurpool_ce_full", "本次实验"),
)

SEEDS = (42, 123, 2026)


def history_path(base_run_name: str, seed: int) -> Path:
    """复刻 controlled_ce_training.run_name_for_seed 的命名规则。"""
    suffix = "" if seed == 42 else f"_seed{seed}"
    return METRICS_DIR / f"{base_run_name}{suffix}_history.csv"


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_state(base_run_name: str, seed: int) -> dict:
    rows = read_history(history_path(base_run_name, seed))
    suffix = "" if seed == 42 else f"_seed{seed}"
    run_name = f"{base_run_name}{suffix}"
    checkpoint_dir = CHECKPOINT_DIR / run_name

    state: dict = {
        "epochs_done": len(rows),
        "best_epoch": None,
        "best_macro_f1": None,
        "best_accuracy": None,
        "latest_macro_f1": None,
        "seconds_per_epoch": None,
        "complete": len(rows) >= EPOCHS,
        "has_manifest": (checkpoint_dir / "run_manifest.json").is_file(),
        "checkpoint_dir": checkpoint_dir.name,
    }
    if not rows:
        return state

    best = max(rows, key=lambda row: float(row["val_macro_f1"]))
    state["best_epoch"] = int(best["epoch"])
    state["best_macro_f1"] = float(best["val_macro_f1"])
    state["best_accuracy"] = float(best["val_accuracy"])
    state["latest_macro_f1"] = float(rows[-1]["val_macro_f1"])

    durations = [float(row["epoch_seconds"]) for row in rows if row.get("epoch_seconds")]
    if durations:
        state["seconds_per_epoch"] = sum(durations) / len(durations)
    return state


def active_processes() -> list[str]:
    """返回当前正在训练的训练脚本参数（不依赖 psutil）。"""
    entries: list[str] = []
    proc_dir = Path("/proc")
    if proc_dir.is_dir():
        for candidate in proc_dir.iterdir():
            if not candidate.name.isdigit():
                continue
            try:
                cmdline = (candidate / "cmdline").read_bytes().decode(errors="replace")
            except OSError:
                continue
            if "train_shufflenet_stem" in cmdline:
                entries.append(cmdline.replace("\0", " ").strip())
        return entries

    # macOS: 退回解析 ps 输出
    import subprocess

    result = subprocess.run(
        ["ps", "-eo", "pid,etime,command"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if "train_shufflenet_stem" in line and "grep" not in line:
            entries.append(line.strip())
    return entries


def format_row(label: str, seed: int, state: dict, is_new_run: bool) -> str:
    if state["epochs_done"] == 0:
        return f"  {label:<28} seed {seed:<5} {'未开始':>10}"

    progress = f"{state['epochs_done']}/{EPOCHS}"
    if state["complete"]:
        # 冻结的历史 run 没有 run_manifest.json（该机制是后来才加的），
        # 只有本次新增的 run 才要求 manifest 落盘。
        if is_new_run and not state["has_manifest"]:
            progress += " 待落盘"
        else:
            progress += " 完成"
    else:
        progress += " 训练中"

    speed = (
        f"{state['seconds_per_epoch']:.0f}s/ep"
        if state["seconds_per_epoch"]
        else "—"
    )
    return (
        f"  {label:<28} seed {seed:<5} {progress:<16} "
        f"最优 Macro-F1={state['best_macro_f1']:.4f}@ep{state['best_epoch']:<3} "
        f"Acc={state['best_accuracy']:.4f}  "
        f"最新={state['latest_macro_f1']:.4f}  {speed}"
    )


def render() -> tuple[str, dict]:
    lines = [
        "=" * 96,
        f"blur-pool 消融训练进度    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 96,
    ]

    summary: dict = {}
    totals = {"runs": 0, "complete": 0}

    for group in ("参照(冻结)", "本次实验"):
        lines.append(f"\n[{group}]")
        for label, base_run_name, run_group in RUNS:
            if run_group != group:
                continue
            for seed in SEEDS:
                state = run_state(base_run_name, seed)
                lines.append(
                    format_row(label, seed, state, is_new_run=(group == "本次实验"))
                )
                if group == "本次实验":
                    totals["runs"] += 1
                    if state["complete"] and state["has_manifest"]:
                        totals["complete"] += 1
                    summary[f"{base_run_name}|{seed}"] = state

    running = active_processes()
    lines.append("\n[当前进程]")
    if running:
        for entry in running:
            lines.append(f"  {entry}")
    else:
        lines.append("  没有训练进程在运行")

    lines.append(
        f"\n[本次实验进度] {totals['complete']}/{totals['runs']} 个 run 已全部完成并落盘"
    )
    if totals["complete"] < totals["runs"]:
        remaining = totals["runs"] - totals["complete"]
        in_flight = [s for s in summary.values() if 0 < s["epochs_done"] < EPOCHS]
        estimate = ""
        if in_flight and in_flight[0]["seconds_per_epoch"]:
            seconds = in_flight[0]["seconds_per_epoch"]
            epochs_left = sum(EPOCHS - s["epochs_done"] for s in in_flight)
            total_left = epochs_left + (remaining - len(in_flight)) * EPOCHS
            estimate = f"  预计还需约 {total_left * seconds / 3600:.1f} 小时"
        lines.append(f"  剩余 {remaining} 个 run{estimate}")

    lines.append(
        "\n提示：这里只有训练期的**验证集**精度。预注册判定要等 6 个 run 全部完成，"
        "\n      再做一次性**测试集**评估才能算，见 "
        "00_项目管理/20260913_混叠消融预注册判定规则.md"
    )
    return "\n".join(lines), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="持续刷新")
    parser.add_argument("--interval", type=int, default=30, help="刷新间隔秒数")
    args = parser.parse_args(argv)

    if not METRICS_DIR.is_dir():
        print(f"找不到指标目录：{METRICS_DIR}", file=sys.stderr)
        return 1

    while True:
        text, _ = render()
        if args.watch:
            os.system("clear")
        print(text, flush=True)

        if not args.watch:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n已停止监视。")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
