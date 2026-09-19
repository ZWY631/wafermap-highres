#!/bin/bash
# 分辨率×stem 消融：一键进度面板
# 用法：bash 03_代码/scripts/status_resolution_ablation.sh

ROOT="/Users/mima0000/论文111"
PY="/Users/mima0000/miniforge3/envs/wafer-sci/bin/python"
RAW="$ROOT/02_数据/raw/mirlab_2022"
LOG="$ROOT/04_实验/logs/_session_runner/resolution_ablation_pipeline.log"
ARCHIVE_BYTES=344542743

cd "$ROOT" || exit 1
echo "════════════════════════════════════════════════════════════════"
echo " 分辨率 × stem 消融  进度面板        $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"

# ---- 1. 原始数据 ----
echo
echo "【1/5 原始数据】"
if [ -f "$RAW/LSWMD.pkl" ] && [ "$(stat -f%z "$RAW/LSWMD.pkl" 2>/dev/null || echo 0)" -gt 1000000000 ]; then
  echo "  ✅ 已就位（手动放入）：$(du -h "$RAW/LSWMD.pkl" | cut -f1)"
elif [ -f "$RAW/MIR-WM811K.zip" ] && [ "$(stat -f%z "$RAW/MIR-WM811K.zip" 2>/dev/null || echo 0)" -eq "$ARCHIVE_BYTES" ]; then
  echo "  ✅ 归档完整（$(du -h "$RAW/MIR-WM811K.zip" | cut -f1)），等待解压"
else
  done_bytes=$("$PY" - <<PYEOF
import glob, os
print(sum(os.path.getsize(f) for f in glob.glob("$RAW/seg.*") + glob.glob("$RAW/.cur")))
PYEOF
)
  pct=$(( done_bytes * 100 / ARCHIVE_BYTES ))
  echo "  ⏳ 下载中：${pct}%  （$(( done_bytes / 1048576 )) MiB / 328 MiB）"
  bars=$(( pct / 3 ))
  printf "     ["
  printf '█%.0s' $(seq 1 $bars 2>/dev/null)
  printf '░%.0s' $(seq 1 $(( 33 - bars )) 2>/dev/null)
  printf "]\n"
  if [ "$done_bytes" -gt 0 ]; then
    last=$("$PY" - <<PYEOF
import glob, os
try:
    files = glob.glob("$RAW/seg.*") + glob.glob("$RAW/.cur")
    newest = max(files, key=os.path.getmtime)
    age = __import__("time").time() - os.path.getmtime(newest)
    print(f"{age:.0f}")
except Exception:
    print("-1")
PYEOF
)
    if [ "$last" != "-1" ] && [ "$last" -lt 120 ]; then
      echo "     ↳ 数据仍在写入（最近 ${last} 秒内有更新）"
    else
      echo "     ↳ ⚠️ ${last} 秒无更新 —— 可能又被限流；耐心下载器会自动重试，无需干预"
    fi
  fi
fi
echo "  MIRLab 状态：$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 -r 0-999 "http://mirlab.org/dataset/public/MIR-WM811K.zip" 2>/dev/null || echo '---')  (206=可下载, 000=被限流/超时)"

# ---- 2. 预处理 ----
echo
echo "【2/5 预处理 128×128】"
if [ -f "$ROOT/02_数据/processed/wm811k_labeled_128x128/images_uint8.npy" ]; then
  echo "  ✅ 已完成：$(du -h "$ROOT/02_数据/processed/wm811k_labeled_128x128/images_uint8.npy" | cut -f1)"
else
  echo "  ○ 等待原始数据"
fi

# ---- 3. 64×64 对照门禁 ----
echo
echo "【3/5 64×64 对照门禁】（标准值 standard 0.812563 / highres 0.905377）"
"$PY" - <<'PYEOF'
import glob
import pandas as pd

GATES = {
    "shufflenet_v2_standard_res64_ce_full": ("standard", 0.812563),
    "shufflenet_v2_highres_res64_ce_full": ("highres", 0.905377),
}
for run, (tag, expected) in GATES.items():
    path = f"04_实验/metrics/{run}_history.csv"
    try:
        frame = pd.read_csv(path)
    except Exception:
        print(f"  ○ {tag:9s} 尚未开始")
        continue
    col = [c for c in frame.columns if "val" in c.lower() and "macro" in c.lower()][0]
    best = float(frame[col].max())
    epoch = len(frame)
    if epoch < 30:
        print(f"  ⏳ {tag:9s} epoch {epoch}/30，当前 best={best:.4f}（门禁值 {expected:.6f}）")
    else:
        delta = abs(best - expected)
        mark = "✅ PASS" if delta <= 0.002 else "❌ FAIL"
        print(f"  {mark} {tag:9s} 30/30  best={best:.6f}  门禁={expected:.6f}  差={delta:.6f}")
PYEOF

# ---- 4. 128×128 训练 ----
echo
echo "【4/5 128×128 训练】（预期单 run 约 4.5–5 小时）"
found=0
for stem in standard highres; do
  for seed in 42 123 2026; do
    run="shufflenet_v2_${stem}_res128_ce_full"
    [ "$seed" != "42" ] && run="${run}_seed${seed}"
    path="04_实验/metrics/${run}_history.csv"
    if [ -f "$path" ]; then
      found=1
      "$PY" - "$path" "$stem" "$seed" <<'PYEOF'
import sys
import pandas as pd
path, stem, seed = sys.argv[1], sys.argv[2], sys.argv[3]
frame = pd.read_csv(path)
if len(frame) == 0:
    print(f"  ⏳ {stem:9s} seed{seed:5s} 已启动，首轮尚未完成")
    raise SystemExit
col = [c for c in frame.columns if "val" in c.lower() and "macro" in c.lower()][0]
best = float(frame[col].max())
secs = frame["epoch_seconds"].iloc[-1] if "epoch_seconds" in frame else float("nan")
left = (30 - len(frame)) * secs / 60 if secs == secs else float("nan")
print(f"  ⏳ {stem:9s} seed{seed:5s} epoch {len(frame):2d}/30  best={best:.4f}  "
      f"约 {secs/60:.1f} 分/epoch，剩余约 {left/60:.1f} 小时")
PYEOF
    fi
  done
done
[ "$found" = "0" ] && echo "  ○ 尚未开始（等门禁通过）"
if [ -f "$ROOT/05_结果/tables/table_v7_resolution_ablation_multiseed_test.csv" ]; then
  n=$("$PY" -c "
import pandas as pd
d=pd.read_csv('$ROOT/05_结果/tables/table_v7_resolution_ablation_multiseed_test.csv')
print(len(d))" 2>/dev/null || echo 0)
  echo "  📊 已评估 $n 个格子（共 6）"
fi

# ---- 5. 流水线阶段 ----
echo
echo "【5/5 流水线】"
if [ -f "$LOG" ]; then
  echo "  最近日志："
  tail -6 "$LOG" | sed 's/^/    /'
else
  echo "  ○ 无日志"
fi

echo
echo "────────────────────────────────────────────────────────────────"
dl=$(pgrep -f "fetch_patient.sh" | wc -l | tr -d ' ')
pl=$(pgrep -fx "bash 03_代码/scripts/run_resolution_ablation_pipeline.sh" | wc -l | tr -d ' ')
tr=$(pgrep -f "train_resolution_ablation.py" | wc -l | tr -d ' ')
echo " 进程：耐心下载器 ${dl} ｜ 流水线 ${pl} ｜ 训练 ${tr}   （流水线应为 1；若为 2 说明有重复，需杀掉一个）"
echo " 看实时日志：tail -f $LOG"
echo " 完成后：$ROOT/05_结果/tables/table_v7_resolution_ablation_multiseed_test.csv"
echo "════════════════════════════════════════════════════════════════"
