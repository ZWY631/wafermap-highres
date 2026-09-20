#!/bin/bash
# 守望任务：等 highres@128 的 seed42 跑满 30 轮后，自动只评这一个 seed，
# 并与冻结的 HighRes@64 对比，给出"是否已饱和"的判断。
# 用途：把上限检查的第一个数据点自动送到手上，不必人工盯。
set -u
ROOT="/Users/mima0000/论文111"
PY="/Users/mima0000/miniforge3/envs/wafer-sci/bin/python"
HIST="$ROOT/04_实验/metrics/shufflenet_v2_highres_res128_ce_full_history.csv"
LOG="$ROOT/04_实验/logs/_session_runner/highres128_seed42_checkpoint.log"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "=================================================="
echo "watcher start $(date '+%Y-%m-%d %H:%M:%S')"
echo "waiting for 30 epochs in $HIST"

for _ in $(seq 1 576); do   # 最多 48 小时
  if [ -f "$HIST" ]; then
    done_epochs=$("$PY" - <<PYEOF
import pandas as pd
try:
    print(len(pd.read_csv("$HIST")))
except Exception:
    print(0)
PYEOF
)
    [ "$done_epochs" -ge 30 ] && break
  fi
  sleep 300
done

epochs=$("$PY" -c "
import pandas as pd
print(len(pd.read_csv('$HIST')))" 2>/dev/null || echo 0)
echo "epochs now: $epochs at $(date '+%H:%M:%S')"
if [ "$epochs" -lt 30 ]; then
  echo "ABORT: seed42 never reached 30 epochs"
  exit 1
fi

echo "--- evaluating R128-S1N seed 42 ---"
"$PY" "$ROOT/03_代码/scripts/evaluate_resolution_ablation_multiseed.py" \
  --execute --cells R128-S1N --seeds 42

echo "--- comparison against the frozen HighRes@64 ---"
"$PY" - <<'PYEOF'
import pandas as pd
frozen = pd.read_csv("05_结果/tables/table_stem_five_config_multiseed_test.csv").set_index(
    "configuration_id"
)
h64 = float(frozen.loc["S1N", "macro_f1_mean"]) * 100.0
h64_scratch = None
per_class = pd.read_csv("05_结果/tables/table_stem_per_class_f1_five_config.csv")
h64_scratch = float(per_class.loc[per_class.class_name == "Scratch", "S1N"].iloc[0]) * 100.0

rows = pd.read_csv("04_实验/metrics/20260914_resolution_ablation_multiseed_test/per_seed_metrics.csv")
row = rows[(rows.cell_id == "R128-S1N") & (rows.seed == 42)]
if row.empty:
    print("no R128-S1N seed42 record; evaluation did not produce it")
    raise SystemExit
row = row.iloc[0]
m = row.macro_f1 * 100
s = row.scratch_f1 * 100
print(f"  highres@128 seed42 : Macro-F1 {m:.4f}  Scratch {s:.4f}")
print(f"  frozen highres@64  : Macro-F1 {h64:.4f}  Scratch {h64_scratch:.4f}")
print(f"  delta              : Macro-F1 {m - h64:+.4f} pp  Scratch {s - h64_scratch:+.4f} pp")
if m - h64 <= 0.5:
    print("  VERDICT: saturated - the 64 x 64 operating point is near-optimal.")
else:
    print("  VERDICT: still improving - more resolution helps; run all three seeds "
          "and discuss the 64 x 64 choice in the paper.")
PYEOF
echo "watcher end $(date '+%Y-%m-%d %H:%M:%S')"
