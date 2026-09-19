#!/bin/bash
# Unattended pipeline for the input-resolution x stem ablation.
#
# Stages, each with a hard gate that aborts the pipeline rather than producing
# unusable numbers:
#   1. wait for the MIRLab archive, verify it, extract the raw pickle
#   2. preprocess 128 x 128 (the script itself refuses on a split-manifest mismatch)
#   3. require the 64 x 64 control cells to reproduce the frozen validation values
#   4. train the 128 x 128 cells (2 stems x 3 seeds), standard stem first
#   5. evaluate every cell; the evaluator refuses to report new numbers unless
#      the frozen 64 x 64 anchors reproduce the published values
#
# Usage: bash 03_代码/scripts/run_resolution_ablation_pipeline.sh
set -u

ROOT="/Users/mima0000/论文111"
PY="/Users/mima0000/miniforge3/envs/wafer-sci/bin/python"
RAW="$ROOT/02_数据/raw/mirlab_2022"
ZIP="$RAW/MIR-WM811K.zip"
LOG="$ROOT/04_实验/logs/_session_runner/resolution_ablation_pipeline.log"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo "==================================================="
echo "pipeline start $(date '+%Y-%m-%d %H:%M:%S')"

abort() { echo "ABORT: $*"; echo "pipeline end $(date '+%Y-%m-%d %H:%M:%S')"; exit 1; }

# ---- stage 1: archive -------------------------------------------------------
echo "[1/5] waiting for $ZIP"
for _ in $(seq 1 240); do
  [ -f "$ZIP" ] && break
  sleep 120
done
[ -f "$ZIP" ] || abort "archive never appeared"
size=$(stat -f%z "$ZIP")
echo "archive size=$size"
unzip -t "$ZIP" > /dev/null || abort "archive failed its integrity test"
echo "archive integrity OK"
cd "$RAW" || abort "cannot enter $RAW"
unzip -oq "$ZIP" || abort "extraction failed"
PICKLE=$(find "$RAW" -name "*.pkl" -type f | head -1)
[ -n "$PICKLE" ] || abort "no .pkl inside the archive"
cp -f "$PICKLE" "$RAW/LSWMD.pkl" || abort "cannot stage LSWMD.pkl"
shasum -a 256 "$RAW/LSWMD.pkl" | tee "$RAW/SHA256SUMS.txt"
echo "extracted from: $PICKLE"

# ---- stage 2: preprocessing -------------------------------------------------
echo "[2/5] preprocessing 128 x 128 (split-manifest consistency is enforced)"
export WM811K_RAW_DIR="$RAW"
"$PY" "$ROOT/03_代码/scripts/preprocess_wm811k_at_size.py" --image-size 128 \
  || abort "preprocessing failed (see the manifest-mismatch message above)"
[ -f "$ROOT/02_数据/processed/wm811k_labeled_128x128/images_uint8.npy" ] \
  || abort "128 x 128 arrays were not produced"
echo "preprocessing OK"

# ---- stage 3: control gate --------------------------------------------------
echo "[3/5] waiting for the 64 x 64 control runs"
for _ in $(seq 1 120); do
  if [ -f "$ROOT/04_实验/metrics/shufflenet_v2_standard_res64_ce_full_history.csv" ] \
     && [ -f "$ROOT/04_实验/metrics/shufflenet_v2_highres_res64_ce_full_history.csv" ]; then
    break
  fi
  sleep 60
done
"$PY" - <<'PYEOF' || abort "control gate failed"
import sys
import pandas as pd

GATES = {
    "shufflenet_v2_standard_res64_ce_full": ("standard", 0.812563),
    "shufflenet_v2_highres_res64_ce_full": ("highres", 0.905377),
}
TOLERANCE = 0.002
ok = True
for run, (tag, expected) in GATES.items():
    path = f"04_实验/metrics/{run}_history.csv"
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        print(f"FAIL {tag}: missing {path}")
        ok = False
        continue
    column = [c for c in frame.columns if "val" in c.lower() and "macro" in c.lower()][0]
    best = float(frame[column].max())
    epoch = int(frame.loc[frame[column].idxmax(), "epoch"])
    delta = abs(best - expected)
    verdict = "PASS" if delta <= TOLERANCE else "FAIL"
    print(f"{verdict} {tag:9s} val_macro_f1={best:.6f} frozen={expected:.6f} "
          f"delta={delta:.6f} best_epoch={epoch}")
    ok = ok and delta <= TOLERANCE
sys.exit(0 if ok else 1)
PYEOF

# ---- stage 4: 128 x 128 training -------------------------------------------
echo "[4/5] training 128 x 128 cells (standard stem first)"
for stem in standard highres; do
  for seed in 42 123 2026; do
    echo "--- train res128 stem=$stem seed=$seed $(date '+%H:%M:%S')"
    /usr/bin/caffeinate -dimsu "$PY" \
      "$ROOT/03_代码/scripts/train_resolution_ablation.py" \
      --image-size 128 --stem "$stem" --seed "$seed"
    echo "--- done rc=$? $(date '+%H:%M:%S')"
  done
done

# ---- stage 5: evaluation ----------------------------------------------------
echo "[5/5] evaluating resolution-ablation cells"
"$PY" "$ROOT/03_代码/scripts/evaluate_resolution_ablation_multiseed.py" --execute

echo "pipeline end $(date '+%Y-%m-%d %H:%M:%S')"
