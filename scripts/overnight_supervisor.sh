#!/usr/bin/env bash
# Unattended: wait for training, evaluate it on real data, decide whether more
# training is justified, and either continue or stop with a written verdict.
#
# Everything is sequential -- training and evaluation never overlap, so only one
# heat source is active at a time. Evaluation runs on the GPU (4.5x faster, and
# accuracy is device-independent); the CPU runtime figure that feeds the
# efficiency component is measured separately at the end.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/PHASES.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }

BASE_SETB=0.6986        # shipped weights, measured on ext_p2
PREV_SETB=0.7158        # the 8-epoch checkpoint

score() {   # score <weights> <outcsv>  -> prints "setB_credit total_points"
  ./venv-train/bin/python scripts/eval_ext.py \
     data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 \
     data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000 \
     --weights "$1" --jobs 3 --threads 2 --stride 1 --threshold 0.115 \
     --out "$2" > .agents/eval_$(basename "$2" .csv).log 2>&1
  ./venv-train/bin/python - "$2" <<'PY'
import sys, pandas as pd
sys.path.insert(0, "scripts")
from compare_ext import prep, summary
s = summary(prep(pd.read_csv(sys.argv[1])), 0.115)
print(f"{s['locB']:.4f} {s['pts']:.2f}")
PY
}

round=1
while [ "$round" -le 2 ]; do
  say "=== round $round: waiting for training to finish ==="
  while pgrep -f "train\.py --train-dirs" >/dev/null; do sleep 60; done
  ep=$(grep -cE "^epoch" ".agents/train_$([ "$round" -eq 1 ] && echo p6 || echo p7).log" 2>/dev/null || echo 0)
  say "training stopped after $ep epochs"
  grep -q ALERT .agents/WATCHDOG.txt 2>/dev/null && say "NOTE watchdog raised: $(grep ALERT .agents/WATCHDOG.txt | tail -1)"

  # Derive the checkpoint from the round, and prefer *_last.pt. Two traps here,
  # both hit on 2026-08-30: hardcoding p6 inside this loop made round 2 re-score
  # round 1's weights and "conclude" no improvement by comparing a checkpoint
  # against itself; and <name>.pt is the best-by-val file, which is meaningless
  # because val_p2 saturates at acc@2 1.000 in epoch 0 -- it is an epoch-0
  # checkpoint. _last.pt is the trained one.
  tag=$([ "$round" -eq 1 ] && echo p6 || echo p7)
  W="weights/driftsense_${tag}_last.pt"; [ -f "$W" ] || W="weights/driftsense_${tag}.pt"
  [ -f "$W" ] || { say "no checkpoint produced -- stopping"; break; }

  say "=== evaluating $W on the full 2500-pair set (gpu) ==="
  read -r setb pts < <(score "$W" ".agents/ext_r${round}.csv")
  say "RESULT round $round: Set B credit ${setb}   total ${pts} / 85"
  say "  reference: shipped ${BASE_SETB}, 8-epoch checkpoint ${PREV_SETB}"
  ./venv-train/bin/python scripts/optimize_threshold.py ".agents/ext_r${round}.csv" 2>&1 | tail -8 >> "$P"

  better=$(awk -v a="$setb" -v b="$PREV_SETB" 'BEGIN{print (a > b + 0.015) ? 1 : 0}')
  if [ "$better" = "1" ]; then
    say "VERDICT: Set B improved materially (${setb} vs ${PREV_SETB}). More training is justified."
    PREV_SETB="$setb"
    if [ "$round" -lt 2 ]; then
      say "=== fetching more Set C shards (rejection is the weak half) ==="
      ./scripts/fetch_shards.sh C 60 6 >> "$P" 2>&1 || say "download failed (Drive quota?) -- training on what we have"
      say "=== round $((round+1)) training ==="
      ./venv-train/bin/python train.py --train-dirs data/ext_train \
        --val-dir data/val_p2 --phase2 --val-limit 12 \
        --resume "$W" --finetune --lr 1e-4 --epochs 30 --samples-per-epoch 20000 \
        --workers 4 --batch-size 16 --device cuda --amp \
        --out weights/driftsense_p7.pt > .agents/train_p7.log 2>&1 &
      sleep 30
      setsid nohup ./scripts/watchdog.sh .agents/train_p7.log > /dev/null 2>&1 < /dev/null &
    fi
  else
    say "VERDICT: Set B did not improve materially (${setb} vs ${PREV_SETB}). More training is NOT justified -- stopping."
    break
  fi
  round=$((round+1))
done

say "=== CPU runtime check (efficiency component -- must be CPU, idle) ==="
./venv/bin/python scripts/profile_pair.py data/ext_p2/test_B_0000 --n 15 --threads 4 2>&1 | grep -E "median|budget" >> "$P"
say "ALLPHASESDONE"
