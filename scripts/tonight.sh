#!/usr/bin/env bash
# Overnight chain, 2026-09-01. Arm A (setc completion) is already running as PID 58242.
set -u
cd /home/pranesh/Documents/semicon/semicon-driftsense
P=.agents/TONIGHT.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
EVAL="data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000"
: > "$P"

say "waiting on Arm A (setc completion, 22 epochs) pid 58242"
while kill -0 58242 2>/dev/null; do sleep 60; done
say "Arm A finished after $(grep -cE '^epoch' .agents/train_setcfull.log) epochs"

if [ -f weights/driftsense_setcfull_last.pt ]; then
  say "eval Arm A (no-band, shipped decode)"
  ./venv-train/bin/python scripts/eval_ext.py $EVAL --weights weights/driftsense_setcfull_last.pt \
     --no-band --features --jobs 3 --out .agents/feat_setcfull.csv > .agents/eval_setcfull.log 2>&1
  grep -E "SUBTOTAL|set A |set B |Rejection|best-possible" .agents/eval_setcfull.log | tee -a "$P"
else
  say "Arm A produced no checkpoint -- skipping its eval"
fi

say "=== Arm B: same recipe + jitter-weighted sampler (p=1) ==="
./venv-train/bin/python train.py --train-dirs data/ext_train \
   --val-dir data/val_p2 --phase2 --val-limit 12 --width 96 --ctx 48 --head 96 \
   --resume weights/driftsense_wide_last.pt --finetune --jitter-power -1 --ema 0.999 \
   --sampler-jitter-power 1.0 \
   --lr 8e-5 --epochs 22 --samples-per-epoch 30000 \
   --workers 6 --batch-size 32 --device cuda --amp \
   --out weights/driftsense_jw.pt > .agents/train_jw.log 2>&1
say "Arm B finished after $(grep -cE '^epoch' .agents/train_jw.log) epochs"

if [ -f weights/driftsense_jw_last.pt ]; then
  say "eval Arm B"
  ./venv-train/bin/python scripts/eval_ext.py $EVAL --weights weights/driftsense_jw_last.pt \
     --no-band --features --jobs 3 --out .agents/feat_jw.csv > .agents/eval_jw.log 2>&1
  grep -E "SUBTOTAL|set A |set B |Rejection|best-possible" .agents/eval_jw.log | tee -a "$P"
fi

say "=== CV comparison of every candidate (threshold fitted per checkpoint) ==="
for c in feat_base_nb feat_setc_nb feat_setcfull feat_jw; do
  [ -f .agents/$c.csv ] || continue
  echo "--- $c ---" | tee -a "$P"
  ./venv-train/bin/python /tmp/claude-1000/-home-pranesh-Documents-semicon/8e7a27b9-eaf7-4428-bba5-a3f90558a8d0/scratchpad/nl_rejector.py \
     .agents/$c.csv 2>&1 | grep -E "shipped min" | tee -a "$P"
done
say "TONIGHTDONE"
