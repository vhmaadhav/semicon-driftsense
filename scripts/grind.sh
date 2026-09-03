#!/usr/bin/env bash
# Autonomous chain. Everything sequential; GPU work and CPU/network work never
# overlap, so only one heat source is ever active.
#
# The experiment being run is a three-arm test of the label-noise weighting on
# the offset head, all arms resumed from the same weights with an identical
# recipe so the only difference is --jitter-power:
#
#   p8   power +1   the drift part of the offset target is noise -> down-weight
#   p9   power -1   the drift part is learnable structure -> up-weight
#   p10  power  0   control: same 30 extra epochs, no weighting at all
#
# The control is not optional. p8 against p6_last would confound the weighting
# with the extra epochs, and a wrong attribution here costs the remaining days.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/GRIND.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
: > "$P"

BASE=weights/driftsense_p6_last.pt      # every arm starts here
EVAL="data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000"

rank() {   # rank <tag>  -> prints one scored line and appends to the table
  ./venv-train/bin/python - "$1" <<'PY' | tee -a "$P"
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from optimize_threshold import prep, points
tag = sys.argv[1]
d = prep(pd.read_csv(f".agents/cand_{tag}.csv"))
s = d.score.values
g = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
b = points(d, s, float(g[int(np.argmax([points(d, s, t) for t in g]))]), breakdown=True)
i = int(np.argmax([points(d, s, t) for t in g]))
print(f"  {tag:<26} total {b['total']:>6.2f}/85  t={g[i]:.4f}  locA {b['locA']:.4f}  "
      f"locB {b['locB']:.4f}  F1 {b['f1']:.4f}  AUC {b['auc']:.4f}")
PY
}

train_arm() {  # train_arm <tag> <power>
  local tag="$1" pw="$2"
  say "=== GPU: training $tag (jitter-power $pw), 30 epochs from $(basename $BASE) ==="
  ./venv-train/bin/python train.py --train-dirs data/ext_train \
     --val-dir data/val_p2 --phase2 --val-limit 12 \
     --resume "$BASE" --finetune --jitter-power "$pw" \
     --lr 8e-5 --epochs 30 --samples-per-epoch 30000 \
     --workers 4 --batch-size 16 --device cuda --amp \
     --out "weights/driftsense_${tag}.pt" > ".agents/train_${tag}.log" 2>&1
  local W="weights/driftsense_${tag}_last.pt"
  [ -f "$W" ] || { say "  ! no checkpoint for $tag"; return 1; }
  say "=== GPU: evaluating $tag on 2250 held-out pairs ==="
  ./venv-train/bin/python scripts/eval_ext.py $EVAL --weights "$W" \
     --jobs 3 --threads 2 --stride 1 --threshold 0.1907 \
     --out ".agents/cand_driftsense_${tag}_last.csv" > ".agents/eval_${tag}.log" 2>&1
  rank "driftsense_${tag}_last"
}

# ---- phase 1: let the p8 arm, already running, finish -----------------------
say "=== waiting for the p8 arm (already running) ==="
while pgrep -f "train\.py --train-dirs" >/dev/null || pgrep -f "labelnoise_run" >/dev/null; do sleep 120; done
say "p8 arm done"
grep -E "VERDICT|p8_last:" .agents/LABELNOISE.txt 2>/dev/null | tee -a "$P"

# ---- phase 2: CPU/network only, GPU idle ------------------------------------
say "=== CPU/network: attempting more shards (Drive quota permitting) ==="
FID=$(awk -F'\t' '$1=="train" && $2=="C" {print $4; exit}' .agents/shards.tsv)
probe=/tmp/grind_probe.bin
curl -sL --max-time 60 "https://drive.usercontent.google.com/download?id=${FID}&export=download&confirm=t" -o "$probe" 2>/dev/null
sz=$(stat -c%s "$probe" 2>/dev/null || echo 0); rm -f "$probe"
if [ "$sz" -gt 100000 ]; then
  say "quota is open (probe ${sz} bytes) -- fetching"
  # Set C first: absent pairs are the only training signal for rejection, which
  # holds 1.83 of the remaining points, and the pool is only 17% absent.
  ./scripts/fetch_shards.sh C 70 5 >> "$P" 2>&1 || say "  C fetch failed"
  ./scripts/fetch_shards.sh A 40 5 >> "$P" 2>&1 || say "  A fetch failed"
  ./scripts/fetch_shards.sh B 40 5 >> "$P" 2>&1 || say "  B fetch failed"
  say "pool now $(ls -d data/ext_train/*/ | wc -l) shards, $(df -h /home | awk 'NR==2{print $4}') free"
else
  say "Drive quota still exceeded (probe returned ${sz} bytes) -- training on the 167 shards we hold"
fi

# ---- phase 3: the two remaining arms, GPU ----------------------------------
train_arm p9 -1     # drift is learnable structure
train_arm p10 0     # control: same epochs, no weighting

# ---- phase 4: rank everything on the same 2250 pairs ------------------------
say "=== FINAL RANKING (same 2250 held-out pairs, each at its own best threshold) ==="
for t in driftsense_p6_last driftsense_p8_last driftsense_p9_last driftsense_p10_last; do
  [ -f ".agents/cand_${t}.csv" ] && rank "$t"
done
say "  p6_last is the shipped baseline. p8=+1 (drift is noise), p9=-1 (drift is"
say "  learnable), p10=0 (control: same extra epochs, no weighting)."
say "  Nothing is auto-shipped -- weights/driftsense.pt is untouched."

say "=== CPU, idle: runtime for the efficiency component ==="
./venv/bin/python scripts/profile_pair.py data/ext_p2/test_B_0000 --n 15 --threads 4 2>&1 \
  | grep -E "median|budget" | tee -a "$P"
say "GRINDDONE"
