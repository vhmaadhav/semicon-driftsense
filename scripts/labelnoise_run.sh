#!/usr/bin/env bash
# Train with the label-noise-weighted offset loss and ship only if it wins.
#
# The hypothesis is specific and measured: localisation error is 0.72x the
# frame's raster drift, so the sub-pixel target carries per-pair noise spanning
# a 6x range in variance (drift_jitter_px 0.34 -> 2.10). offset_loss now weights
# each pair by sigma0^2/(sigma0^2+sigma_i^2), normalised to mean one, on the
# offset head only. If the model stops fitting drift it cannot predict, the
# <=1px tier should improve; if the hypothesis is wrong this run loses nothing
# because the gate below refuses to ship it.
#
# Gate: set B localisation credit on the full held-out set must beat the
# shipped 0.7689. Ties do not ship -- a wash is not worth a weights change this
# close to the deadline.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/LABELNOISE.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
: > "$P"

SHIPPED_LOCB=0.7689
SHIPPED_TOTAL=75.35

say "=== GPU: 30 epochs from p6_last with the label-noise-weighted offset loss ==="
./venv-train/bin/python train.py --train-dirs data/ext_train \
   --val-dir data/val_p2 --phase2 --val-limit 12 \
   --resume weights/driftsense_p6_last.pt --finetune \
   --lr 8e-5 --epochs 30 --samples-per-epoch 30000 \
   --workers 4 --batch-size 16 --device cuda --amp \
   --out weights/driftsense_p8.pt > .agents/train_p8.log 2>&1
say "training exited after $(grep -cE '^epoch' .agents/train_p8.log) epochs"

# _last.pt, never .pt: val_p2 saturates in epoch 0 so the best-by-val file is an
# epoch-0 checkpoint. This is the trap that made the overnight run report a
# one-epoch result as if it were trained.
W=weights/driftsense_p8_last.pt
[ -f "$W" ] || { say "no checkpoint produced -- nothing to evaluate"; say "LABELNOISEDONE"; exit 1; }

say "=== GPU: evaluate $W on the 2250 held-out A/B/C pairs ==="
./venv-train/bin/python scripts/eval_ext.py \
   data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 \
   data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000 \
   --weights "$W" --jobs 3 --threads 2 --stride 1 --threshold 0.1907 \
   --out .agents/cand_driftsense_p8_last.csv > .agents/eval_p8.log 2>&1

./venv-train/bin/python - <<'PY' | tee -a "$P"
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from optimize_threshold import prep, points
d = prep(pd.read_csv(".agents/cand_driftsense_p8_last.csv"))
s = d.score.values
grid = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
i = int(np.argmax([points(d, s, t) for t in grid]))
b = points(d, s, float(grid[i]), breakdown=True)
print(f"p8_last: total {b['total']:.2f}/85 at t={grid[i]:.4f}  "
      f"locA {b['locA']:.4f}  locB {b['locB']:.4f}  F1 {b['f1']:.4f}  AUC {b['auc']:.4f}")
print(f"shipped: total 75.35/85 at t=0.1907  locA 0.9680  locB 0.7689  F1 0.8779  AUC 0.9873")
open(".agents/p8_locb.txt", "w").write(f"{b['locB']:.4f} {b['total']:.2f} {grid[i]:.4f}")
PY

read -r locb total thr < .agents/p8_locb.txt
win=$(awk -v a="$locb" -v b="$SHIPPED_LOCB" 'BEGIN{print (a > b) ? 1 : 0}')
if [ "$win" = "1" ]; then
  say "VERDICT: label-noise weighting WINS (set B $locb vs $SHIPPED_LOCB, total $total vs $SHIPPED_TOTAL)."
  say "  NOT auto-shipped -- weights/driftsense.pt is left alone for a human to confirm."
  say "  To ship:  cp weights/driftsense_p8_last.pt weights/driftsense.pt   (threshold $thr)"
else
  say "VERDICT: label-noise weighting does NOT win (set B $locb vs $SHIPPED_LOCB, total $total vs $SHIPPED_TOTAL)."
  say "  Keeping the shipped p6_last weights. The loss change stays in the tree; it is"
  say "  correct and tested, it just did not pay on this data."
fi
say "LABELNOISEDONE"
