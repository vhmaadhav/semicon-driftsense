#!/usr/bin/env bash
# Overnight chain, 2026-08-31. Strictly one phase at a time: the CPU-bound
# evaluation and the GPU-bound training never overlap, because the coarse sweep
# is 66.8% of a pair with no GPU path and would starve the training dataloader.
#
# Every phase writes its verdict to .agents/NIGHT.txt. Nothing is auto-shipped:
# weights/driftsense.pt stays where it is and each candidate has to clear the
# same gate -- paired bootstrap delta >= +0.35 on the full 2250 with no
# component regression.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/NIGHT.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
: > "$P"

BASE=75.96          # p9_last with the --no-band decode, the number to beat
EVAL="data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000"

# ---------------------------------------------------------- 1. wide training
say "PHASE 1  waiting for the 1.02M training to finish (GPU)"
while ! grep -q WIDEDONE .agents/WIDE.txt 2>/dev/null; do sleep 120; done
say "PHASE 1  done"
grep -E "0\.456M|1\.02M|paired delta|gate|median" .agents/WIDE.txt 2>/dev/null | sed 's/^/    /' | tee -a "$P"

# Pick whichever base is actually better; everything downstream builds on it.
BEST=weights/driftsense_p9_last.pt
if grep -q "PROMOTE" .agents/WIDE.txt 2>/dev/null && ! grep -q "DO NOT PROMOTE" .agents/WIDE.txt 2>/dev/null; then
  BEST=weights/driftsense_wide_last.pt
  say "  the 1.02M model cleared the gate -- it is the base for phase 4"
else
  say "  the 1.02M model did not clear the gate -- staying on p9 for phase 4"
fi

# ------------------------------------------------- 2. does the wide model overfit
if [ -f weights/driftsense_wide_last.pt ]; then
  say "PHASE 2  overfit probe on the 1.02M model (GPU) -- 2.24x capacity may"
  say "         overfit where 0.456M did not, which decides whether bulk data helps"
  ./venv-train/bin/python scripts/overfit_probe.py \
     --weights weights/driftsense_wide_last.pt --n 1600 2>&1 | tail -8 | sed 's/^/    /' | tee -a "$P"
fi

# ------------------------------------------------------- 3. rescue pass (CPU)
say "PHASE 3  rescue-pass sweep (CPU only, GPU idle)"
RESCUE_WEIGHTS="$BEST" ./scripts/rescue_sweep.sh > /dev/null 2>&1
grep -vE "^\[" .agents/RESCUE.txt 2>/dev/null | sed 's/^/    /' | tee -a "$P"

# --------------------------------------- 4. more set C, then a fine-tune (net, GPU)
say "PHASE 4  fetching Set C shards over the Drive API (network, GPU idle)"
say "         rejection F1 is 0.8893 against the 0.90 bonus threshold; Set C is"
say "         16.8% of the pool and we hold 28 of 198 shards"
./venv-hf/bin/python scripts/fetch_setc_api.py --set C --count 60 2>&1 | tail -4 | sed 's/^/    /' | tee -a "$P"
say "  pool now $(ls -d data/ext_train/*/ 2>/dev/null | wc -l) shards, $(df -h /home | awk 'NR==2{print $4}') free"

say "PHASE 5  fine-tuning $(basename "$BEST") on the expanded pool (GPU only)"
EXTRA=""
case "$BEST" in *wide*) EXTRA="--width 96 --ctx 48 --head 96";; esac
./venv-train/bin/python train.py --train-dirs data/ext_train \
   --val-dir data/val_p2 --phase2 --val-limit 12 $EXTRA \
   --resume "$BEST" --finetune --jitter-power -1 --ema 0.999 \
   --lr 8e-5 --epochs 22 --samples-per-epoch 30000 \
   --workers 6 --batch-size 32 --device cuda --amp \
   --out weights/driftsense_setc.pt > .agents/train_setc.log 2>&1
say "  training exited after $(grep -cE '^epoch' .agents/train_setc.log) epochs"

W=weights/driftsense_setc_last.pt
if [ -f "$W" ]; then
  say "PHASE 6  evaluating the Set C fine-tune on the full 2250 (no-band decode)"
  ./venv-train/bin/python scripts/eval_ext.py $EVAL --weights "$W" \
     --jobs 3 --threads 2 --stride 1 --threshold 0.2007 --no-band \
     --out .agents/cand_setc.csv > .agents/eval_setc.log 2>&1
  ./venv-train/bin/python - <<'PY' | tee -a "$P"
import sys; import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from optimize_threshold import prep, points
def load(f):
    d = prep(pd.read_csv(f)).sort_values("pair_id").reset_index(drop=True)
    return d[d["set"].isin(["A","B","C"])].reset_index(drop=True)
def best(d):
    s = d.score.values
    g = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
    i = int(np.argmax([points(d, s, x) for x in g]))
    return float(g[i]), points(d, s, float(g[i]), breakdown=True)
base, new = load(".agents/cand_p9_noband.csv"), load(".agents/cand_setc.csv")
tb, bb = best(base); tn, bn = best(new)
for n, t, b in (("p9 no-band (baseline)", tb, bb), ("Set C fine-tune", tn, bn)):
    print(f"    {n:<24} t={t:.4f}  total {b['total']:.2f}/85  locB {b['locB']:.4f}  "
          f"F1 {b['f1']:.4f}  AUC {b['auc']:.4f}")
print(f"    rejection F1 {bn['f1']:.4f} -- +4 bonus needs 0.90 -> "
      f"{'REACHED' if bn['f1'] >= 0.90 else 'not reached'}")
rng = np.random.RandomState(0); n = len(base); d = []
for _ in range(1000):
    i = rng.choice(n, n, replace=True)
    d.append(points(new.iloc[i], new.score.values[i], tn)
             - points(base.iloc[i], base.score.values[i], tb))
a = np.array(d)
print(f"    paired delta {a.mean():+.3f}  95% CI [{np.percentile(a,2.5):+.3f}, "
      f"{np.percentile(a,97.5):+.3f}]  P(better) {100*(a>0).mean():.1f}%  -> "
      f"{'PROMOTE' if a.mean() >= 0.35 else 'below the +0.35 gate'}")
PY
fi

say "ALLDONE  nothing was shipped; weights/driftsense.pt is untouched"
