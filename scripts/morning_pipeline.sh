#!/usr/bin/env bash
# Finish the overnight work, correctly this time.
#
# The overnight supervisor had two faults: it hardcoded weights/driftsense_p6.pt
# inside its loop, so round 2 re-scored round 1's checkpoint and "concluded" no
# improvement; and driftsense_p6.pt is the best-by-*val* checkpoint, which is
# meaningless here because val_p2 saturated at acc@2 = 1.000 in epoch 0. The
# 73.36 it reported therefore came from a one-epoch checkpoint. The 40-epoch and
# 30-epoch weights have never been scored at all.
#
# Phases are strictly sequential and GPU work never overlaps CPU work.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/MORNING.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
: > "$P"

EVAL="data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000"

# ---------------------------------------------------------------- phase A: GPU
say "=== PHASE A (gpu): score every candidate checkpoint on 2500 held-out pairs ==="
for w in driftsense_p6 driftsense_p6_last driftsense_p7_last; do
  [ -f "weights/$w.pt" ] || { say "  skip $w (absent)"; continue; }
  say "  evaluating $w"
  ./venv-train/bin/python scripts/eval_ext.py $EVAL \
     --weights "weights/$w.pt" --jobs 3 --threads 2 --stride 1 --threshold 0.115 \
     --out ".agents/cand_$w.csv" > ".agents/eval_$w.log" 2>&1 \
     || say "  ! eval failed for $w (see .agents/eval_$w.log)"
done

say "=== ranking candidates (best threshold per candidate, full rubric) ==="
./venv-train/bin/python - <<'PY' | tee -a "$P"
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from optimize_threshold import prep, points

best = None
print(f"{'checkpoint':<26}{'best t':>9}{'total/85':>10}{'locA':>8}{'locB':>8}{'F1':>8}{'AUC':>8}")
print("-" * 77)
for f in sorted(glob.glob(".agents/cand_*.csv")):
    d = prep(pd.read_csv(f))
    s = d.score.values
    grid = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
    vals = [points(d, s, t) for t in grid]
    i = int(np.argmax(vals))
    b = points(d, s, float(grid[i]), breakdown=True)
    name = os.path.basename(f)[5:-4]
    print(f"{name:<26}{grid[i]:>9.4f}{b['total']:>10.2f}{b['locA']:>8.4f}"
          f"{b['locB']:>8.4f}{b['f1']:>8.4f}{b['auc']:>8.4f}")
    if best is None or b['total'] > best[1]:
        best = (name, b['total'], float(grid[i]))
print(f"\nWINNER {best[0]}  total {best[1]:.2f}/85  at threshold {best[2]:.4f}")
open(".agents/winner.txt", "w").write(f"{best[0]}\n{best[2]:.4f}\n")
PY

WIN=$(sed -n 1p .agents/winner.txt); THR=$(sed -n 2p .agents/winner.txt)
say "selected weights/$WIN.pt (threshold $THR)"

# ---------------------------------------------------------------- phase B: GPU
# Feature extraction for the rejector. fit_rejector.py's own extractor is a CPU
# multiprocessing pool and needs ~2.5 h for this many pairs; eval_ext.py now
# records the same six features and runs on the GPU, so this is ~18 min instead.
# Stride 5, not 4: manifests cycle severity 1,2,3,4 and a stride of 4 would
# sample severity 1 only -- that aliasing invalidated two earlier sweeps.
say "=== PHASE B (gpu): extract rejector features from 28 B + 28 C train shards ==="
SHARDS="$(ls -d data/ext_train/B_*/ | head -28) $(ls -d data/ext_train/C_*/)"
./venv-train/bin/python scripts/eval_ext.py $SHARDS \
   --weights "weights/$WIN.pt" --jobs 3 --threads 2 --stride 5 --threshold 0.115 \
   --out .agents/rejector_train.csv > .agents/extract_rejector.log 2>&1 \
   || say "  ! extraction failed (see .agents/extract_rejector.log)"
say "extracted $(( $(wc -l < .agents/rejector_train.csv) - 1 )) pairs"

# ---------------------------------------------------------------- phase C: CPU
say "=== PHASE C (cpu): fit the logistic rejector ==="
./venv/bin/python - <<'PY' | tee -a "$P"
# Convert the eval_ext CSV into the cache layout fit_rejector.py expects, so the
# fitting code that is already tested is reused rather than reimplemented.
import numpy as np, pandas as pd
d = pd.read_csv(".agents/rejector_train.csv")
d["found"] = d.gt_found
d["err"] = np.where(d.gt_found == 1, np.hypot(d.x - d.gt_x, d.y - d.gt_y), np.nan)
cols = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce", "found", "err"]
d[cols].to_csv(".agents/rejector_features_v2.csv", index=False)
print(f"cache: {len(d)} pairs, {int((d.found==1).sum())} present, {int((d.found==0).sum())} absent")
PY
./venv/bin/python scripts/fit_rejector.py data/ext_train/B_0000 \
   --cache .agents/rejector_features_v2.csv --out weights/rejector.json 2>&1 | tee -a "$P"

say "=== PHASE D (cpu): score the rejector against the full rubric on held-out data ==="
./venv/bin/python scripts/apply_rejector.py ".agents/cand_$WIN.csv" \
   --rejector weights/rejector.json 2>&1 | tee -a "$P"

say "=== PHASE E (cpu, idle): runtime for the efficiency component ==="
./venv/bin/python scripts/profile_pair.py data/ext_p2/test_B_0000 --n 15 --threads 4 \
   --weights "weights/$WIN.pt" 2>&1 | grep -E "median|budget" | tee -a "$P" \
   || ./venv/bin/python scripts/profile_pair.py data/ext_p2/test_B_0000 --n 15 --threads 4 2>&1 \
      | grep -E "median|budget" | tee -a "$P"

say "MORNINGDONE"
