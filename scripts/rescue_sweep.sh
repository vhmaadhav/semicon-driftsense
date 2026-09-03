#!/usr/bin/env bash
# Tune the margin-gated rescue pass (issue #5) on the full held-out set.
#
# Runs only after the wide training finishes: the sweep is CPU-bound (the coarse
# sweep is 66.8% of a pair and has no GPU path), and taking cores from the
# training dataloader would slow both.
#
# Every config uses --no-band, the decode measured at +0.439 (95% CI
# [+0.132, +0.767]) on the shipped weights -- comparing a new inference policy
# against a stale decode would attribute the decode's gain to the policy.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/RESCUE.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
: > "$P"

W="${RESCUE_WEIGHTS:-weights/driftsense_p9_last.pt}"
E="data/ext_p2/test_A_0000 data/ext_p2/test_A_0001 data/ext_p2/test_B_0000 data/ext_p2/test_B_0001 data/ext_p2/test_C_0000"
say "weights $W, full 2250 pairs, no-band decode"

run() {  # run <tag> <extra flags...>
  local tag="$1"; shift
  [ -f ".agents/cand_rescue_$tag.csv" ] && { say "  $tag cached"; return; }
  say "  evaluating $tag"
  ./venv-train/bin/python scripts/eval_ext.py $E --weights "$W" \
     --jobs 3 --threads 2 --stride 1 --threshold 0.2007 --no-band "$@" \
     --out ".agents/cand_rescue_$tag.csv" > ".agents/eval_rescue_$tag.log" 2>&1
}

run off
run m03 --rescue-margin 0.03
run m05 --rescue-margin 0.05
run m08 --rescue-margin 0.08
run m12 --rescue-margin 0.12
run m08d01 --rescue-margin 0.08 --rescue-delta 0.01

say "=== results (paired against rescue off) ==="
./venv-train/bin/python - <<'PY' | tee -a "$P"
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from optimize_threshold import prep, points

def load(tag):
    d = prep(pd.read_csv(f".agents/cand_rescue_{tag}.csv")).sort_values("pair_id").reset_index(drop=True)
    return d[d["set"].isin(["A", "B", "C"])].reset_index(drop=True)

def best(d):
    s = d.score.values
    g = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
    i = int(np.argmax([points(d, s, x) for x in g]))
    return float(g[i]), points(d, s, float(g[i]), breakdown=True)

tags = [t for t in ("off", "m03", "m05", "m08", "m12", "m08d01")
        if os.path.exists(f".agents/cand_rescue_{t}.csv")]
print(f"{'config':<10}{'best t':>9}{'total/85':>10}{'locA':>9}{'locB':>9}{'F1':>9}"
      f"{'fired%':>9}{'hyp':>7}{'s/pair':>9}")
print("-" * 81)
store = {}
for t in tags:
    d = load(t); th, b = best(d); store[t] = (d, th)
    fired = 100 * d.rescued.mean() if "rescued" in d.columns else float("nan")
    print(f"{t:<10}{th:>9.4f}{b['total']:>10.2f}{b['locA']:>9.4f}{b['locB']:>9.4f}"
          f"{b['f1']:>9.4f}{fired:>8.1f}%{d.n_hyp.mean():>7.2f}{d.secs.median():>9.2f}")

bd, bt = store["off"]
rng = np.random.RandomState(0); n = len(bd)
print()
for t in tags[1:]:
    d, th = store[t]
    diffs = []
    for _ in range(1000):
        i = rng.choice(n, n, replace=True)
        diffs.append(points(d.iloc[i], d.score.values[i], th)
                     - points(bd.iloc[i], bd.score.values[i], bt))
    a = np.array(diffs)
    gate = "PROMOTE" if a.mean() >= 0.35 else "below +0.35 gate"
    print(f"  {t:<8} paired delta {a.mean():+.3f}  95% CI [{np.percentile(a,2.5):+.3f}, "
          f"{np.percentile(a,97.5):+.3f}]  P(better) {100*(a>0).mean():5.1f}%  -> {gate}")
PY
say "RESCUEDONE"
