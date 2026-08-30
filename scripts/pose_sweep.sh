#!/usr/bin/env bash
# Sweep the pose-hypothesis generator against Set B.
#
# The oracle says 58% of the remaining Set B gap is the pose search, and the
# failures have a median scale error of 6.65% against 0.60% on successes -- they
# are wrong-scale lock-ons. An earlier measurement found 76% of Set B failures
# never had a correct hypothesis generated at all, so the lever is *generating*
# candidates, not selecting among them: coarse scale count and hypothesis count.
#
# 43 is not an arbitrary top end. Template rasterisation makes only ~43 distinct
# magnifications realizable across [8,12]; asking for more returns duplicates.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
L=.agents/POSE_SWEEP.txt
: > "$L"
S="data/ext_p2/test_B_0000 data/ext_p2/test_B_0001"

run() {  # run <tag> <extra flags...>
  local tag="$1"; shift
  ./venv-train/bin/python scripts/eval_ext.py $S --jobs 3 --threads 2 --stride 3 \
     --threshold 0.1907 --out ".agents/sweep_$tag.csv" "$@" \
     > ".agents/sweep_$tag.log" 2>&1
  ./venv-train/bin/python - "$tag" ".agents/sweep_$tag.csv" <<'PY' >> "$L"
import sys, numpy as np, pandas as np_pd, pandas as pd
sys.path.insert(0, "scripts")
from eval_ext import LOC_TIERS, tier
tag, f = sys.argv[1], sys.argv[2]
d = pd.read_csv(f); d = d[d.gt_found == 1]
err = np.hypot(d.x - d.gt_x, d.y - d.gt_y)
cred = np.mean([tier(e, LOC_TIERS) for e in err])
print(f"{tag:<26}{len(d):>6}{cred:>10.4f}{(err<=5).mean()*100:>10.1f}%"
      f"{(err<=1).mean()*100:>9.1f}%{np.median(err):>9.2f}{d.secs.median():>9.2f}")
PY
}

printf '%-26s%6s%10s%10s%9s%9s%9s\n' "config" "n" "credit" "<=5px" "<=1px" "med" "s/pair" | tee -a "$L"
printf -- '-%.0s' {1..79}; echo | tee -a "$L"

run "baseline (17sc, 3hyp)"
run "coarse-29"            --coarse-scales 29
run "coarse-43"            --coarse-scales 43
run "hyp-4"                --hypotheses 4
run "hyp-4 + coarse-43"    --hypotheses 4 --coarse-scales 43
run "hyp-5 + coarse-43"    --hypotheses 5 --coarse-scales 43

cat "$L"
echo POSESWEEPDONE >> "$L"
