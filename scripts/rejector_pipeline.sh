#!/usr/bin/env bash
# Fit the rejector on data locate_phase2 can actually consume.
#
# The first attempt fitted on data/ext_train shards and produced garbage: those
# shards carry reference_px=100 (pre-cropped templates), while locate_phase2
# builds its own template from a *full* reference. Median localisation error on
# them was 600 px and not one present pair landed within 5 px, so every feature
# fed to the logistic was noise. Only the `test` split has 1000 px references.
#
# CPU/network phase first (download), then GPU (feature extraction), then CPU
# (fit + validate). Never overlapping.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
P=.agents/REJECTOR.txt
say() { echo "[$(date '+%H:%M')] $*" | tee -a "$P"; }
: > "$P"

WIN=driftsense_p6_last
FIT=data/ext_fit
mkdir -p "$FIT"

# Skip the indices data/ext_p2 already holds, so the fit set cannot overlap the
# evaluation set. Verified again by sha256 below -- the index lists two
# generator bundles under the same shard number, so name-matching alone is not
# proof.
for d in A_0000 A_0001 B_0000 B_0001 C_0000; do mkdir -p "$FIT/$d"; done

say "=== PHASE 1 (cpu/network): fetch full-reference test shards ==="
for spec in "A 3" "B 3" "C 2"; do
  set -- $spec
  SPLIT=test DEST="$FIT" ./scripts/fetch_shards.sh "$1" "$2" 4 >> "$P" 2>&1 \
    || say "  ! fetch failed for set $1"
done
for d in A_0000 A_0001 B_0000 B_0001 C_0000; do rmdir "$FIT/$d" 2>/dev/null; done

say "=== PHASE 2 (cpu): verify the fit set is disjoint from the evaluation set ==="
./venv/bin/python - <<'PY' | tee -a "$P"
import glob, os, shutil
import pandas as pd
ev = set()
for m in glob.glob("data/ext_p2/*/manifest.csv"):
    ev |= set(pd.read_csv(m).pair_sha256)
kept = []
for d in sorted(glob.glob("data/ext_fit/*/")):
    f = os.path.join(d, "manifest.csv")
    if not os.path.exists(f):
        shutil.rmtree(d, ignore_errors=True); continue
    m = pd.read_csv(f)
    ov = len(ev & set(m.pair_sha256))
    if ov:
        print(f"  DISCARD {d} -- {ov} pairs overlap the evaluation set")
        shutil.rmtree(d, ignore_errors=True)
    else:
        kept.append((d, len(m), int((m.found == 1).sum()), int(m.reference_px.iloc[0])))
tot = sum(k[1] for k in kept); pres = sum(k[2] for k in kept)
for d, n, p, rp in kept:
    print(f"  keep {d:<28} n={n} present={p} reference_px={rp}")
print(f"\nfit set: {tot} pairs, {pres} present, {tot-pres} absent")
bad = [d for d, n, p, rp in kept if rp != 1000]
if bad:
    print(f"FATAL: reference_px != 1000 in {bad} -- locate_phase2 cannot use these")
open(".agents/fit_ok.txt", "w").write("0" if (bad or tot < 1000 or pres in (0, tot)) else "1")
PY

[ "$(cat .agents/fit_ok.txt 2>/dev/null)" = "1" ] || { say "fit set unusable -- stopping"; say "REJECTORDONE"; exit 1; }

say "=== PHASE 3 (gpu): extract features with weights/$WIN.pt ==="
./venv-train/bin/python scripts/eval_ext.py $(ls -d "$FIT"/*/) \
   --weights "weights/$WIN.pt" --jobs 3 --threads 2 --stride 1 --threshold 0.1907 \
   --out .agents/rejector_fit.csv > .agents/extract_fit.log 2>&1 \
   || say "  ! extraction failed (see .agents/extract_fit.log)"

say "=== PHASE 4 (cpu): sanity-check the features before fitting ==="
./venv/bin/python - <<'PY' | tee -a "$P"
import numpy as np, pandas as pd
d = pd.read_csv(".agents/rejector_fit.csv")
d["found"] = d.gt_found
d["err"] = np.where(d.gt_found == 1, np.hypot(d.x - d.gt_x, d.y - d.gt_y), np.nan)
pres = d[d.found == 1]
within = float((pres.err <= 5).mean())
print(f"{len(d)} pairs, {len(pres)} present; within 5px {within:.1%}, median {pres.err.median():.2f}px")
# The check the first attempt lacked. If localisation is at chance the features
# are noise and the fit is meaningless, however good its F1 looks.
print("GUARD:", "ok" if within > 0.5 else "FAIL -- localisation at chance, features are noise")
cols = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce", "found", "err"]
d[cols].to_csv(".agents/rejector_features_v3.csv", index=False)
PY
grep -q "GUARD: ok" "$P" || { say "feature guard failed -- not fitting"; say "REJECTORDONE"; exit 1; }

say "=== PHASE 5 (cpu): fit and validate against the full rubric ==="
./venv/bin/python scripts/fit_rejector.py "$FIT" \
   --cache .agents/rejector_features_v3.csv --out weights/rejector.json 2>&1 | tee -a "$P"
./venv/bin/python scripts/apply_rejector.py ".agents/cand_$WIN.csv" \
   --rejector weights/rejector.json 2>&1 | tee -a "$P"

say "REJECTORDONE"
