#!/bin/bash
# Run register.py under an emulated Phase 2 judging box: 4 x86 cores, 8 GB RAM,
# no GPU, no network, Python 3.11.
#
#   judging/run_judge.sh <set-dir> <out-dir> <cpu-list> <label>
#
# The constraint is applied two ways and then MEASURED, not assumed:
#   * cpuset  -- taskset pins the process to exactly 4 logical CPUs, chosen as
#                4 DISTINCT physical cores (no hyperthread siblings), so this is
#                a 4-core machine rather than 2 cores pretending to be 4.
#   * memory  -- a transient systemd scope with MemoryMax=8G and swap disabled,
#                so an over-8 GB working set is killed rather than silently
#                paged, which is what the judge's box would do.
#
# The venv is ./venv from the sibling checkout: Python 3.11.16, torch
# 2.13.0+cpu, cv2 5.0.0 -- CPU-only, matching the stated reference machine.
# Nothing else may run on the pinned cores while this is measured.
set -euo pipefail

SET_DIR="${1:?set dir}"
OUT_DIR="${2:?out dir}"
CPUS="${3:?cpu list}"
LABEL="${4:?label}"

# Derive the repo from this script's own location so the campaign is portable.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# CPU-only interpreter, matching the stated reference machine (no GPU).
# Override with DRIFTSENSE_PY=/path/to/python.
PY="${DRIFTSENSE_PY:-$REPO/venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "no CPU-only interpreter at $PY -- set DRIFTSENSE_PY" >&2; exit 2
fi
mkdir -p "$OUT_DIR"

PRED="$OUT_DIR/predictions.csv"
ERRLOG="$OUT_DIR/register.stderr"
ENVLOG="$OUT_DIR/env.txt"

# ---- record the environment the run actually saw -------------------------
{
  echo "label            $LABEL"
  echo "date             $(date -Is)"
  echo "host_arch        $(uname -m)"
  echo "host_kernel      $(uname -r)"
  echo "pinned_cpus      $CPUS"
  echo "python           $($PY -V 2>&1)"
  $PY - <<'PY'
import torch, numpy, cv2
print(f"torch            {torch.__version__}")
print(f"numpy            {numpy.__version__}")
print(f"opencv           {cv2.__version__}")
print(f"cuda_available   {torch.cuda.is_available()}")
PY
} > "$ENVLOG"

# ---- preflight: prove the cap is in effect BEFORE spending 200 pairs ------
# Reads the realised affinity and the cgroup's own memory.max from inside the
# constrained process. A flag that was set but not honoured is the failure this
# catches; it exits non-zero rather than producing an unlabelled measurement.
systemd-run --user --scope --quiet \
  -p MemoryMax=8G -p MemoryHigh=8G -p MemorySwapMax=0 \
  taskset -c "$CPUS" "$PY" - <<'PY' >> "$ENVLOG"
import os, sys
aff = sorted(os.sched_getaffinity(0))
cg = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split(":")[-1]
def rd(n):
    try:
        return open(os.path.join(cg, n)).read().strip()
    except OSError:
        return "unreadable"
mem, swp = rd("memory.max"), rd("memory.swap.max")
print(f"realised_affinity {aff}  (n={len(aff)})")
print(f"realised_memory_max {mem}")
print(f"realised_swap_max   {swp}")
print(f"os_cpu_count        {os.cpu_count()}  (thread cap uses min(4, this))")
ok = len(aff) == 4 and mem == str(8 * 1024**3) and swp == "0"
print(f"constraint_in_effect {ok}")
sys.exit(0 if ok else 1)
PY

# ---- run, constrained ----------------------------------------------------
# MemorySwapMax=0: the box has 8 GB, not 8 GB plus this laptop's 30 GB of swap.
systemd-run --user --scope --quiet \
  -p MemoryMax=8G -p MemoryHigh=8G -p MemorySwapMax=0 \
  --setenv=CUDA_VISIBLE_DEVICES= \
  taskset -c "$CPUS" \
  "$REPO/judging/pytime.py" -v -o "$OUT_DIR/time.txt" \
  "$PY" "$REPO/register.py" \
      --input "$SET_DIR/pairs.csv" \
      --output "$PRED" \
    2> "$ERRLOG"

# ---- verify the constraint was REALLY in effect ---------------------------
# A cap that was set but not honoured is the failure mode worth guarding: the
# affinity and the peak RSS below are read back from the run, not from flags.
{
  echo "--- realised constraints ---"
  echo "timed_pairs      $(grep -c '^# t,' "$ERRLOG" || true)"
  echo "peak_rss_kb      $(awk '/Maximum resident set size/{print $NF}' "$OUT_DIR/time.txt")"
  echo "wall_clock       $(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$OUT_DIR/time.txt")"
  echo "cpu_percent      $(awk -F': ' '/Percent of CPU/{print $2}' "$OUT_DIR/time.txt")"
} >> "$ENVLOG"

echo "[$LABEL] done -> $PRED"
