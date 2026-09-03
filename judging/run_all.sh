#!/bin/bash
# Drive the whole constrained campaign: wait for generation to finish, then run
# the five 200-pair inference passes BACK-TO-BACK on an otherwise idle machine,
# scoring each as it lands.
#
# Serial by construction. Running two passes at once would halve the effective
# core count per pass and inflate every per-pair time -- the exact methodology
# error PR #51 had to discard readings for. Nothing else may hold cores here.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${DRIFTSENSE_PY:-./venv/bin/python}"
LOG=judging/run_all.log

# 4 DISTINCT physical cores. This CPU is heterogeneous, so the choice matters
# and is recorded: 0,2,4,6 are P-cores (4.8 GHz), 16-17,18,19 are E-cores (3.7 GHz).
PCORES="0,2,4,6"
ECORES="16,17,18,19"

echo "=== campaign start $(date -Is) ===" > "$LOG"

# If any set has fewer than 200 pairs and generator is not running, run generation
need_gen=0
for s in 1 2 3 4 5; do
  n=$(ls "judging/S$s/reference" 2>/dev/null | wc -l)
  if [ "$n" -ne 200 ]; then
    need_gen=1
  fi
done

if [ "$need_gen" -eq 1 ]; then
  if ! pgrep -f "gen_200.py --output-dir" > /dev/null; then
    echo "Starting generation..." >> "$LOG"
    judging/generate_all.sh >> "$LOG" 2>&1
  fi
fi

while pgrep -f "gen_200.py --output-dir" > /dev/null; do sleep 10; done
echo "generation complete $(date -Is)" >> "$LOG"

for s in 1 2 3 4 5; do
  n=$(ls "judging/S$s/reference" 2>/dev/null | wc -l)
  echo "S$s pairs: $n/200" >> "$LOG"
  if [ "$n" -ne 200 ]; then
    echo "ERROR: S$s is incomplete ($n/200)" >> "$LOG"
    exit 1
  fi
done

# Let the machine settle so the first pass is not measured against a page-cache
# storm left by the generator.
sleep 10

run_one () {  # <set> <cpus> <tag>
  local s="$1" cpus="$2" tag="$3"
  local out="judging/out/${tag}"
  echo "--- $tag start $(date -Is) ---" >> "$LOG"
  judging/run_judge.sh "judging/S$s" "$out" "$cpus" "$tag" >> "$LOG" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "$tag FAILED rc=$rc" >> "$LOG"; return $rc; fi
  $PY judging/score_rubric.py \
      --pred "$out/predictions.csv" \
      --truth "judging/S$s/ground_truth.csv" \
      --timing "$out/register.stderr" \
      --label "$tag" \
      --json-out "$out/rubric.json" > "$out/rubric.txt" 2>&1
  echo "$tag scored rc=$?" >> "$LOG"
}

for s in 1 2 3 4 5; do
  run_one "$s" "$PCORES" "S$s"
done

# Slower-box sensitivity: the same 200 pairs on 4 E-cores.
run_one 1 "$ECORES" "S1_ecore"

echo "ALL_RUNS_DONE $(date -Is)" >> "$LOG"
echo "=== AGGREGATE SUMMARY ===" >> "$LOG"
$PY judging/aggregate.py judging/out/S1/rubric.json judging/out/S2/rubric.json judging/out/S3/rubric.json judging/out/S4/rubric.json judging/out/S5/rubric.json --also judging/out/S1_ecore/rubric.json | tee -a "$LOG"
