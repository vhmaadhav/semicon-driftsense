#!/usr/bin/env bash
# Live status of Drift-Sense work.   Run: ./status.sh      Watch: watch -n10 ./status.sh
cd "$(dirname "$0")"
echo "================= $(date '+%H:%M:%S') ================="
echo "RUNNING:"
ps -eo etime=,pcpu=,comm=,args= | grep -E "train\.py|eval_ext|fit_rejector|verify_scores|row_shift" | grep -v grep \
  | awk '{printf "  %-9s %6s%%  %s\n", $1, $2, $5}' | sed 's|.*/python ||;s|scripts/||' | sort -u | head -6
ps -eo args= | grep -qE "train\.py|eval_ext|fit_rejector" || echo "  (idle)"
echo
echo "TRAINING  (target: beat Set B credit 0.7158 on real data)"
grep -E "^epoch" .agents/train_p4.log 2>/dev/null | tail -4 | sed 's/^/  /'
tail -1 .agents/train_p4.log 2>/dev/null | grep -E "^  e" | sed 's/^/  /'
echo
echo "HARDWARE:  gpu $(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null)   cpu $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"
echo
echo "SCORE (full 2,250-pair A/B/C hold-out, corrected grader semantics):"
echo "  shipped (band=False)  75.92 / 85   (PR #18 campaign; see .agents/INFERENCE_TWEAKS.md)"
echo "  historical: 72.55/95 (pre-Phase-2 readout), 76.23/85 (pre-#22 unmasked scorer)"
