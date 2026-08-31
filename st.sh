#!/usr/bin/env bash
# One-shot status. The phase scripts only write their verdict at the end, so
# progress has to be read from the per-run logs instead of the marker files.
cd "$(dirname "$0")"
printf '\n  %s\n\n' "$(date '+%H:%M:%S')"
printf '  PHASE\n'
if pgrep -f "[t]rain\.py --train-dir" >/dev/null; then
  ep=$(grep -cE "^epoch" .agents/train_wide.log 2>/dev/null || echo 0)
  printf '    training 1.02M model   epoch %s/34\n' "$ep"
  tail -1 .agents/train_wide.log 2>/dev/null | sed 's/^/      /'
elif pgrep -f "[e]val_ext.py" >/dev/null; then
  printf '    evaluating\n'
  for f in .agents/eval_p9_band.log .agents/eval_p9_noband.log .agents/eval_wide.log; do
    [ -f "$f" ] || continue
    p=$(grep -oE '[0-9]+/[0-9]+' "$f" 2>/dev/null | tail -1)
    [ -n "$p" ] && printf '      %-26s %s\n' "$(basename "$f")" "$p"
  done
else
  printf '    nothing running\n'
fi
printf '\n  GPU   %s\n' "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader)"
printf '  load  %s\n\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
printf '  RESULTS SO FAR\n'
for f in .agents/BAND_AB.txt .agents/WIDE.txt; do
  if [ -s "$f" ]; then printf '    --- %s ---\n' "$(basename "$f")"; sed 's/^/    /' "$f"; fi
done
[ -s .agents/BAND_AB.txt ] || printf '    band A/B: still running, writes its verdict at the end\n'
echo
