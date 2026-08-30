#!/usr/bin/env bash
# Watches the training run and stops it if it goes wrong, so a bad run does not
# burn the whole night. Writes one line per event to .agents/WATCHDOG.txt.
#
# Three failure modes it catches, all seen in practice on this project:
#   * process died   -- CUDA OOM, a truncated shard, an unhandled exception
#   * log stalled    -- alive but producing nothing for 15 min (deadlocked
#                       dataloader; a shard with a missing PNG did this once)
#   * loss diverged  -- NaN/inf, or loss climbing well above where it started,
#                       which means the LR is wrong and more epochs cannot help
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
LOG="${1:-.agents/train_p6.log}"
W=.agents/WATCHDOG.txt
STALL=900          # seconds without log growth before calling it stuck
say() { echo "[$(date '+%H:%M:%S')] $*" >> "$W"; }

say "watchdog started on $LOG"
start_loss=""
while true; do
  if ! pgrep -f "[t]rain\.py --train-dirs" >/dev/null; then
    grep -q "SFT3DONE\|TRAIN.*DONE" "$LOG" 2>/dev/null && say "training exited normally" \
      || say "ALERT training process gone without a completion marker"
    exit 0
  fi

  age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || date +%s) ))
  if [ "$age" -gt "$STALL" ]; then
    say "ALERT log stalled ${age}s -- killing training so the eval phase can still run"
    pkill -9 -f "[t]rain\.py --train-dirs"; exit 1
  fi

  line=$(grep -E "^  e[0-9]+ \[" "$LOG" 2>/dev/null | tail -1)
  loss=$(echo "$line" | grep -oE "loss [0-9.]+|loss nan|loss inf" | awk '{print $2}')
  if [ -n "$loss" ]; then
    case "$loss" in
      nan|inf) say "ALERT loss=$loss -- diverged, killing"; pkill -9 -f "[t]rain\.py --train-dirs"; exit 1;;
    esac
    [ -z "$start_loss" ] && { start_loss="$loss"; say "baseline loss $start_loss"; }
    bad=$(awk -v a="$loss" -v b="$start_loss" 'BEGIN{print (a > b*1.6) ? 1 : 0}')
    [ "$bad" = "1" ] && { say "ALERT loss $loss >> start $start_loss -- diverging, killing"; pkill -9 -f "[t]rain\.py --train-dirs"; exit 1; }
  fi
  sleep 60
done
