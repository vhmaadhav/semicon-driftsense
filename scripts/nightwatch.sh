#!/usr/bin/env bash
# Passive overnight recorder: temps + training liveness. Never touches the run.
set -u
cd /home/pranesh/Documents/semicon/semicon-driftsense
L=.agents/nightwatch.log
: > "$L"
last_line=""; stall=0
while :; do
  pkg=$(sensors 2>/dev/null | awk -F'[+.]' '/Package id 0/{print $2; exit}')
  gpu=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null | tr -d ' ')
  cur=$( { tail -1 .agents/train_jw.log 2>/dev/null || true; tail -1 .agents/train_setcfull.log 2>/dev/null || true; } | tail -1)
  if [ "$cur" = "$last_line" ]; then stall=$((stall+1)); else stall=0; last_line="$cur"; fi
  flag=""
  [ -n "${pkg:-}" ] && [ "$pkg" -ge 100 ] 2>/dev/null && flag=" HOT"
  [ "$stall" -ge 10 ] && flag="$flag STALLED(${stall}x60s)"
  echo "$(date '+%H:%M:%S') pkg=${pkg}C gpu=${gpu}${flag}" >> "$L"
  grep -q TONIGHTDONE .agents/TONIGHT.txt 2>/dev/null && { echo "$(date '+%H:%M:%S') chain finished - watcher exiting" >> "$L"; break; }
  sleep 60
done
