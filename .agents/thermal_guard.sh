#!/bin/bash
# Thermal safety net for unattended runs.
#
# Pauses (SIGSTOP) the scene generators when the CPU package gets close to its
# 100C limit and resumes them once it has cooled. Generation is pure compute
# holding no locks, so stopping it mid-flight is safe -- the parent simply
# waits. Training on the GPU is deliberately left alone: it is the long pole,
# and the GPU has its own headroom.
HOT=93        # pause at/above this package temp
COOL=84       # resume at/below this
LOG=.agents/thermal.log
paused=0
pkg() { sensors 2>/dev/null | awk -F'[+.]' '/Package id 0/{print $2; exit}'; }
gens() { ps -eo pid,args | grep "[g]en_data.py" | awk '{print $1}'; }

while true; do
  t=$(pkg); [ -z "$t" ] && { sleep 30; continue; }
  ps=$(gens)
  if [ -z "$ps" ]; then
    [ "$paused" = 1 ] && paused=0
    sleep 60; continue
  fi
  if [ "$paused" = 0 ] && [ "$t" -ge "$HOT" ]; then
    echo "$(date +%H:%M:%S) ${t}C >= ${HOT}C - pausing generation" >> $LOG
    echo "$ps" | xargs -r kill -STOP 2>/dev/null; paused=1
  elif [ "$paused" = 1 ] && [ "$t" -le "$COOL" ]; then
    echo "$(date +%H:%M:%S) ${t}C <= ${COOL}C - resuming generation" >> $LOG
    echo "$ps" | xargs -r kill -CONT 2>/dev/null; paused=0
  fi
  echo "$(date +%H:%M:%S) pkg=${t}C paused=${paused}" >> $LOG
  sleep 30
done
