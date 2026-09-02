#!/usr/bin/env bash
set -u
cd /home/pranesh/Documents/semicon/semicon-driftsense
V=.agents/VERDICT.txt
for i in $(seq 1 900); do            # up to 15h
  if grep -q TONIGHTDONE .agents/TONIGHT.txt 2>/dev/null; then
    { echo "=== MORNING VERDICT  $(date '+%Y-%m-%d %H:%M') ==="
      echo
      ./venv-train/bin/python scripts/morning_verdict.py 2>&1
      echo
      echo "=== thermal / liveness overnight ==="
      awk '{print $2}' .agents/nightwatch.log | sed 's/pkg=//;s/C//' | sort -n | \
        awk '{a[NR]=$1} END{if(NR)printf "  package temp: min %s  median %s  max %s  (%d samples)\n",a[1],a[int(NR/2)+1],a[NR],NR}'
      grep -c HOT .agents/nightwatch.log 2>/dev/null | xargs -I{} echo "  samples at >=100C: {}"
      grep -c STALLED .agents/nightwatch.log 2>/dev/null | xargs -I{} echo "  stall warnings: {}"
      echo
      echo "=== epochs completed ==="
      for f in setcfull jw; do
        [ -f .agents/train_$f.log ] && echo "  $f: $(grep -cE '^epoch' .agents/train_$f.log) epochs"
      done
      echo
      echo "Nothing was promoted. weights/driftsense.pt is untouched."
    } > "$V" 2>&1
    exit 0
  fi
  sleep 60
done
echo "chain did not finish within 15h" > "$V"
