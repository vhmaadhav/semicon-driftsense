#!/usr/bin/env bash
# Live view of the Hugging Face port. The uploader only prints when a whole
# shard lands, which at ~1 MB/s is every half hour, so the log looks frozen even
# when everything is fine. This reads the state file, the staging directory and
# the kernel's byte counters instead, so progress is visible continuously.
cd "$(dirname "$0")"
STATE=.agents/hf_port_state.tsv
LOG=.agents/hf_port.log
IF=$(ip -br addr | awk '$2=="UP" && $1!="lo" {print $1; exit}')
TOTAL=${1:-175}

bytes() { cat /sys/class/net/$IF/statistics/tx_bytes 2>/dev/null || echo 0; }

prev=$(bytes); prevt=$(date +%s)
while true; do
  sleep 3
  now=$(bytes); nowt=$(date +%s)
  dt=$(( nowt - prevt )); [ "$dt" -lt 1 ] && dt=1
  rate=$(( (now - prev) / dt ))            # bytes/sec
  prev=$now; prevt=$nowt

  done_n=$(grep -c "	done	" "$STATE" 2>/dev/null || echo 0)
  fail_n=$(grep -c "	failed	" "$STATE" 2>/dev/null || echo 0)
  left=$(( TOTAL - done_n ))
  pct=$(( done_n * 100 / (TOTAL>0?TOTAL:1) ))

  # 430 MB is the mean shard; good enough for an ETA that is honest about being
  # an estimate rather than pretending to know the tail.
  if [ "$rate" -gt 50000 ]; then
    eta_min=$(( left * 430 * 1024 * 1024 / rate / 60 ))
  else
    eta_min=-1
  fi

  clear
  printf '  HUGGING FACE PORT — local shards (no Drive calls)\n'
  printf '  %s\n\n' "$(date '+%H:%M:%S')"
  filled=$(( pct * 40 / 100 ))
  printf '  ['
  for i in $(seq 1 40); do [ "$i" -le "$filled" ] && printf '#' || printf '.'; done
  printf ']  %d%%\n' "$pct"
  printf '  uploaded %d / %d      failed %d      remaining %d\n\n' \
         "$done_n" "$TOTAL" "$fail_n" "$left"
  printf '  throughput   %s KB/s  (%s Mbit/s)\n' \
         "$(( rate / 1024 ))" "$(( rate * 8 / 1000000 ))"
  if [ "$eta_min" -ge 0 ]; then
    printf '  eta          ~%dh %dm\n' "$(( eta_min / 60 ))" "$(( eta_min % 60 ))"
  else
    printf '  eta          (stalled or between shards)\n'
  fi
  printf '  data sent    %s GB total on %s\n\n' \
         "$(( now / 1073741824 ))" "$IF"

  printf '  IN FLIGHT (%s staged, %s)\n' \
         "$(ls .hf_stage 2>/dev/null | wc -l)" "$(du -sh .hf_stage 2>/dev/null | cut -f1)"
  for f in .hf_stage/*.tar; do
    [ -e "$f" ] || { printf '    (none)\n'; break; }
    printf '    %-22s %6s MB\n' "$(basename "$f" .tar)" "$(( $(stat -c%s "$f") / 1048576 ))"
  done

  printf '\n  LAST COMPLETED\n'
  grep -E "^\[" "$LOG" 2>/dev/null | grep -E "OK|FAIL" | tail -4 | sed 's/^/    /'
  printf '\n  ctrl-c to stop watching (the upload keeps running)\n'
done
