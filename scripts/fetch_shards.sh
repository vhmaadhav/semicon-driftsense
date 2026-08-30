#!/usr/bin/env bash
# Parallel shard download + extract.  CPU phase only -- run this with the GPU
# idle, so the package has the whole thermal budget to itself.
#
#   ./scripts/fetch_shards.sh B 150        # 150 more Set B shards
#   ./scripts/fetch_shards.sh C 80 6       # 80 Set C shards, 6 workers
#
# Each worker streams one tar, extracts it, writes a COMPLETE marker and
# deletes the tar, so peak disk stays at ~(workers x 500 MB) above the pool
# rather than the size of everything queued. A shard that arrives truncated is
# deleted rather than half-added -- train.py would otherwise read it and fail
# mid-epoch on a missing PNG.
set -u
SET="${1:?usage: fetch_shards.sh <A|B|C|D> <count> [workers]}"
WANT="${2:?count required}"
NPROC="${3:-5}"
# Which split to pull, and where to put it. Defaults reproduce the original
# behaviour exactly. The `test` split is the only source of full 1000 px
# references -- `train` shards carry 100 px pre-cropped templates, which
# locate_phase2 cannot consume because it builds its own template.
SPLIT="${SPLIT:-train}"
DEST="${DEST:-data/ext_train}"

R="$(cd "$(dirname "$0")/.." && pwd)"
D="$R/$DEST"
IDX="$R/.agents/shards.tsv"
LOG="$R/.agents/fetch_${SPLIT}_${SET}.log"
mkdir -p "$D"

[ -s "$IDX" ] || { echo "missing $IDX -- run scripts/index_drive.sh first"; exit 1; }

# Queue the shards we do not already hold, newest ids last.
queue=$(mktemp)
awk -F'\t' -v s="$SET" -v sp="$SPLIT" '$1==sp && $2==s {print $3"\t"$4}' "$IDX" | while IFS=$'\t' read -r idx fid; do
  dir="$D/${SET}_$(printf %04d "$idx")"
  [ -d "$dir" ] || printf '%s\t%s\n' "$idx" "$fid"
done | head -n "$WANT" > "$queue"
total=$(wc -l < "$queue")
echo "[$(date +%H:%M)] queued $total $SPLIT shards of set $SET -> $DEST across $NPROC workers" | tee -a "$LOG"

fetch_one() {
  local idx="$1" fid="$2"
  local dir="$D/${SET}_$(printf %04d "$idx")" tar="$D/.${SET}_${idx}.tar"
  local url="https://drive.usercontent.google.com/download?id=${fid}&export=download&confirm=t"
  curl -sL --retry 2 --max-time 900 "$url" -o "$tar" || { rm -f "$tar"; return 1; }
  # Google serves an HTML interstitial instead of the file when it feels like it.
  if [ "$(stat -c%s "$tar" 2>/dev/null || echo 0)" -lt 100000 ]; then rm -f "$tar"; return 1; fi
  mkdir -p "$dir"
  tar xf "$tar" -C "$dir" 2>/dev/null || { rm -rf "$dir" "$tar"; return 1; }
  rm -f "$tar"
  local want have
  want=$(( $(wc -l < "$dir/manifest.csv" 2>/dev/null || echo 1) - 1 ))
  have=$(ls "$dir/search" 2>/dev/null | wc -l)
  if [ "$want" -gt 0 ] && [ "$have" -eq "$want" ]; then
    echo "$want pairs" > "$dir/COMPLETE"
    echo "[$(date +%H:%M)] + ${SET}_$(printf %04d "$idx")  pool=$(ls -d "$D"/*/ 2>/dev/null | wc -l)" >> "$LOG"
  else
    rm -rf "$dir"
    echo "[$(date +%H:%M)] ! ${SET}_$(printf %04d "$idx") truncated ($have/$want) - discarded" >> "$LOG"
  fi
}
export -f fetch_one; export D SET LOG

xargs -a "$queue" -P "$NPROC" -n 2 bash -c 'fetch_one "$0" "$1"'
rm -f "$queue"
echo "[$(date +%H:%M)] DONE set $SET  pool=$(ls -d "$D"/*/ 2>/dev/null | wc -l) shards, $(cat "$D"/*/manifest.csv 2>/dev/null | grep -vc '^pair_id') pairs" | tee -a "$LOG"
