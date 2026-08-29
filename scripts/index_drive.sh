#!/usr/bin/env bash
# Rebuild .agents/shards.tsv from the public Drive folder.
# The normal folder URL server-renders only the first 50 entries; the embedded
# list view returns all of them in one plain page.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
FOLDER="${1:-1w5BoAvPIXQJH1gWfQQ8-ADsUSQ3J99tj}"
tmp=$(mktemp)
curl -sL "https://drive.google.com/embeddedfolderview?id=${FOLDER}#list" -o "$tmp"
python3 - "$tmp" "$R/.agents/shards.tsv" <<'PY'
import re, sys, collections
h = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
ent = re.findall(r'id="entry-([^"]+)".*?<div class="flip-entry-title">([^<]+)</div>', h, re.S)
rows = []
for fid, name in ent:
    m = re.search(r'_(train|test)_([ABCD])_(\d+)\.tar$', name)
    if m:
        rows.append((m.group(1), m.group(2), int(m.group(3)), fid, name))
rows.sort(key=lambda r: (r[0], r[1], r[2]))
open(sys.argv[2], "w").write("".join("\t".join(map(str, r)) + "\n" for r in rows))
c = collections.Counter((r[0], r[1]) for r in rows)
print(f"indexed {len(rows)} shards")
for k in sorted(c): print(f"  {k[0]}_{k[1]}: {c[k]}")
PY
rm -f "$tmp"
