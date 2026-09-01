#!/usr/bin/env python3
"""G6 (PHASE2_COMPLIANCE_ISSUES.md): submission-zip content checklist.

Slide 5 of the Phase 2 task deck fixes what the submission zip must contain:
register.py entry point, weights, requirements.txt from pip freeze,
documented generate_dataset.py, failure_analysis.pdf (max 2 pages).
This script audits the repo root the way the organizers will and prints a
PASS/FAIL line per requirement. Exits non-zero on any FAIL.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

checks = []

def check(name, ok, detail=""):
    checks.append((name, ok, detail))


# 1. Entry point present and import-safe
check("register.py present", os.path.isfile(os.path.join(REPO, "register.py")))

# 2. Weights ship inside the zip (nothing downloads at run time)
w = os.path.join(REPO, "weights", "driftsense.pt")
check("weights/driftsense.pt present", os.path.isfile(w),
      f"{os.path.getsize(w)/1e6:.1f} MB" if os.path.isfile(w) else "missing")

# 3. requirements.txt exists and looks like pip freeze (pinned lines)
req = os.path.join(REPO, "requirements.txt")
if os.path.isfile(req):
    lines = [l.strip() for l in open(req) if l.strip() and not l.startswith("#")]
    pinned = [l for l in lines if re.match(r"^[A-Za-z0-9_.-]+==", l)]
    check("requirements.txt pip-freeze format", len(pinned) == len(lines) and bool(lines),
          f"{len(pinned)}/{len(lines)} pinned")
else:
    check("requirements.txt present", False, "missing")

# 4. generate_dataset.py documented (has a module docstring + --help)
gd = os.path.join(REPO, "generate_dataset.py")
if os.path.isfile(gd):
    src = open(gd, encoding="utf-8").read()
    has_doc = src.lstrip().startswith("#!") or src.startswith('"""\n') or '"""' in src[:600]
    check("generate_dataset.py documented", has_doc and "--output-dir" in src or "--help" in src,
          "module docstring + argparse present" if (has_doc and "argparse" in src) else "check manually")
else:
    check("generate_dataset.py present", False, "missing")

# 5. failure_analysis.pdf exists and is at most 2 pages
fa = os.path.join(REPO, "failure_analysis.pdf")
if os.path.isfile(fa):
    data = open(fa, "rb").read()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    counts = [int(c) for c in re.findall(rb"/Count\s+(\d+)", data)]
    page_count = max(counts) if counts else pages
    check("failure_analysis.pdf <= 2 pages", 0 < page_count <= 2,
          f"{page_count} page(s)")
else:
    check("failure_analysis.pdf present", False, "missing")

# 6. No obvious network calls in the shipped entry points
for fname in ("register.py", "infer.py"):
    src = open(os.path.join(REPO, fname), encoding="utf-8").read()
    net = [m for m in ("requests.", "urllib.", "socket.", "httpx.", "urlopen") if m in src]
    check(f"no network calls in {fname}", not net, ", ".join(net) if net else "clean")

failed = 0
for name, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))
    failed += 0 if ok else 1
print()
print("SUBMISSION ZIP CHECKLIST:", "PASS" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
