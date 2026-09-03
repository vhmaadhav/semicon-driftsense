"""phase1/ archive isolation: nothing outside it may import or path-load it.

phase1/ is a frozen historical snapshot of the pre-Phase-2 repo (see the
promote-phase2-to-main restructuring). test_checkpoint_safety.py excludes the whole phase1/ subtree from its
unrestricted-pickle-loading scan on the premise that phase1/ is never
reachable from anything at root -- that exclusion is only correct as long
as this test passes. If it ever fails, the correct fix is to remove the
dependency on phase1/, not to loosen this test.
"""

from __future__ import annotations

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMPORT_PATTERN = re.compile(r"^\s*(import\s+phase1(\.\w+)*\b|from\s+phase1(\.\w+)*\s+import)")
SYS_PATH_PATTERN = re.compile(r"sys\.path\.(insert|append)\([^)]*phase1")


def test_no_root_code_imports_phase1():
    bad = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "phase1", ".venv")
                       and not d.startswith("venv")
                       and not os.path.exists(os.path.join(dirpath, d, "pyvenv.cfg"))]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            if path == os.path.abspath(__file__):
                continue  # this scanner's own source quotes the pattern
            rel = os.path.relpath(path, REPO_ROOT)
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if IMPORT_PATTERN.match(line) or SYS_PATH_PATTERN.search(line):
                        bad.append(f"{rel}:{lineno}: {line.strip()}")
    assert bad == [], (
        "root/Phase-2 code must not import or path-load the frozen phase1/ "
        "archive: " + ", ".join(bad))
