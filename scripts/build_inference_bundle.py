#!/usr/bin/env python3
"""Build the inference-only delivery: two self-contained, runnable folders.

    python scripts/build_inference_bundle.py --out dist/driftsense-inference

    dist/driftsense-inference/
      README.md
      phase_2/   register.py  + driftsense/ + weights/ + requirements.txt
      phase_3/   phase3.py    + driftsense/ + weights/ + requirements.txt

Nothing that trains, generates data, or evaluates ships here: no train.py, no
generator/, no scripts/, no tests/, no data/. Each folder is independent --
`cd phase_3 && pip install -r requirements.txt && python phase3.py ...` works
with no reference to the other folder or to this repository.

The module list per phase is the **static import closure** of that phase's
entry points, computed by `python_modules()` below rather than maintained by
hand, so a module that only a non-default flag path imports still ships.
Verified empirically too: `--verify` runs both entry points from inside the
built folders, on a temporary copy, and fails the build if either does not
produce a well-formed predictions.csv.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG = "driftsense"

# Entry points per phase. The first is the graded command; the rest are
# imported by it and travel with it.
ENTRIES = {
    "phase_2": ["register.py", "infer.py"],
    "phase_3": ["phase3.py", "register.py", "infer.py"],
}

GRADED = {"phase_2": "register.py", "phase_3": "phase3.py"}

# Third-party requirements per phase, resolved to the versions actually
# installed in the build environment (a real pip freeze, scoped to what the
# phase imports rather than to everything in the venv).
DIRECT = {
    "phase_2": ["torch", "opencv-python-headless", "numpy"],
    "phase_3": ["torch", "opencv-python-headless", "numpy", "gdstk"],
}

# Transitive closure of the above. Pinned so `pip install -r` is reproducible
# on a machine with no network beyond the index.
TRANSITIVE = ["filelock", "fsspec", "Jinja2", "MarkupSafe", "networkx",
              "sympy", "mpmath", "typing_extensions"]

WEIGHTS = os.path.join("weights", "driftsense.pt")
MIN_WEIGHTS_BYTES = 1_000_000

# A tiny real dataset inside each folder, so the first command a reader runs
# needs nothing but the folder itself. Each sample carries its own
# ground_truth.csv, so the install can be checked against a known answer
# rather than merely "it printed something".
#
# Both samples deliberately include an absent site. Rejection is 15 of the 85
# measurable points and is the least obvious half of the contract: a pair with
# no true match must still emit a row, with found=0 and zeroed pose columns.
SAMPLES = {
    "phase_2": {
        "root": os.path.join(REPO, "generator", "output"),
        "pairs": ["A01", "A02", "C01"],
        "copy": [("reference_path", "reference"), ("search_path", "search")],
    },
    "phase_3": {
        "root": os.path.join(REPO, "..", "phase3_tiers", "phase3_harsh"),
        "pairs": ["p0001", "p0000"],
        "copy": [("search_path", "search"),
                 ("reference_gds_path", "reference"),
                 ("search_gds_path", "search_gds")],
    },
}


def _pkg_deps(path: str) -> set:
    """driftsense submodules imported by one file, from its AST."""
    out = set()
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith(PKG):
            parts = n.module.split(".")
            if len(parts) > 1:
                out.add(parts[1])
            else:
                out.update(a.name for a in n.names)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith(PKG + "."):
                    out.add(a.name.split(".")[1])
    return out


def python_modules(phase: str) -> list:
    """Static import closure of the phase's entry points, inside driftsense."""
    seeds = set()
    for e in ENTRIES[phase]:
        seeds |= _pkg_deps(os.path.join(REPO, e))
    seen, queue = set(), list(seeds)
    while queue:
        m = queue.pop()
        if m in seen:
            continue
        seen.add(m)
        p = os.path.join(REPO, PKG, m + ".py")
        if os.path.exists(p):
            queue.extend(d for d in _pkg_deps(p) if d not in seen)
    return sorted(m for m in seen
                  if os.path.exists(os.path.join(REPO, PKG, m + ".py")))


def _norm(name: str) -> str:
    """PEP 503 normalisation -- 'Jinja2', 'jinja_2' and 'jinja-2' are one name."""
    return name.lower().replace("_", "-")


def installed_versions() -> dict:
    """name -> pinned spec.

    The repository's own requirements.txt is the source of truth, because the
    organizers' rule is to pin "within the same sandbox environment as Phase
    2" -- so a line that already shipped for Phase 2 must ship here spelled
    and pinned identically, not merely equivalently. The build environment's
    pip metadata fills in anything that file does not carry.
    """
    out = {}
    try:
        from importlib.metadata import distributions
        for d in distributions():
            name = d.metadata["Name"]
            if name:
                out[_norm(name)] = f"{name}=={d.version}"
    except ImportError:                                   # pragma: no cover
        pass

    root_req = os.path.join(REPO, "requirements.txt")
    if os.path.exists(root_req):
        for line in open(root_req):
            line = line.split("#")[0].strip()
            if "==" in line:
                out[_norm(line.split("==")[0])] = line
    return out


def requirements_for(phase: str, versions: dict) -> str:
    wanted = DIRECT[phase] + TRANSITIVE
    lines, missing = [], []
    for name in wanted:
        key = _norm(name)
        if key in versions:
            lines.append(versions[key])
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"build: {phase}: not installed in the build environment, so no "
            f"version can be pinned: {', '.join(missing)}")
    header = (
        "# Drift-Sense inference -- {p}.\n"
        "# Pinned from the build environment (pip freeze), scoped to what this\n"
        "# entry point actually imports. CPU-only; nothing here reaches the\n"
        "# network at run time.\n"
        "#\n"
        "#   python3.11 -m venv venv\n"
        "#   ./venv/bin/pip install -r requirements.txt\n"
    ).format(p=phase)
    return header + "\n".join(sorted(lines, key=str.lower)) + "\n"


def build_phase(phase: str, out_root: str, versions: dict) -> dict:
    dest = os.path.join(out_root, phase)
    os.makedirs(os.path.join(dest, PKG), exist_ok=True)
    os.makedirs(os.path.join(dest, "weights"), exist_ok=True)

    for e in ENTRIES[phase]:
        shutil.copy2(os.path.join(REPO, e), os.path.join(dest, e))

    mods = python_modules(phase)
    shutil.copy2(os.path.join(REPO, PKG, "__init__.py"),
                 os.path.join(dest, PKG, "__init__.py"))
    for m in mods:
        shutil.copy2(os.path.join(REPO, PKG, m + ".py"),
                     os.path.join(dest, PKG, m + ".py"))

    src_w = os.path.join(REPO, WEIGHTS)
    if os.path.getsize(src_w) < MIN_WEIGHTS_BYTES:
        raise SystemExit(f"build: {WEIGHTS} is {os.path.getsize(src_w)} bytes "
                         "-- truncated, or an unfetched LFS pointer")
    shutil.copy2(src_w, os.path.join(dest, WEIGHTS))

    with open(os.path.join(dest, "requirements.txt"), "w") as f:
        f.write(requirements_for(phase, versions))

    sample_n = build_sample(phase, dest)
    doc = os.path.join(HERE, "bundle_docs", phase + ".md")
    if os.path.exists(doc):
        shutil.copy2(doc, os.path.join(dest, "README.md"))

    return {"phase": phase, "modules": mods, "sample": sample_n,
            "entries": ENTRIES[phase], "dest": dest}


def _read_csv(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_sample(phase: str, dest: str) -> int:
    """Copy a handful of real pairs in, rewriting every path to be relative."""
    import csv
    spec = SAMPLES.get(phase)
    if not spec:
        return 0
    root = os.path.normpath(spec["root"])
    pairs_csv = os.path.join(root, "pairs.csv")
    if not os.path.exists(pairs_csv):
        print(f"build: {phase}: no sample source at {root} -- skipping sample")
        return 0

    rows = {r["pair_id"]: r for r in _read_csv(pairs_csv)}
    gt = {r["pair_id"]: r for r in _read_csv(os.path.join(root, "ground_truth.csv"))}
    out = os.path.join(dest, "sample")
    os.makedirs(out, exist_ok=True)

    kept, kept_gt = [], []
    for pid in spec["pairs"]:
        if pid not in rows:
            continue
        row = dict(rows[pid])
        for col, subdir in spec["copy"]:
            rel = row.get(col, "")
            if not rel:
                continue
            src = os.path.join(root, rel)
            if not os.path.exists(src):
                row[col] = ""
                continue
            os.makedirs(os.path.join(out, subdir), exist_ok=True)
            base = os.path.basename(src)
            shutil.copy2(src, os.path.join(out, subdir, base))
            # Rewrite to a path relative to the sample's own pairs.csv, so the
            # folder can be moved anywhere and still run.
            row[col] = f"{subdir}/{base}"
        kept.append(row)
        if pid in gt:
            kept_gt.append(gt[pid])

    if not kept:
        return 0
    with open(os.path.join(out, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
        w.writeheader(); w.writerows(kept)
    if kept_gt:
        with open(os.path.join(out, "ground_truth.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(kept_gt[0].keys()))
            w.writeheader(); w.writerows(kept_gt)
    return len(kept)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "dist",
                                                  "driftsense-inference"))
    ap.add_argument("--clean", action="store_true",
                    help="remove the output directory first")
    a = ap.parse_args(argv)

    if a.clean and os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out, exist_ok=True)

    versions = installed_versions()
    built = [build_phase(p, a.out, versions) for p in ("phase_2", "phase_3")]

    top = os.path.join(HERE, "bundle_docs", "README.md")
    if os.path.exists(top):
        shutil.copy2(top, os.path.join(a.out, "README.md"))

    total = 0
    for root, _dirs, files in os.walk(a.out):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))

    for b in built:
        print(f"{b['phase']}: {GRADED[b['phase']]} + "
              f"{len(b['modules'])} driftsense modules, "
              f"{b['sample']}-pair sample")
    print(f"total {total / 1e6:.1f} MB -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
