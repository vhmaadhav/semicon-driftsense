#!/usr/bin/env python3
"""G6 (PHASE2_COMPLIANCE_ISSUES.md): submission artifact audit.

Two modes:

*   Artifact audit (the default answer to "is the submission compliant"):

        python scripts/check_submission_zip.py dist/submission.zip

    The ZIP is extracted into a temporary directory and ONLY that extraction
    is audited -- the repository checkout is never consulted. Checks run
    against the extraction: required root layout (register.py at the root,
    weights/driftsense.pt next to it, requirements.txt, failure_analysis.pdf,
    generate_dataset.py), an actual torch.load(weights_only=True) of the
    shipped checkpoint, --help smoke tests for the organizer entry points
    run from the extraction directory, a transitive-import network scan,
    PDF page count, and the requirements pin check read from inside the ZIP.

*   Preflight (no argument, or --preflight):

        python scripts/check_submission_zip.py --preflight

    Audits the repository working tree. Every line is labelled PREFLIGHT.
    A preflight PASS means the checkout is submission-ready; it is NOT
    evidence that the artifact actually submitted is compliant. Always run
    the artifact audit on the final ZIP before uploading.

Exits non-zero on any FAIL. SKIPped checks (e.g. torch unavailable for the
load test) are reported and never counted as PASS.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Root layout the organizer command needs. register.py resolves weights via
# infer.DEFAULT_WEIGHTS = <dir of infer.py>/weights/driftsense.pt, and
# register.py puts its own directory on sys.path, so register.py, infer.py
# and weights/driftsense.pt must sit together at the extraction root.
REQUIRED_PATHS = [
    "register.py",
    "infer.py",
    "requirements.txt",
    "failure_analysis.pdf",
    "generate_dataset.py",
    os.path.join("weights", "driftsense.pt"),
]
WEIGHTS_REL = os.path.join("weights", "driftsense.pt")
ORGANIZER_CMD = "python register.py --input pairs.csv --output predictions.csv"

NETWORK_MARKERS = ("requests.", "urllib.", "socket.", "httpx.", "urlopen")
SUBPROC_NET_RE = re.compile(r"\b(curl|wget)\b")
SUBPROC_CALL_RE = re.compile(
    r"subprocess|Popen|os\.system|check_output|check_call|\.run\(")

checks = []


def check(name, ok, detail="", skipped=False):
    checks.append({"name": name, "ok": ok, "detail": detail, "skipped": skipped})


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# transitive local-import resolution
# --------------------------------------------------------------------------

def imported_names(src):
    """Absolute import module names mentioned in a source file."""
    names = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def resolve_local(root, module):
    """Map a dotted module name to a .py file inside root, else None."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        rel = os.path.join(root, *parts[:i]) + ".py"
        if os.path.isfile(rel):
            return rel
        pkg_init = os.path.join(root, *parts[:i], "__init__.py")
        if os.path.isfile(pkg_init):
            return pkg_init
    return None


def transitive_local_modules(root, entry_files, max_depth=64):
    """entry_files plus every local .py they transitively import from root."""
    seen = []
    queue = [(os.path.abspath(f), 0) for f in entry_files
             if os.path.isfile(f)]
    visited = set(path for path, _ in queue)
    while queue:
        path, depth = queue.pop(0)
        if depth > max_depth:
            continue
        seen.append(path)
        try:
            src = read_text(path)
        except (OSError, UnicodeDecodeError):
            continue
        for module in imported_names(src):
            target = resolve_local(root, module)
            if target and target not in visited:
                visited.add(target)
                queue.append((target, depth + 1))
    return seen


def network_findings(src):
    findings = []
    for line_no, line in enumerate(src.splitlines(), 1):
        hit = None
        for marker in NETWORK_MARKERS:
            if marker in line:
                hit = marker.rstrip(".")
                break
        if hit is None:
            m = SUBPROC_NET_RE.search(line)
            if m and SUBPROC_CALL_RE.search(line):
                hit = m.group(1) + " (subprocess)"
        if hit:
            findings.append(f"line {line_no}: {hit}")
    return findings


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------

def audit_layout(root):
    missing = [p for p in REQUIRED_PATHS
               if not os.path.isfile(os.path.join(root, p))]
    try:
        entries = [e for e in os.listdir(root) if not e.startswith(".")]
    except OSError:
        entries = []
    if "register.py" in missing and len(entries) == 1 \
            and os.path.isdir(os.path.join(root, entries[0])):
        # Everything nested under one directory means the organizer command
        # fails from the extraction root even if the files exist below it.
        nested = os.path.join(root, entries[0], "register.py")
        check("root layout (organizer command)", False,
              "all contents nested under " + entries[0] + "/; the organizer "
              "command (" + ORGANIZER_CMD + ") would fail from the extraction "
              "root" + (" (register.py found there -- re-zip flat)"
                        if os.path.isfile(nested) else ""))
        return
    check("required paths present (register.py, infer.py, requirements.txt, "
          "failure_analysis.pdf, generate_dataset.py, weights/driftsense.pt)",
          not missing,
          "all present" if not missing else "missing: " + ", ".join(missing))

    w = os.path.join(root, WEIGHTS_REL)
    if os.path.isfile(w):
        check("weights/driftsense.pt present", True,
              format(os.path.getsize(w) / 1e6, ".1f") + " MB")


def audit_requirements(root):
    """Pin check read from the file inside the artifact, not the checkout."""
    req = os.path.join(root, "requirements.txt")
    if not os.path.isfile(req):
        check("requirements.txt present", False, "missing from artifact")
        return
    lines = [l.strip() for l in open(req, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    pinned = [l for l in lines if re.match(r"^[A-Za-z0-9_.-]+==", l)]
    check("requirements.txt pip-freeze format (pinned)",
          bool(lines) and len(pinned) == len(lines),
          str(len(pinned)) + "/" + str(len(lines)) + " pinned")


def audit_weights_load(root):
    """Actual load test, mirroring tests/test_checkpoint_safety.py."""
    w = os.path.join(root, WEIGHTS_REL)
    name = "checkpoint loads (torch.load weights_only=True, has 'model' key)"
    if not os.path.isfile(w):
        check(name, False, WEIGHTS_REL + " missing from the artifact")
        return
    try:
        import torch  # noqa: F401
    except Exception as exc:  # torch absent / broken in the auditing env
        check(name, None,
              "SKIP: torch unavailable in auditing environment ("
              + str(exc) + ")", skipped=True)
        return
    try:
        ckpt = torch.load(w, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            extra = ""
            if isinstance(ckpt, dict):
                extra = " with keys " + ", ".join(sorted(ckpt))
            check(name, False,
                  "loaded object is " + type(ckpt).__name__ + extra
                  + "; no 'model' key")
            return
        check(name, True, "loaded with weights_only=True on CPU")
    except Exception as exc:
        check(name, False,
              "torch.load failed: " + type(exc).__name__ + ": " + str(exc))


def audit_documentation(root):
    gd = os.path.join(root, "generate_dataset.py")
    if not os.path.isfile(gd):
        check("generate_dataset.py present", False, "missing from artifact")
        return
    src = read_text(gd)
    # Substantive documentation predicate: a real module docstring (parsed
    # from the AST, so a shebang or a stray string literal cannot satisfy
    # it) that documents its own arguments. Each term is parenthesised
    # explicitly -- the old (A and B) or C precedence bug cannot recur.
    docstring = None
    try:
        docstring = ast.get_docstring(ast.parse(src))
    except SyntaxError:
        pass
    has_doc = isinstance(docstring, str) and bool(docstring.strip())
    args_doc = any(flag in (docstring or "") for flag in
                   ("--output-dir", "--num-pairs", "--seed", "--noise"))
    documented = has_doc and args_doc
    check("generate_dataset.py documented (module docstring names its arguments)",
          documented,
          "docstring documents CLI arguments" if documented else
          "docstring missing or does not mention arguments "
          "(--output-dir/--num-pairs/--seed/--noise); a shebang does not count")
    if not has_doc:
        return
    # Entry-point smoke tests, run from the extraction directory (outside
    # the repository checkout in artifact mode). --help needs torch/cv2
    # importable, so prefer a repo venv interpreter when one exists and only
    # hold the smoke test to the exit-0 bar when that interpreter can
    # actually import torch.
    interpreter = None
    torch_ok = False
    for cand in (os.path.join(REPO, "venv313", "bin", "python"),
                 os.path.join(REPO, "venv", "bin", "python"),
                 sys.executable):
        if not os.path.isfile(cand) and not shutil.which(cand):
            continue
        probe = subprocess.run(
            [cand, "-c", "import torch, cv2"],
            capture_output=True, text=True, timeout=120)
        interpreter = cand
        torch_ok = probe.returncode == 0
        break
    if interpreter is None:
        interpreter = sys.executable
    for script in ("register.py", "generate_dataset.py"):
        name = script + " --help exits 0 from the extraction dir"
        if not torch_ok:
            check(name, None,
                  "SKIP: no interpreter with torch+cv2 available to run the "
                  "smoke test (tried venv313/venv/" + sys.executable + ")",
                  skipped=True)
            continue
        proc = subprocess.run(
            [interpreter, script, "--help"],
            cwd=root, capture_output=True, text=True, timeout=120)
        ok = proc.returncode == 0
        detail = "exit 0" if ok else "exit " + str(proc.returncode)
        if not ok:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            if tail:
                detail += ": " + tail[-1][:200]
        check(script + " --help exits 0 from the extraction dir", ok, detail)


def audit_network(root):
    entry = os.path.join(root, "register.py")
    files = transitive_local_modules(
        root, [entry, os.path.join(root, "infer.py")])
    if not os.path.isfile(entry):
        check("no network calls in entry-point import closure",
              False, "register.py missing; cannot scan")
        return
    dirty = []
    for path in files:
        try:
            src = read_text(path)
        except (OSError, UnicodeDecodeError):
            continue
        hits = network_findings(src)
        if hits:
            dirty.append((os.path.relpath(path, root), hits))
    scanned = ", ".join(sorted(os.path.relpath(p, root) for p in files))
    check("no network calls in entry-point import closure "
          "(register.py + transitive local imports)", not dirty,
          ("scanned " + str(len(files)) + " file(s): " + scanned + "; clean")
          if not dirty else
          "; ".join(rel + ": " + ", ".join(h[:5]) for rel, h in dirty))


def audit_pdf(root):
    fa = os.path.join(root, "failure_analysis.pdf")
    if not os.path.isfile(fa):
        check("failure_analysis.pdf present", False, "missing from artifact")
        return
    with open(fa, "rb") as fh:
        data = fh.read()
    page_objs = len(re.findall(rb"/Type\s*/Page(?![s])", data))
    counts = [int(c) for c in re.findall(rb"/Count\s+(\d+)", data)]
    page_count = max([page_objs] + counts)
    check("failure_analysis.pdf <= 2 pages", 0 < page_count <= 2,
          str(page_count) + " page(s) (page objects: " + str(page_objs)
          + ", max /Count: " + str(max(counts) if counts else 0) + ")")


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def extract_zip(zip_path, dest):
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                raise SystemExit("unsafe zip member: " + repr(name))
        zf.extractall(dest)


def run_audit(root, label):
    del checks[:]
    audit_layout(root)
    audit_requirements(root)
    audit_documentation(root)
    audit_weights_load(root)
    audit_network(root)
    audit_pdf(root)

    failed = sum(1 for c in checks if not c["skipped"] and not c["ok"])
    skipped = sum(1 for c in checks if c["skipped"])
    for c in checks:
        mark = "SKIP" if c["skipped"] else ("PASS" if c["ok"] else "FAIL")
        print("[" + label + "] [" + mark + "] " + c["name"]
              + (("  (" + c["detail"] + ")") if c["detail"] else ""))
    print()
    print(label + " SUBMISSION AUDIT: "
          + ("PASS" if not failed else str(failed) + " FAILED")
          + ((" (" + str(skipped) + " SKIPPED)") if skipped else "")
          + ("" if label == "PREFLIGHT"
             else "  [artifact-level audit of the extracted ZIP]"))
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(
        description="Audit a submission ZIP (artifact audit) or the repo "
                    "tree (preflight). See the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path", nargs="?",
                    help="path to the submission ZIP to audit (artifact mode)")
    ap.add_argument("--preflight", action="store_true",
                    help="audit the repository working tree instead of a ZIP "
                         "(labelled PREFLIGHT; not artifact evidence)")
    args = ap.parse_args()

    if args.zip_path and args.preflight:
        ap.error("give either a ZIP path (artifact audit) or --preflight, "
                 "not both")

    if args.zip_path:
        label = "ARTIFACT"
        with tempfile.TemporaryDirectory(prefix="submission-audit-") as tmp:
            try:
                extract_zip(args.zip_path, tmp)
            except zipfile.BadZipFile as exc:
                print("[" + label + "] [FAIL] ZIP is unreadable: " + str(exc))
                return 1
            return run_audit(tmp, label)

    print("note: no ZIP given -- running a PREFLIGHT of the repository tree; "
          "this is not an artifact audit. Pass the ZIP path to audit the "
          "artifact.")
    return run_audit(REPO, "PREFLIGHT")


if __name__ == "__main__":
    sys.exit(main())
