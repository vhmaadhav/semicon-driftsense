#!/usr/bin/env python3
"""One-command submission self-check.

Answers the question a grader actually has -- *does this run, and is it the
model the README describes?* -- on a machine with nothing pre-generated. It
uses the **upstream generator** (`generator/src/pipeline.py`) rather than ours,
and scores against the **upstream label** `gt_x`/`gt_y`, which is the only one
that generator emits. So nothing here depends on our data pipeline or our
label convention: it is the submission judged on neutral ground.

    python scripts/selfcheck.py             # ~1 min, 6 scenes
    python scripts/selfcheck.py --n 24      # tighter estimate

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

TOLERANCE_PX = 5.0
# Recorded in README.md; a mismatch means the weights are not the shipped ones.
EXPECTED_SHA256 = "90db89f9861c2c9ea386eaa03e45ff03fc4962dc7e349aa00423621a5fce1488"
EXPECTED_EPOCHS = 24

PASS, FAIL, WARN = "  PASS", "  FAIL", "  WARN"
_failures: list[str] = []


def _progress(msg: str) -> None:
    """In-place progress, but only on a terminal -- redirected to a file or a
    CI log, carriage returns smear every update onto one unreadable line."""
    if sys.stdout.isatty():
        print(msg, end="\r", flush=True)


def _progress_done() -> None:
    if sys.stdout.isatty():
        print(" " * 48, end="\r")


def check(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> bool:
    tag = PASS if ok else (WARN if warn_only else FAIL)
    print(f"{tag}  {label}" + (f"  --  {detail}" if detail else ""))
    if not ok and not warn_only:
        _failures.append(label)
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def check_environment() -> None:
    section("environment")
    v = sys.version_info
    check((3, 10) <= (v.major, v.minor) <= (3, 13),
          "python 3.10-3.13", f"running {v.major}.{v.minor}.{v.micro}")
    try:
        import cv2, torch  # noqa: E402
        check(True, "imports", f"torch {torch.__version__}, cv2 {cv2.__version__}, "
                               f"numpy {np.__version__}")
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        check(True, "compute device", dev)
    except Exception as e:
        check(False, "imports", str(e))


def check_weights() -> None:
    section("weights")
    path = os.path.join(REPO_ROOT, "weights", "driftsense.pt")
    if not check(os.path.exists(path), "weights present", path):
        return
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    check(digest == EXPECTED_SHA256, "sha256 matches the README",
          digest[:24] + ("..." if digest == EXPECTED_SHA256
                         else f"... (expected {EXPECTED_SHA256[:24]}...)"))
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        check(ckpt.get("epoch") == EXPECTED_EPOCHS, "checkpoint identity",
              f"{ckpt.get('epoch')} trained epochs (expected {EXPECTED_EPOCHS})")
    except Exception as e:
        check(False, "checkpoint loads", str(e))


def make_upstream_scenes(n: int, seed: int, out_dir: str) -> list[dict]:
    """Render scenes with the UPSTREAM generator, untouched."""
    import cv2
    from src.pipeline import GenerationParams, generate_sample
    from src.presets import PRESETS

    # Interleave the two families so even a 6-scene default run exercises both
    # DRAM and FinFET -- PRESETS lists all the DRAM variants first, so a plain
    # round-robin over its keys would silently test one family only.
    keys = list(PRESETS.keys())
    dram = [k for k in keys if k.startswith("dram")]
    finfet = [k for k in keys if k.startswith("finfet")]
    archs = [k for pair in zip(dram, finfet) for k in pair] or keys
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    scenes = []
    for i in range(n):
        s = generate_sample(archs[i % len(archs)], rng, GenerationParams())
        rp = os.path.join(out_dir, f"{i:03d}_reference.png")
        sp = os.path.join(out_dir, f"{i:03d}_search.png")
        cv2.imwrite(rp, s["reference_img"])
        cv2.imwrite(sp, s["search_img"])
        scenes.append({"reference": rp, "search": sp,
                       "gt_x": s["gt_x"], "gt_y": s["gt_y"],
                       "architecture": s["architecture"]})
        _progress(f"    generated {i + 1}/{n}")
    _progress_done()
    return scenes


def check_cli_contract(scene: dict) -> None:
    section("cli contract (infer.py)")
    p = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "infer.py"),
                        "--reference", scene["reference"], "--search", scene["search"]],
                       capture_output=True, text=True, cwd=REPO_ROOT)
    check(p.returncode == 0, "exits 0", p.stderr.strip()[:80])
    lines = p.stdout.strip().splitlines()
    check(len(lines) == 1, "single line on stdout", f"got {len(lines)}")
    if lines:
        check(bool(re.fullmatch(r"-?\d+\.\d{2},-?\d+\.\d{2}", lines[0])),
              "format is 'x,y' to 2dp", lines[0])


def check_accuracy(scenes: list[dict]) -> None:
    section(f"localisation accuracy on {len(scenes)} upstream scenes")
    from infer import predict

    errs, per_arch = [], {}
    for i, s in enumerate(scenes):
        r = predict(s["reference"], s["search"])
        e = float(np.hypot(r["x"] - s["gt_x"], r["y"] - s["gt_y"]))
        errs.append(e)
        per_arch.setdefault(s["architecture"], []).append(e)
        _progress(f"    scored {i + 1}/{len(scenes)}")
    _progress_done()

    errs = np.array(errs)
    hits = int((errs <= TOLERANCE_PX).sum())
    print(f"       median {np.median(errs):.2f} px   mean {errs.mean():.2f} px   "
          f"max {errs.max():.2f} px")
    print(f"       scored against the upstream label gt_x/gt_y "
          f"(the only one this generator emits)")
    for arch, e in sorted(per_arch.items()):
        print(f"         {arch:<14} n={len(e):<3} median {np.median(e):6.2f} px")

    check(hits == len(errs), f"all within {TOLERANCE_PX:g} px",
          f"{hits}/{len(errs)}")
    # A wrong-repeat lock-on is tens of px out; that is the failure that matters.
    wrong_repeat = int((errs > 10.0).sum())
    check(wrong_repeat == 0, "no wrong-repeat lock-ons (>10 px)",
          f"{wrong_repeat}/{len(errs)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=6, help="scenes to render (default 6)")
    p.add_argument("--seed", type=int, default=20260818)
    args = p.parse_args()

    print("Drift-Sense self-check")
    print("=" * 60)
    check_environment()
    check_weights()

    section(f"rendering {args.n} scenes with the upstream generator")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            scenes = make_upstream_scenes(args.n, args.seed, tmp)
            check(len(scenes) == args.n, "generator runs", f"{len(scenes)} scenes")
        except Exception as e:
            check(False, "generator runs", str(e))
            return 1
        check_cli_contract(scenes[0])
        check_accuracy(scenes)

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + "; ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
