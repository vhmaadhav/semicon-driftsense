"""register.py failure-path contract.

Two properties of the graded entry point are pinned here, neither of which had
any test coverage before (audit 2026-09-04: register.py was the least-covered
shipped module at 56%, and lines 537-542 -- the exception handlers implementing
the single most important guarantee in the output contract -- were untested).

1. **A pair that fails still emits a row.** `register.py`'s own docstring calls
   this non-negotiable: "a missing row scores zero, so declining beats
   disappearing." Unreadable, truncated and structurally impossible inputs must
   each produce exactly one declined row, and must not abort the batch.

2. **A systematic failure is loud.** The forgiveness in (1) has a cost: a run
   that fails on every pair is otherwise indistinguishable from a run that
   confidently declined every pair -- same well-formed CSV, same exit code 0.
   The mass-failure banner exists to make that distinguishable, and must fire
   on a broken run without firing on a small or merely-unlucky one.

Runs the real entry point as a subprocess, so what is tested is what the grader
executes.
"""

import csv
import os
import subprocess
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTER = os.path.join(REPO_ROOT, "register.py")

OUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]


def _write_broken_inputs(d):
    """Three ways an image can be unusable, all of which a judge set could
    plausibly contain: absent, truncated mid-stream, and not an image at all."""
    real = np.random.default_rng(3).integers(0, 255, (200, 200), dtype=np.uint8)
    cv2.imwrite(str(d / "real.png"), real)
    data = (d / "real.png").read_bytes()
    (d / "truncated.png").write_bytes(data[:len(data) // 3])
    (d / "notanimage.png").write_text("this is plainly not a PNG")
    # "missing.png" is deliberately never created.


def _pairs_csv(d, name, rows):
    p = d / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "reference_path", "search_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def _run(pairs_csv, tmp_path, tag):
    out_csv = str(tmp_path / f"preds_{tag}.csv")
    p = subprocess.run(
        [sys.executable, REGISTER, "--input", pairs_csv, "--output", out_csv,
         "--quiet"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    return out_csv, p


def _read(out_csv):
    with open(out_csv, newline="") as f:
        return list(csv.DictReader(f))


def test_every_broken_pair_still_emits_exactly_one_declined_row(tmp_path):
    """The contract guarantee: bad input degrades to a decline, never to a
    missing row and never to an aborted batch."""
    _write_broken_inputs(tmp_path)
    rows = [
        {"pair_id": "missing", "reference_path": "real.png",
         "search_path": "missing.png"},
        {"pair_id": "truncated", "reference_path": "real.png",
         "search_path": "truncated.png"},
        {"pair_id": "notanimage", "reference_path": "notanimage.png",
         "search_path": "real.png"},
    ]
    pairs = _pairs_csv(tmp_path, "broken.csv", rows)
    out_csv, p = _run(pairs, tmp_path, "broken")

    assert p.returncode == 0, p.stderr
    got = _read(out_csv)
    assert list(got[0].keys()) == OUT_FIELDS
    # Exactly one row per input pair, in input order, none dropped.
    assert [r["pair_id"] for r in got] == ["missing", "truncated", "notanimage"]
    for r in got:
        assert r["found"] == "0", r
        # A declined row zero-fills every pose column (contract, README).
        assert (r["x"], r["y"], r["theta"], r["scale"]) == ("0", "0", "0", "0"), r
        assert float(r["score"]) == 0.0, r


def test_mass_failure_banner_fires_and_still_writes_every_row(tmp_path):
    """A systematically broken run must be impossible to mistake for a good one
    -- while still writing all its rows and exiting 0."""
    _write_broken_inputs(tmp_path)
    n = 10
    rows = [{"pair_id": f"p{i:03d}", "reference_path": "notanimage.png",
             "search_path": "notanimage.png"} for i in range(n)]
    pairs = _pairs_csv(tmp_path, "allbad.csv", rows)
    out_csv, p = _run(pairs, tmp_path, "allbad")

    # Still a well-formed, complete, successful-looking run -- that is the point.
    assert p.returncode == 0, p.stderr
    assert len(_read(out_csv)) == n

    # ...but unmistakably flagged, with a machine-readable marker for a harness.
    assert "[MASS FAILURE]" in p.stderr
    assert f"# mass_failure: errors={n} found=0 n={n}" in p.stderr


def test_mass_failure_banner_is_suppressed_on_a_run_too_small_to_judge(tmp_path):
    """Below MASS_FAILURE_MIN_PAIRS the rates are noise. A 3-pair set that
    happens to contain 3 bad images is not evidence of a broken system."""
    _write_broken_inputs(tmp_path)
    rows = [{"pair_id": f"q{i}", "reference_path": "notanimage.png",
             "search_path": "notanimage.png"} for i in range(3)]
    pairs = _pairs_csv(tmp_path, "tiny.csv", rows)
    out_csv, p = _run(pairs, tmp_path, "tiny")

    assert p.returncode == 0, p.stderr
    assert len(_read(out_csv)) == 3
    assert "# mass_failure:" not in p.stderr
    assert "[MASS FAILURE]" not in p.stderr


def test_mass_failure_thresholds_stay_loose_enough_for_a_real_run():
    """These thresholds must never fire on a genuinely hard blind set -- they
    detect breakage, not difficulty. Pinned so a future tightening is a
    deliberate act with a test to change, not a silent one."""
    sys.path.insert(0, REPO_ROOT)
    import register

    # The disclosed composition is ~80% present. The found-rate alarm has to sit
    # far below that, or a hard set trips it.
    assert register.MASS_FAILURE_FOUND_FRAC <= 0.40
    # And an error alarm at <=0 or >1 would be meaningless.
    assert 0.0 < register.MASS_FAILURE_ERROR_FRAC <= 0.50
    # Small runs must be exempt, or the audit fixtures and smoke tests trip it.
    assert register.MASS_FAILURE_MIN_PAIRS >= 5
