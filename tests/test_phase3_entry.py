"""Phase 3 entry point (``phase3.py``).

The contract this file pins, in order of how expensive it would be to get wrong:

1. **A non-Phase-3 input must abort loudly, before writing anything.** The
   Phase 3 header has two columns matching "reference"
   (``reference_gds_path``, ``reference_sem_path``). ``register.pick_column``'s
   substring fallback picks the GDS one, hands it to ``cv2.imread``, gets
   ``None``, raises ``SystemExit`` inside ``read_gray``, and the per-pair
   handler swallows it into a *declined* row -- producing a well-formed,
   exit-0, all-declined ``predictions.csv`` that is indistinguishable from an
   honest all-reject run. These tests assert that cannot happen.

2. **The output contract is unchanged from Phase 2.** Same header, one row per
   ``pair_id``, exactly once, in input order, and every row present even when
   the pair fails. A missing row scores zero; a declined row does not.

3. **The withheld columns are never dereferenced at inference.** The blind split
   leaves ``reference_sem_path`` and ``params_json_path`` empty, and the deck is
   explicit that code needing them must fail on data we hold rather than on the
   scored run.

These run the real entry point as a subprocess against a real GDS written by
gdstk, so the CLI contract itself is exercised, not just the functions.
"""

import csv
import os
import subprocess
import sys

import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import phase3  # noqa: E402
from driftsense.pairs3 import PHASE3_FIELDS, WITHHELD_FIELDS  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _write_gds(path, layers=(0, 3, 6), size=1000.0):
    cell = gdstk.Cell("REF")
    for layer in layers:
        for i in range(3):
            lo = (i * 120.0, i * 140.0)
            cell.add(gdstk.rectangle(lo, (lo[0] + 260.0, lo[1] + 200.0),
                                     layer=layer))
    lib = gdstk.Library()
    lib.add(cell)
    lib.write_gds(str(path))
    return str(path)


def _write_png(path, seed=0, size=1000):
    import cv2
    rng = np.random.default_rng(seed)
    img = (rng.random((size, size)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def _pairs_csv(path, rows, blind=True):
    """Write a Phase 3 pairs.csv. `blind=True` leaves the withheld fields empty."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PHASE3_FIELDS)
        for i, (gds_p, png_p) in enumerate(rows):
            w.writerow([f"p{i}", png_p, gds_p, "",
                        "" if blind else "ref_sem.png",
                        "" if blind else "params.json"])
    return str(path)


def _run(args):
    return subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "phase3.py"), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=900)


# --------------------------------------------------------------------------
# 1. the schema refusal
# --------------------------------------------------------------------------

def test_phase3_layout_is_accepted(tmp_path):
    g = _write_gds(tmp_path / "r.gds")
    p = _write_png(tmp_path / "s.png")
    csvp = _pairs_csv(tmp_path / "pairs.csv", [(g, p)])
    r = _run(["--input", csvp, "--output", str(tmp_path / "out.csv"),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, r.stderr
    assert os.path.exists(tmp_path / "out.csv")


def test_a_phase2_header_is_refused_not_guessed(tmp_path):
    """A Phase 2 manifest must NOT be silently accepted as Phase 3."""
    g = _write_gds(tmp_path / "r.gds")
    p = _write_png(tmp_path / "s.png")
    csvp = tmp_path / "pairs2.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "reference_path", "search_path"])
        w.writerow(["p0", g, p])
    out = tmp_path / "out.csv"
    r = _run(["--input", str(csvp), "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode != 0, "a Phase 2 header must not be accepted"
    assert not out.exists(), "nothing may be written on a schema refusal"
    assert "phase3" in r.stderr.lower() or "pairs" in r.stderr.lower()


def test_ambiguous_reference_columns_are_refused(tmp_path):
    """The exact hazard: both reference* columns present, one ambiguous role."""
    csvp = tmp_path / "ambiguous.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "search_path", "reference", "reference_path"])
        w.writerow(["p0", "s.png", "a.gds", "b.png"])
    out = tmp_path / "out.csv"
    r = _run(["--input", str(csvp), "--output", str(out), "--quiet"])
    assert r.returncode != 0
    assert not out.exists()


def test_refusal_message_names_the_expected_columns(tmp_path):
    csvp = tmp_path / "bad.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "c"])
        w.writerow(["1", "2", "3"])
    out = tmp_path / "out.csv"
    r = _run(["--input", str(csvp), "--output", str(out), "--quiet"])
    assert r.returncode != 0
    assert "reference_gds_path" in r.stderr, "the error must say what is wanted"


def test_empty_input_is_refused(tmp_path):
    csvp = tmp_path / "empty.csv"
    with open(csvp, "w", newline="") as f:
        csv.writer(f).writerow(PHASE3_FIELDS)
    out = tmp_path / "out.csv"
    r = _run(["--input", str(csvp), "--output", str(out), "--quiet"])
    assert r.returncode != 0
    assert not out.exists()


def test_absent_input_file_is_refused(tmp_path):
    out = tmp_path / "out.csv"
    r = _run(["--input", str(tmp_path / "nope.csv"), "--output", str(out),
              "--quiet"])
    assert r.returncode != 0
    assert not out.exists()


# --------------------------------------------------------------------------
# 2. the output contract
# --------------------------------------------------------------------------

def test_output_header_is_the_phase2_contract(tmp_path):
    g = _write_gds(tmp_path / "r.gds")
    p = _write_png(tmp_path / "s.png")
    csvp = _pairs_csv(tmp_path / "pairs.csv", [(g, p)])
    out = tmp_path / "out.csv"
    r = _run(["--input", csvp, "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, r.stderr
    with open(out, newline="") as f:
        header = next(csv.reader(f))
    assert header == ["pair_id", "x", "y", "theta", "scale", "found", "score"]


def test_one_row_per_pair_in_input_order(tmp_path):
    rows = []
    for i in range(4):
        rows.append((_write_gds(tmp_path / f"r{i}.gds", layers=(0, 2 + i)),
                     _write_png(tmp_path / f"s{i}.png", seed=i)))
    csvp = _pairs_csv(tmp_path / "pairs.csv", rows)
    out = tmp_path / "out.csv"
    r = _run(["--input", csvp, "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, r.stderr
    with open(out, newline="") as f:
        got = [row["pair_id"] for row in csv.DictReader(f)]
    assert got == ["p0", "p1", "p2", "p3"], "order and completeness must hold"


def test_a_failing_pair_still_gets_a_declined_row(tmp_path):
    """A missing row scores zero; a declined row does not. Never drop a row."""
    good_g = _write_gds(tmp_path / "good.gds")
    good_p = _write_png(tmp_path / "good.png")
    bad_p = tmp_path / "missing.png"          # never created
    csvp = _pairs_csv(tmp_path / "pairs.csv",
                      [(good_g, good_p), (good_g, str(bad_p))])
    out = tmp_path / "out.csv"
    r = _run(["--input", csvp, "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, r.stderr
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2, "the failing pair must still emit a row"
    bad = [row for row in got if row["pair_id"] == "p1"][0]
    assert bad["found"] == "0"
    assert bad["x"] == "0" and bad["y"] == "0"
    assert bad["theta"] == "0" and bad["scale"] == "0"


def test_an_unreadable_gds_declines_that_pair_only(tmp_path):
    good_g = _write_gds(tmp_path / "good.gds")
    good_p = _write_png(tmp_path / "good.png")
    broken = tmp_path / "broken.gds"
    broken.write_bytes(b"not a gds file")
    csvp = _pairs_csv(tmp_path / "pairs.csv",
                      [(good_g, good_p), (str(broken), good_p)])
    out = tmp_path / "out.csv"
    r = _run(["--input", csvp, "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, r.stderr
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2
    assert [row for row in got if row["pair_id"] == "p1"][0]["found"] == "0"
    assert "warn" in r.stderr.lower(), "the failure must be reported"


def test_output_parent_directory_is_created(tmp_path):
    g = _write_gds(tmp_path / "r.gds")
    p = _write_png(tmp_path / "s.png")
    csvp = _pairs_csv(tmp_path / "pairs.csv", [(g, p)])
    out = tmp_path / "deep" / "nested" / "out.csv"
    r = _run(["--input", csvp, "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, r.stderr
    assert out.exists()


# --------------------------------------------------------------------------
# 3. the withheld columns
# --------------------------------------------------------------------------

def test_blind_split_withheld_fields_are_never_read(tmp_path):
    """Point them at paths that do not exist; a run must still succeed."""
    g = _write_gds(tmp_path / "r.gds")
    p = _write_png(tmp_path / "s.png")
    csvp = tmp_path / "pairs.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PHASE3_FIELDS)
        # deliberately bogus withheld values -- dereferencing either must fail
        w.writerow(["p0", p, g, "", "__DOES_NOT_EXIST__.png",
                    "__DOES_NOT_EXIST__.json"])
    out = tmp_path / "out.csv"
    r = _run(["--input", str(csvp), "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert r.returncode == 0, (
        "the inference path must not touch reference_sem_path/params_json_path: "
        + r.stderr)
    assert "DOES_NOT_EXIST" not in r.stdout
    assert "DOES_NOT_EXIST" not in r.stderr


def test_withheld_fields_are_the_documented_two():
    assert set(WITHHELD_FIELDS) == {"reference_sem_path", "params_json_path"}


# --------------------------------------------------------------------------
# constants / wiring
# --------------------------------------------------------------------------

def test_phase3_reuses_the_phase2_output_fields():
    """The contract must not be re-declared independently and drift."""
    import register
    assert phase3.OUT_FIELDS is register.OUT_FIELDS


def test_mass_failure_thresholds_are_present_even_when_register_lacks_them():
    assert phase3.MASS_FAILURE_ERROR_FRAC > 0
    assert 0 < phase3.MASS_FAILURE_FOUND_FRAC < 1
    assert phase3.MASS_FAILURE_MIN_PAIRS >= 1


def test_a_systematic_failure_raises_the_mass_failure_marker(tmp_path):
    """Every pair unreadable -> the alarm must fire, not just a scroll of warns."""
    g = _write_gds(tmp_path / "r.gds")
    good_p = _write_png(tmp_path / "s.png")
    rows = []
    for i in range(10):
        rows.append((str(tmp_path / f"missing_{i}.gds"), str(good_p)))
    csvp = _pairs_csv(tmp_path / "pairs.csv", rows)
    out = tmp_path / "out.csv"
    r = _run(["--input", csvp, "--output", str(out),
              "--allow-fallback", "--quiet"])
    assert "mass_failure" in r.stderr, (
        "a 100%-failure run must be impossible to miss")
