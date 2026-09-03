"""register.py runtime metadata contract (workstream A, deliverable 1).

register.py is the Phase 2 entry point the judge harness runs. Two properties
are pinned here:

* The CSV contract on stdout... on the output file -- is sacred: every input
  pair_id appears exactly once, columns are the Phase 2 set, found/score are
  well-formed. Nothing about the thread caps or timing emission may perturb
  a single byte of the predictions file.
* Per-pair timing goes to STDERR (never stdout): a `# per-pair seconds`
  header, one `# t,<pair_id>,<seconds>` line per pair, and a final
  `# runtime: median X p90 Y max Z n=N` summary.

Runs the real entry point as a subprocess on a tiny synthetic pairs.csv, so
what is tested is what the grader executes. Works with or without the model
weights: if torch/weights cannot load, register.py falls back to zncc_fallback
and the CSV contract assertions still hold (mirrors test_inference.py's stance
that the no-torch path must work).
"""

import csv
import os
import re
import subprocess
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTER = os.path.join(REPO_ROOT, "register.py")

OUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]


@pytest.fixture(scope="module")
def tiny_pairs(tmp_path_factory):
    """2 pairs: 100x100 reference, 1000x1000 search, content planted so the
    classical fallback can find it without any weights."""
    rng = np.random.default_rng(7)
    search = rng.integers(40, 210, (1000, 1000), dtype=np.uint8)
    search = cv2.GaussianBlur(search, (0, 0), 2.0)
    x0, y0 = 500, 330
    patch = search[y0:y0 + 100, x0:x0 + 100]
    reference = cv2.resize(patch, (100, 100), interpolation=cv2.INTER_NEAREST)

    d = tmp_path_factory.mktemp("pairs")
    cv2.imwrite(str(d / "reference_a.png"), reference)
    cv2.imwrite(str(d / "search_a.png"), search)
    cv2.imwrite(str(d / "reference_b.png"), reference)
    cv2.imwrite(str(d / "search_b.png"), search)
    pairs_csv = d / "pairs.csv"
    pairs_csv.write_text(
        "pair_id,search_path,reference_path\n"
        "ta,search_a.png,reference_a.png\n"
        "tb,search_b.png,reference_b.png\n")
    return str(pairs_csv)


def _run_register(pairs_csv, tmp_path):
    out_csv = str(tmp_path / "preds.csv")
    p = subprocess.run(
        [sys.executable, REGISTER, "--input", pairs_csv, "--output", out_csv],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert p.returncode == 0, p.stderr
    return out_csv, p


def test_csv_contract_unchanged_and_valid(tiny_pairs, tmp_path):
    """Every input pair_id appears exactly once with valid found/score --
    the property the scoring rules make non-negotiable."""
    out_csv, p = _run_register(tiny_pairs, tmp_path)
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["pair_id"] for r in rows] == ["ta", "tb"]
    for r in rows:
        assert set(r.keys()) == set(OUT_FIELDS)
        assert r["found"] in ("0", "1")
        score = float(r["score"])
        assert np.isfinite(score)
        if r["found"] == "1":
            assert score > 0
            # x/y inside the 1000x1000 search frame, formatted 4dp
            assert re.fullmatch(r"-?\d+\.\d{4}", r["x"]) and -1 <= float(r["x"]) <= 1001
            assert re.fullmatch(r"-?\d+\.\d{4}", r["y"]) and -1 <= float(r["y"]) <= 1001


def test_stderr_contains_per_pair_timing_lines(tiny_pairs, tmp_path):
    """Header, one t-line per pair, and the final summary -- stderr only."""
    out_csv, p = _run_register(tiny_pairs, tmp_path)
    err = p.stderr
    assert "# per-pair seconds" in err
    t_lines = re.findall(r"^# t,([^,]+),([0-9.]+)$", err, re.M)
    assert [pid for pid, _ in t_lines] == ["ta", "tb"]
    for _, secs in t_lines:
        assert float(secs) >= 0.0
    m = re.search(
        r"^# runtime: median ([0-9.]+) p90 ([0-9.]+) max ([0-9.]+) n=(\d+)$",
        err, re.M)
    assert m, "missing '# runtime: median X p90 Y max Z n=N' summary on stderr"
    n = int(m.group(4))
    assert n == 2
    # summary consistent with the per-pair lines
    per_pair = [float(s) for _, s in t_lines]
    assert float(m.group(1)) == pytest.approx(float(np.median(per_pair)), abs=0.01)
    assert float(m.group(3)) >= float(m.group(1))


def test_stdout_free_of_timing_metadata(tiny_pairs, tmp_path):
    """stdout keeps only the human progress/summary lines -- no '#' timing
    metadata may leak into the stream the CSV contract is judged from."""
    out_csv, p = _run_register(tiny_pairs, tmp_path)
    for line in p.stdout.splitlines():
        assert not line.startswith("#"), line
    assert "wrote 2 rows to" in p.stdout


def test_piped_stdout_is_plain_text(tiny_pairs, tmp_path):
    """No ANSI escapes when stdout is not a terminal.

    The dashboard is for humans; a redirected run must stay greppable, and a
    log full of escape sequences is neither greppable nor reviewable.
    """
    out_csv, p = _run_register(tiny_pairs, tmp_path)
    assert "\033" not in p.stdout, "ANSI escape leaked into a piped stdout"
    assert "\033" not in p.stderr


def test_interactive_run_keeps_timings_out_of_the_terminal(tiny_pairs, tmp_path):
    """On a tty the per-pair records go to a sidecar file, not the screen.

    One '# t,<pair>,<secs>' line per pair would scroll the dashboard away, so
    an interactive run routes them to '<output>.timing'. They must still all
    be there -- suppressing them outright would lose the per-pair audit trail
    (PR #51 review).
    """
    pty = pytest.importorskip("pty")
    out_csv = str(tmp_path / "preds_tty.csv")
    chunks = []
    status = pty.spawn(
        [sys.executable, REGISTER, "--input", tiny_pairs, "--output", out_csv],
        lambda fd: (lambda d: (chunks.append(d), d)[1])(os.read(fd, 4096)))
    assert status == 0 or os.waitstatus_to_exitcode(status) == 0
    screen = b"".join(chunks).decode("utf-8", "replace")

    assert "# t," not in screen, "per-pair timing lines reached the terminal"
    assert "# per-pair seconds" not in screen
    # the one-line summary is still on stderr, and the CSV is still correct
    assert re.search(r"# runtime: median [0-9.]+ p90 [0-9.]+ max [0-9.]+ n=2", screen)
    assert "wrote 2 rows to" in screen

    sidecar = out_csv + ".timing"
    assert os.path.exists(sidecar), "interactive run dropped the timing records"
    body = open(sidecar).read()
    assert "# per-pair seconds" in body
    assert [pid for pid, _ in re.findall(r"^# t,([^,]+),([0-9.]+)$", body, re.M)] \
        == ["ta", "tb"]


def test_nested_output_directory_is_created(tiny_pairs, tmp_path):
    """A nested --output path must work when its directory does not exist.

    Issue #52: an interim revision of this branch deleted the
    `os.makedirs(os.path.dirname(...))` call, so `--output results/preds.csv`
    raised FileNotFoundError and lost the entire run. The judge names the
    output path, so this is not a hypothetical.
    """
    out_csv = str(tmp_path / "results" / "nested" / "predictions.csv")
    assert not os.path.exists(os.path.dirname(out_csv))
    p = subprocess.run(
        [sys.executable, REGISTER, "--input", tiny_pairs, "--output", out_csv],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert p.returncode == 0, p.stderr
    assert os.path.exists(out_csv), "nested output directory was not created"
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["pair_id"] for r in rows] == ["ta", "tb"]
