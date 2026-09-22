"""The Phase 2 v2 validation-set generator (issue #85).

The plan is pure numpy and always tested. Generating real pairs needs the
mentor's confidential generator under phase2_v2/generator, which is
git-ignored, so that part skips cleanly where it is absent.
"""
import csv
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import gen_phase2_v2_val as G  # noqa: E402

MENTOR_DIR = os.path.join(REPO_ROOT, "phase2_v2", "generator")


def test_set_counts_reproduce_the_mentor_proportions():
    assert G.set_counts(25) == {"A": 9, "B": 9, "C": 5, "D": 2}
    for n in (1, 7, 48, 500, 1001):
        counts = G.set_counts(n)
        assert sum(counts.values()) == n
        assert abs(counts["C"] - 0.2 * n) <= 1


def test_plan_is_deterministic_and_seed_dependent():
    a, b, c = G.build_plan(60, 7), G.build_plan(60, 7), G.build_plan(60, 8)
    assert a == b
    assert a != c


def test_plan_respects_the_set_structure_and_disclosed_bounds():
    plan = G.build_plan(500, 850001)
    assert len({r["pair_id"] for r in plan}) == 500
    assert [r["index"] for r in plan] == list(range(500))
    for r in plan:
        assert r["severity"] in G.SET_SEVERITIES[r["set"]]
        assert r["present"] == (r["set"] != "C")
        assert 8.0 <= r["zoom"] <= 12.0
        assert -5.0 <= r["theta"] <= 5.0
        assert r["architecture"] in G.PRESETS
    for s in "ABC":
        rows = [r for r in plan if r["set"] == s]
        assert {r["architecture"] for r in rows} == set(G.PRESETS)       # every preset per set
        levels = [r["severity"] for r in rows]
        assert max(levels.count(v) for v in set(levels)) - min(levels.count(v) for v in set(levels)) <= 1


def test_plan_can_stress_absent_severities():
    plan = G.build_plan(100, 3, c_severities=(0, 1, 2, 3, 4))
    assert {r["severity"] for r in plan if r["set"] == "C"} == {0, 1, 2, 3, 4}


_SMOKE = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import gen_phase2_v2_val as G
out, mentor = sys.argv[2], sys.argv[3]
for sub in ("reference", "search", "records"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)
G._init_worker(mentor)
plan = G.build_plan(25, 11)
picks = [next(r for r in plan if r["set"] == s) for s in ("A", "C", "D")]
recs = [G._generate((r, out, 11, 20)) for r in picks]
again = G._generate((picks[0], out, 11, 20))
print(json.dumps({"picks": picks, "recs": recs, "again": again}))
"""


@pytest.mark.skipif(not os.path.isfile(os.path.join(MENTOR_DIR, "generate_phase2_dataset_v2.py")),
                    reason="mentor v2 generator not present (confidential, git-ignored)")
def test_generated_pairs_use_the_mentor_dataset_format(tmp_path):
    # A clean interpreter: the mentor's generator is a top-level `src` package,
    # and this test process already has the vendored generator/src on sys.path.
    import json
    import subprocess
    out = str(tmp_path)
    proc = subprocess.run([sys.executable, "-c", _SMOKE, os.path.join(REPO_ROOT, "scripts"), out, MENTOR_DIR],
                          capture_output=True, text=True, timeout=600, check=True)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    picks, recs = result["picks"], result["recs"]
    assert not any(r["failed"] for r in recs)
    # a second call is served from the resumable record, not regenerated
    assert result["again"] == recs[0]

    import cv2
    for rec, row in zip(recs, picks):
        ref = cv2.imread(os.path.join(out, rec["pairs"]["reference_path"]), cv2.IMREAD_UNCHANGED)
        sea = cv2.imread(os.path.join(out, rec["pairs"]["search_path"]), cv2.IMREAD_UNCHANGED)
        assert ref.shape[:2] == sea.shape[:2] == (1000, 1000)
        assert (sea.ndim == 3) == (row["set"] == "D")
        assert set(rec["gt"]) == set(G.GT_FIELDS)
        assert set(rec["jury"]) == set(G.JURY_FIELDS)
        assert rec["gt"]["present"] == int(row["present"])
        if row["present"]:
            assert rec["gt"]["scale"] == pytest.approx(row["zoom"], abs=1e-3)
            assert rec["gt"]["theta"] == pytest.approx(row["theta"], abs=1e-3)
            assert 0 < rec["gt"]["x"] < 1000 and 0 < rec["gt"]["y"] < 1000
        else:
            assert rec["gt"]["x"] == rec["gt"]["y"] == 0.0

    # Same header as the mentor's own files, so score_phase2_v2.py reads it unchanged.
    mentor_gt = os.path.join(REPO_ROOT, "phase2_v2", "ground_truth.csv")
    mentor_jury = os.path.join(REPO_ROOT, "phase2_v2", "manifest_jury.csv")
    if os.path.isfile(mentor_gt) and os.path.isfile(mentor_jury):
        with open(mentor_gt, newline="") as f:
            assert next(csv.reader(f)) == G.GT_FIELDS
        with open(mentor_jury, newline="") as f:
            assert next(csv.reader(f)) == G.JURY_FIELDS
