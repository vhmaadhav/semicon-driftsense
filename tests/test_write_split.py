"""C-05 of the static audit: dataset output must be an exact contract.

- `write_split(max_pairs=...)` must produce exactly that many manifest rows
  (the old ceil-canvases x all-crops behavior overshot the request), without
  generating orphaned images for dropped rows.
- Image writes must be checked: a failed `cv2.imwrite` must raise, not leave
  a manifest row pointing at a missing file.
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

gen = pytest.importorskip("driftsense.generate")


def _read_manifest(split_dir):
    with open(os.path.join(split_dir, "manifest.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def _fake_build_one_factory(recorded):
    """Stands in for the expensive canvas builder: honours the per-job crop
    budget and returns that many rows, so write_split's dispatch and manifest
    writing are exercised without generating a single pixel."""

    def fake_build_one(job):
        idx, entropy, architectures, noise, dirs, crops, store_templates, pose = job
        recorded.append(crops)
        return [{"id": f"{idx:05d}_{c}", "canvas_id": idx} for c in range(crops)]

    return fake_build_one


def test_write_split_exact_pair_count(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(gen, "build_one", _fake_build_one_factory(recorded))
    pairs = gen.write_split(str(tmp_path), num_canvases=3, seed=1,
                            noise="clean", architectures=["finfet"],
                            workers=0, crops_per_canvas=2, progress_every=0,
                            max_pairs=5)
    assert pairs == 5
    rows = _read_manifest(tmp_path)
    assert len(rows) == 5
    # Two full canvases (2 crops each) plus one partial canvas (1 crop):
    assert recorded == [2, 2, 1], recorded


def test_write_split_without_cap_unchanged(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(gen, "build_one", _fake_build_one_factory(recorded))
    pairs = gen.write_split(str(tmp_path), num_canvases=2, seed=1,
                            noise="clean", architectures=["finfet"],
                            workers=0, crops_per_canvas=3, progress_every=0)
    assert pairs == 6
    assert recorded == [3, 3]


def test_write_split_cap_multiple_of_crops(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(gen, "build_one", _fake_build_one_factory(recorded))
    pairs = gen.write_split(str(tmp_path), num_canvases=5, seed=1,
                            noise="clean", architectures=["finfet"],
                            workers=0, crops_per_canvas=2, progress_every=0,
                            max_pairs=4)
    assert pairs == 4
    assert recorded == [2, 2]
    assert len(_read_manifest(tmp_path)) == 4


def test_failed_imwrite_raises(tmp_path, monkeypatch):
    import cv2

    def failing_imwrite(path, img):
        return False

    monkeypatch.setattr(cv2, "imwrite", failing_imwrite)
    with pytest.raises(RuntimeError, match="failed to write"):
        gen._write_png(str(tmp_path / "x.png"), np.zeros((4, 4), np.uint8))
