"""H-02 of the static audit: in sampled mode, verify_dataset checked
``n not in pick`` *before* the file-existence test, so unselected rows'
missing files were never reported despite the script's stated full-manifest
file check. Existence must be verified for every row; only image decoding
and pixel checks are sampled.
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

cv2 = pytest.importorskip("cv2")

ROW_FIELDS = ["id", "architecture", "reference_path", "search_path",
              "reference_px", "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
              "gt_x_corr", "gt_y_corr", "label_shift_px"]


def _row(idx):
    return {"id": f"{idx:05d}", "architecture": "finfet_10nm",
            "reference_path": f"reference/r{idx}.png",
            "search_path": f"search/s{idx}.png", "reference_px": "1000",
            "gt_box_x": "10", "gt_box_y": "10", "gt_box_w": "100",
            "gt_box_h": "100", "gt_x_corr": "500", "gt_y_corr": "500",
            "label_shift_px": "0.0"}


def _load_script():
    path = os.path.join(REPO_ROOT, "scripts", "verify_dataset.py")
    spec = importlib.util.spec_from_file_location("verify_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_split(tmp_path, rows, write):
    for row in rows:
        for key in ("reference_path", "search_path"):
            if write.get((row["id"], key), True):
                p = tmp_path / row[key]
                p.parent.mkdir(parents=True, exist_ok=True)
                img = (np.zeros((100, 100), np.uint8)
                       if row["reference_px"] == "100" and key == "reference_path"
                       else np.zeros((1000, 1000), np.uint8))
                assert cv2.imwrite(str(p), img)
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_missing_file_reported_even_when_row_not_sampled(tmp_path, capsys, monkeypatch):
    mod = _load_script()
    rows = [_row(0), _row(1)]
    # Row 1's reference file is missing on disk.
    _make_split(tmp_path, rows, write={("00001", "reference_path"): False})

    # Sample exactly one row and force it to be row 0, so row 1 is only
    # covered by the exhaustive existence check.
    monkeypatch.setattr(mod.np.random, "default_rng",
                        lambda seed: SimpleNamespace_choice())

    ok = mod.check_split(str(tmp_path), sample=1, seed=0)
    out = capsys.readouterr().out
    assert not ok
    assert "missing reference_path" in out, out


class SimpleNamespace_choice:
    def choice(self, n, size, replace=False):
        return np.array([0])
