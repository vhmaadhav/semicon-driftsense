"""H-01 of the static audit: the streaming dataset assigned every worker
``length // nworkers``, so any remainder silently vanished while ``__len__``
kept reporting the full length. Quotas must sum back to exactly ``length``.
"""

import sys
import types
from types import SimpleNamespace

import pytest

rs = pytest.importorskip("driftsense.stream_dataset")
StreamingDriftSense = rs.StreamingDriftSense


@pytest.mark.parametrize("length,nworkers", [
    (10, 4), (7, 3), (100, 8), (5, 8), (1, 1), (9, 1),
])
def test_worker_quota_sums_to_length(length, nworkers):
    quotas = [rs.worker_quota(length, nworkers, wid) for wid in range(nworkers)]
    assert sum(quotas) == length, quotas
    assert max(quotas) - min(quotas) <= 1, quotas
    for wid in range(nworkers):
        assert rs.worker_quota(length, nworkers, wid) == quotas[wid]


def test_worker_quota_single_worker_gets_everything():
    assert rs.worker_quota(14000, 1, 0) == 14000


def test_iteration_yields_exactly_length_across_workers():
    """End-to-end: four faked loader workers together yield __len__ samples.
    (The old floor-division dropped the remainder: 10 // 4 == 2 per worker,
    8 total.)"""
    length, nworkers = 10, 4
    total = 0
    for wid in range(nworkers):
        ds = StreamingDriftSense(length=length, crop=128,
                                 crops_per_canvas=4, noise="low",
                                 architectures=["finfet_10nm"], seed=0)
        fake_info = SimpleNamespace(id=wid, num_workers=nworkers)
        orig_info = rs.get_worker_info
        rs.get_worker_info = lambda info=fake_info: info
        try:
            for _ in ds:
                total += 1
        finally:
            rs.get_worker_info = orig_info
    assert total == length
