"""Issue #21: the template-feature cache is keyed on the input bytes, not on
the model's parameter state. If weights change under a frozen reference (e.g.
load_state_dict between two eval passes), the cache must be invalidated --
otherwise the second pass silently serves the first model's embedding.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.model import DriftSenseNet  # noqa: E402


def test_load_state_dict_invalidates_template_cache():
    net = DriftSenseNet().eval()
    ref = torch.rand(1, 1, 100, 100)
    search = torch.rand(1, 1, 400, 400)
    with torch.no_grad():
        net(ref, search)
    assert net._tf_cache is not None, "cache should be populated after a pass"

    other = DriftSenseNet().eval()
    net.load_state_dict(other.state_dict())
    assert net._tf_cache is None, "load_state_dict must invalidate the cache"


def test_stale_embedding_is_not_served_after_weights_change():
    """Behavioural form: the same input must produce the NEW model's output
    after load_state_dict, not the cached old embedding."""
    net = DriftSenseNet().eval()
    ref = torch.rand(1, 1, 100, 100)
    search = torch.rand(1, 1, 400, 400)
    with torch.no_grad():
        out_old = net(ref, search)
        other = DriftSenseNet().eval()
        expected_new = other(ref, search)
        net.load_state_dict(other.state_dict())
        out_new = net(ref, search)  # same input tensor as out_old's call
    assert not torch.allclose(out_old["logit"], expected_new["logit"])
    assert torch.allclose(out_new["logit"], expected_new["logit"], atol=1e-6)
