"""The Set B severity ladder must never silently render the opposite of what
was asked for (issue #31).

`PoseSpec.severity` and `PoseSpec.polygon_scale` disable themselves when
`hi <= lo`, and when disabled make *no random draw at all*. That is deliberate:
it is what keeps the Phase 1 splits byte-for-byte reproducible.

The trap it creates is that pinning a level with equal endpoints
(`--severity-range 1.0 1.0`, the obvious way to ask for "severity 4 only")
silently produces `severity_continuous = 0.0` -- the EASIEST possible data --
on pairs whose manifest still reports the level that was requested.

This is the defect that retracted the 81.45 / 81.93 headline figures (see
README, "Retired: the 81.45 / 81.93 figures"). It was fixed in the Issue 45
audit fixture with a 1e-6 epsilon, but the core generator kept the strict
comparison, so `generate_dataset.py --severity-range` still walked into it.

Compliance item G2 ("weight severity 3-4 in the next generator run") asks for
exactly this kind of regeneration, so the trap sits directly in the path of
that work. These tests pin the guard that closes it.
"""

import numpy as np
import pytest

from driftsense.generate import SEVERITY_LADDER, sample_severity_params


def _root_generate_dataset():
    """Import the REPO-ROOT generate_dataset.py explicitly.

    conftest.py puts `generator/` ahead of the repo root on sys.path, and there
    are four generate_dataset.py files in this tree (root, generator/, phase1/,
    phase1/generator/). A plain `import generate_dataset` under pytest resolves
    to the vendored generator's copy, which has no build_pose_spec -- so load
    the one whose CLI we are actually pinning, by path.
    """
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "generate_dataset.py")
    spec = importlib.util.spec_from_file_location("_root_generate_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spec_from(argv):
    """Run generate_dataset.py's real CLI -> PoseSpec path on an argv list."""
    import sys

    G = _root_generate_dataset()
    old = sys.argv
    try:
        sys.argv = ["generate_dataset.py", *argv]
        return G.build_pose_spec(G.parse_args())
    finally:
        sys.argv = old


@pytest.mark.parametrize("flag,value", [
    ("--severity-range", "1.0"),
    ("--severity-range", "0.75"),
    ("--polygon-scale-range", "0.1"),
])
def test_equal_endpoints_are_rejected_rather_than_silently_disabled(flag, value):
    """The whole point: asking for a pinned level must fail loudly instead of
    quietly producing nominal data under a harder label."""
    with pytest.raises(SystemExit) as e:
        _spec_from(["--num-pairs", "1", "--output-dir", "unused", flag, value, value])
    msg = str(e.value)
    # The message has to say what went wrong AND how to fix it, or it just
    # moves the confusion somewhere else.
    assert "DISABLE" in msg
    assert flag in msg


def test_zero_zero_remains_the_documented_off_idiom():
    """`--polygon-scale-range 0 0` is documented in --phase2's help as the way
    to get a Set A-style split. It must keep working."""
    spec = _spec_from(["--num-pairs", "1", "--output-dir", "unused", "--phase2",
                       "--polygon-scale-range", "0", "0"])
    assert spec.polygon_scale == (0.0, 0.0)


def test_a_real_range_is_accepted_and_reaches_the_spec():
    spec = _spec_from(["--num-pairs", "1", "--output-dir", "unused", "--phase2",
                       "--severity-range", "0.75", "1.0"])
    assert spec.severity == (0.75, 1.0)


def test_inverted_range_still_rejected_for_severity():
    """--severity-range was missing from the pre-existing hi<lo validation loop."""
    with pytest.raises(SystemExit):
        _spec_from(["--num-pairs", "1", "--output-dir", "unused",
                    "--severity-range", "1.0", "0.2"])


def test_high_severity_band_actually_degrades_harder_than_a_low_one():
    """Guards the direction of the ladder itself, so a future refactor cannot
    invert it without a test failing. Uses the sampler directly -- no I/O."""
    rng_hi = np.random.default_rng(11)
    rng_lo = np.random.default_rng(11)
    hi = [sample_severity_params(rng_hi, (0.75, 1.0)) for _ in range(24)]
    lo = [sample_severity_params(rng_lo, (0.0, 0.25)) for _ in range(24)]

    assert np.mean([h["severity_continuous"] for h in hi]) > \
           np.mean([l["severity_continuous"] for l in lo])
    # Every knob on the ladder is monotone in the latent severity, so each one
    # must be worse on average in the high band.
    for knob in SEVERITY_LADDER:
        assert np.mean([h[knob] for h in hi]) > np.mean([l[knob] for l in lo]), knob
