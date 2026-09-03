"""M-07 of the static audit: `presets_for_kind` mapped every kind other than
"dram" onto the FinFET preset list, so a typo like "finfet_" or "sram"
silently generated the wrong device family. Unknown kinds must raise.
"""

import sys, os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from generator.src.presets import presets_for_kind  # noqa: E402


def test_known_kinds_return_distinct_lists():
    dram = presets_for_kind("dram")
    finfet = presets_for_kind("finfet")
    assert dram and finfet
    assert {tuple(sorted(p.items())) for p in dram}.isdisjoint(
        {tuple(sorted(p.items())) for p in finfet})


@pytest.mark.parametrize("kind", ["sram", "finfet_", "DRAM", "", "fin fet"])
def test_unknown_kinds_raise(kind):
    with pytest.raises(ValueError):
        presets_for_kind(kind)
