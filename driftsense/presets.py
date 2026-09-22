"""Architecture-family -> preset-name mapping.

The upstream generator exposes twelve node presets. The problem statement
talks in terms of two families (DRAM memory arrays and FinFET gate
structures), so this maps a family name onto the presets that belong to it.
Sampling across several nodes within a family, rather than one fixed node,
keeps a model from memorising a single pitch.
"""

from __future__ import annotations

from driftsense._vendored import ensure_generator_on_path

ensure_generator_on_path()

from src.presets import DRAM_PRESET_NAMES, FINFET_PRESET_NAMES  # noqa: E402

FAMILIES = {
    "dram": list(DRAM_PRESET_NAMES),
    "finfet": list(FINFET_PRESET_NAMES),
    "mixed": list(DRAM_PRESET_NAMES) + list(FINFET_PRESET_NAMES),
}


def architecture_presets(family: str) -> list[str]:
    key = family.lower()
    if key not in FAMILIES:
        raise ValueError(f"unknown architecture '{family}'. Choose from {list(FAMILIES)}")
    return list(FAMILIES[key])
