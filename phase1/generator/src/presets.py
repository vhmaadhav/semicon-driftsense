"""
Structure presets, in nanometers.

These are illustrative of publicly known industry scaling trends (6F^2 DRAM
cell architecture, FinFET fin/gate pitch scaling) -- NOT exact proprietary
fab specifications. Override any field via CLI/config for other nodes.

Feature sizes here are intentionally on the larger/coarser end of what's
publicly discussed for these architectures: at 1 px = 1 nm reference
resolution, very small illustrative geometry (sub-20nm F) renders as only a
handful of pixels wide, which is hard to see clearly in a teaching context.
Bigger F keeps structures crisp on-screen while the *relationships* (2F/3F
pitch ratios, fin/gate pitch scaling) stay physically consistent.
"""

DRAM_1X = {
    "kind": "dram",
    "feature_size_nm": 32,          # F
    "word_line_pitch_nm": 64,       # ~2F
    "word_line_width_nm": 32,       # ~F
    "bit_line_pitch_nm": 96,        # ~3F  (6F^2 folded-bitline cell = 2F x 3F)
    "bit_line_width_nm": 32,        # ~F
    "contact_diameter_nm": 32,      # ~F, storage-node landing pad
}

DRAM_DENSE = {
    "kind": "dram",
    "feature_size_nm": 24,
    "word_line_pitch_nm": 48,
    "word_line_width_nm": 24,
    "bit_line_pitch_nm": 72,
    "bit_line_width_nm": 24,
    "contact_diameter_nm": 24,
}

DRAM_LOOSE = {
    "kind": "dram",
    "feature_size_nm": 48,
    "word_line_pitch_nm": 96,
    "word_line_width_nm": 48,
    "bit_line_pitch_nm": 144,
    "bit_line_width_nm": 48,
    "contact_diameter_nm": 48,
}

DRAM_WIDE = {
    "kind": "dram",
    "feature_size_nm": 60,
    "word_line_pitch_nm": 120,
    "word_line_width_nm": 56,
    "bit_line_pitch_nm": 180,
    "bit_line_width_nm": 60,
    "contact_diameter_nm": 58,
}

DRAM_COMPACT = {
    "kind": "dram",
    "feature_size_nm": 36,
    "word_line_pitch_nm": 72,
    "word_line_width_nm": 30,
    "bit_line_pitch_nm": 108,
    "bit_line_width_nm": 34,
    "contact_diameter_nm": 30,
}

DRAM_LEGACY = {
    "kind": "dram",
    "feature_size_nm": 80,
    "word_line_pitch_nm": 160,
    "word_line_width_nm": 78,
    "bit_line_pitch_nm": 240,
    "bit_line_width_nm": 80,
    "contact_diameter_nm": 78,
}

FINFET_10NM = {
    "kind": "finfet",
    "fin_pitch_nm": 48,
    "fin_width_nm": 16,
    "gate_pitch_nm": 90,             # contacted poly pitch (CPP)
    "gate_length_nm": 28,
    "contact_size_nm": 28,
}

FINFET_7NM = {
    "kind": "finfet",
    "fin_pitch_nm": 40,
    "fin_width_nm": 14,
    "gate_pitch_nm": 76,
    "gate_length_nm": 24,
    "contact_size_nm": 24,
}

FINFET_14NM = {
    "kind": "finfet",
    "fin_pitch_nm": 60,
    "fin_width_nm": 20,
    "gate_pitch_nm": 110,
    "gate_length_nm": 34,
    "contact_size_nm": 34,
}

FINFET_22NM = {
    "kind": "finfet",
    "fin_pitch_nm": 80,
    "fin_width_nm": 26,
    "gate_pitch_nm": 150,
    "gate_length_nm": 46,
    "contact_size_nm": 44,
}

FINFET_28NM = {
    "kind": "finfet",
    "fin_pitch_nm": 96,
    "fin_width_nm": 32,
    "gate_pitch_nm": 180,
    "gate_length_nm": 56,
    "contact_size_nm": 52,
}

FINFET_45NM = {
    "kind": "finfet",
    "fin_pitch_nm": 140,
    "fin_width_nm": 46,
    "gate_pitch_nm": 260,
    "gate_length_nm": 80,
    "contact_size_nm": 76,
}

PRESETS = {
    "dram_1x": DRAM_1X,
    "dram_dense": DRAM_DENSE,
    "dram_loose": DRAM_LOOSE,
    "dram_wide": DRAM_WIDE,
    "dram_compact": DRAM_COMPACT,
    "dram_legacy": DRAM_LEGACY,
    "finfet_10nm": FINFET_10NM,
    "finfet_7nm": FINFET_7NM,
    "finfet_14nm": FINFET_14NM,
    "finfet_22nm": FINFET_22NM,
    "finfet_28nm": FINFET_28NM,
    "finfet_45nm": FINFET_45NM,
}

DRAM_PRESET_NAMES = ["dram_1x", "dram_dense", "dram_loose", "dram_wide", "dram_compact", "dram_legacy"]
FINFET_PRESET_NAMES = ["finfet_10nm", "finfet_7nm", "finfet_14nm", "finfet_22nm", "finfet_28nm", "finfet_45nm"]


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return dict(PRESETS[name])


def presets_for_kind(kind: str):
    names = DRAM_PRESET_NAMES if kind == "dram" else FINFET_PRESET_NAMES
    return [get_preset(n) for n in names]
