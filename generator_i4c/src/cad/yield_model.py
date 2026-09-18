"""
Secondary-electron yield model for CAD-derived rendering.

Real SEM contrast comes from secondary-electron (SE) yield: how many
secondary electrons a material emits per incident beam electron, which
depends on material and local topology. Higher-yield surfaces collect more
electrons at the detector and render brighter.

Phase 3's rule: nobody hardcodes a grayscale value per layer name (the
Phase 1/2 pattern generators do exactly that -- WORD_LINE_VAL = 150, etc).
Instead, each GDS layer is just a number, and brightness is *derived* from
that number through a yield model -- the render step has to "figure out"
the yield the same way a real tool operator would reason about a stack:
layers stacked higher (closer to the beam, less overlying material to
attenuate the signal) generically collect more signal than deeply buried
layers, so yield trends upward with layer number. This is illustrative of
the real physical trend, not a measured material database.
"""

from __future__ import annotations

BASE_YIELD = 0.20      # illustrative SE yield of the lowest/most-buried layer
TOP_YIELD = 0.85        # illustrative SE yield of the topmost layer
BACKGROUND_YIELD = 0.12  # bare substrate / field regions with no polygon


def layer_yield(layer_number: int, num_layers: int) -> float:
    """Derive a layer's secondary-electron yield purely from its numeric
    position in the stack (0 = lowest/most-buried, num_layers-1 = topmost).
    Monotonically increasing -- upper layers are generically less attenuated
    by overlying material, so they read out a stronger SE signal.
    """
    if num_layers <= 1:
        return TOP_YIELD
    frac = layer_number / (num_layers - 1)
    return BASE_YIELD + (TOP_YIELD - BASE_YIELD) * frac


def yield_to_intensity(yield_value: float) -> int:
    """Map a [0, 1] SE yield to an 8-bit grayscale intensity."""
    return int(round(max(0.0, min(1.0, yield_value)) * 255))


def background_intensity() -> int:
    return yield_to_intensity(BACKGROUND_YIELD)
