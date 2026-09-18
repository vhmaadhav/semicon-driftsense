"""Phase 3: the training-side ``params.json`` schema.

The Phase 3 ``pairs.csv`` carries a ``params_json_path`` column that the
**blind split leaves empty**. It is a training-time artefact: it records the
generation parameters for a site, including the per-layer brightness the
generator chose, so a model can be *supervised* against the thing the task asks
it to infer.

Two properties matter, and they are why this is a module rather than a
``json.load`` call at each use site:

1. **The inference path must never need it.** ``read_params`` is never called
   from ``phase3.py``. The blind split's field is empty by construction, and the
   deck is explicit that code needing it must fail on data we hold rather than
   on the scored run.

2. **Per-layer brightness is only meaningful where that layer is observable.**
   Two of the eight layers are fully occluded under painter's-algorithm
   compositing (the capacitor is a strict superset of the storage contact; via1
   is a strict subset of the metal2 strap), so their brightness cannot be
   recovered from the rendered image at all. Recording it is still correct --
   the file documents what the generator did -- but ``observable_layers`` is
   reported alongside so a supervision target is never built on a layer the
   image cannot show.
"""

from __future__ import annotations

import json
import os

# The schema version this module reads and writes. Bump on an incompatible
# change so an old file is rejected loudly rather than parsed into defaults.
SCHEMA_VERSION = 1

REQUIRED_KEYS = ("schema_version", "architecture", "num_layers",
                 "layer_intensities")

# Layers that painter's-algorithm compositing hides entirely, per architecture.
# Established by measurement, not assumption: see docs/PHASE3_MEASUREMENT.md.
# The capacitor (5) is drawn at 1.8x the radius of the storage contact (4) at
# the same centre, so it is a strict superset; via1 (6) sits under the metal2
# strap (7) at 0.6x radius, so it is a strict subset.
OCCLUDED_LAYERS = {
    "dram": (4, 6),
    "finfet": (6,),
}


class ParamsError(ValueError):
    """A params.json is missing, unreadable, or does not match the schema."""


def observable_layers(architecture: str, num_layers: int = 8) -> tuple:
    """Layer indices whose brightness the rendered image can actually show."""
    hidden = set(OCCLUDED_LAYERS.get(architecture.lower(), ()))
    return tuple(i for i in range(num_layers) if i not in hidden)


def build_params(architecture: str, num_layers: int,
                 layer_intensities: dict,
                 reference_gds_path: str = "",
                 search_gds_path: str = "",
                 present: bool = True,
                 rotation_deg: float = 0.0,
                 scale: float = 10.0,
                 generation: dict | None = None) -> dict:
    """Assemble a params dict. Never invents intensity values."""
    if not isinstance(layer_intensities, dict) or not layer_intensities:
        raise ParamsError("layer_intensities must be a non-empty mapping")
    clean = {}
    for k, v in layer_intensities.items():
        iv = int(v)
        if not 0 <= iv <= 255:
            raise ParamsError(f"layer {k}: intensity {iv} outside 0..255")
        clean[int(k)] = iv
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": str(architecture).lower(),
        "num_layers": int(num_layers),
        "layer_intensities": clean,
        # Which of those the image can actually reveal. A supervision target
        # must be restricted to these.
        "observable_layers": list(observable_layers(architecture, num_layers)),
        "occluded_layers": sorted(set(OCCLUDED_LAYERS.get(
            str(architecture).lower(), ()))),
        "reference_gds_path": reference_gds_path,
        "search_gds_path": search_gds_path,
        "present": bool(present),
        "rotation_deg": float(rotation_deg),
        "scale": float(scale),
        "generation": dict(generation or {}),
    }


def write_params(path: str, params: dict) -> str:
    """Write a params dict, validating it first."""
    _validate(params)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _validate(d: dict) -> None:
    if not isinstance(d, dict):
        raise ParamsError(f"expected a JSON object, got {type(d).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise ParamsError(f"missing required key(s): {missing}")
    if d["schema_version"] != SCHEMA_VERSION:
        raise ParamsError(
            f"schema_version {d['schema_version']!r} is not supported "
            f"(this reader implements {SCHEMA_VERSION})")
    li = d["layer_intensities"]
    if not isinstance(li, dict) or not li:
        raise ParamsError("layer_intensities must be a non-empty object")
    for k, v in li.items():
        try:
            iv = int(v)
        except (TypeError, ValueError) as exc:
            raise ParamsError(f"layer {k!r}: intensity {v!r} is not an int") from exc
        if not 0 <= iv <= 255:
            raise ParamsError(f"layer {k!r}: intensity {iv} outside 0..255")
    if int(d["num_layers"]) < 1:
        raise ParamsError(f"num_layers must be >= 1, got {d['num_layers']!r}")


def read_params(path: str) -> dict:
    """Read and validate a params.json.

    Training-time only. ``phase3.py`` must never call this: on the blind split
    the path is empty, and depending on it would fail on the scored run.
    """
    if not path:
        raise ParamsError("no params path given")
    if not os.path.isfile(path):
        raise ParamsError(f"params file not found: {path!r}")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except json.JSONDecodeError as exc:
        raise ParamsError(f"{path!r} is not valid JSON: {exc}") from exc
    _validate(d)
    # JSON object keys are always strings, so a round-tripped file has
    # {"0": 51} where build_params produced {0: 51}. Normalise back to int keys
    # so in-memory and on-disk forms compare equal and no caller has to know
    # which one it is holding.
    d["layer_intensities"] = {int(k): int(v)
                              for k, v in d["layer_intensities"].items()}
    return d


def intensity_vector(params: dict, num_layers: int | None = None) -> list:
    """``layer_intensities`` as a dense list indexed by layer (0 if absent)."""
    n = int(num_layers if num_layers is not None else params["num_layers"])
    li = params["layer_intensities"]
    return [int(li.get(str(i), li.get(i, 0))) for i in range(n)]
