"""Phase 3: rasterize a Reference GDSII design into the matching pipeline.

Phase 3 changes what the *reference* is: a ``.gds`` design file instead of an
SEM image. Everything downstream of the reference is unchanged -- the search
image, the answer format, the matching policy. This module is the only new
piece of the read path: ``.gds`` -> a 1000x1000 grayscale raster in exactly the
convention ``driftsense.matching`` already consumes.

Design constraints this module exists to satisfy
------------------------------------------------

* **Brightness is derived from the layer number, never hard-coded per layer
  name.** A GDS layer is just an integer; there is no material table in the
  file. The generator that produced the Phase 3 data derives a layer's
  secondary-electron yield from its *position in the stack* (higher layers are
  less attenuated by overlying material, so they read brighter), and the
  matcher has to invert the same rule. This mirrors the organizer-side
  generator's model rather than inventing a second one.

* **Fail closed.** ``gdstk`` is a new runtime dependency and the graded run has
  no network. A missing wheel, an unreadable file, or a GDS with no polygons
  must abort loudly -- never degrade into a per-pair declined row, which is the
  silent-failure shape the Phase 2 guardrails exist to prevent (a declined row
  is well-formed, scores zero, and exits 0).

* **No layer-name semantics.** Nothing here branches on what a layer is
  *called*. The same code path handles DRAM and FinFET files.
"""

from __future__ import annotations

import numpy as np

# Reference frame geometry, fixed by the problem statement and shared with
# driftsense.model (REF_SIZE = 1000 at 1 nm/px).
REF_SIZE = 1000

# Secondary-electron yield ladder, matching the organizer-side CAD generator
# (src/cad/yield_model.py: BASE_YIELD / TOP_YIELD / BACKGROUND_YIELD). A layer's
# grey level is derived from its index in the stack, not from its name.
BASE_YIELD = 0.20        # lowest / most buried layer
TOP_YIELD = 0.85         # topmost layer
BACKGROUND_YIELD = 0.12  # bare substrate / field with no polygon


class GdsError(RuntimeError):
    """A reference GDS could not be turned into a raster."""


def _require_gdstk():
    """Import gdstk lazily and fail closed with an actionable message."""
    try:
        import gdstk  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise GdsError(
            "Phase 3 needs the 'gdstk' package to read reference .gds files, "
            "and it is not importable. It is pinned in requirements.txt; "
            f"install it into the runtime environment. Original error: {exc!r}"
        ) from exc
    return gdstk


def layer_yield(layer_number: int, num_layers: int) -> float:
    """Yield for a layer, from its stack position (0 = most buried)."""
    if num_layers <= 1:
        return TOP_YIELD
    frac = layer_number / (num_layers - 1)
    return BASE_YIELD + (TOP_YIELD - BASE_YIELD) * frac


def yield_to_intensity(yield_value: float) -> int:
    return int(round(max(0.0, min(1.0, yield_value)) * 255))


def background_intensity() -> int:
    return yield_to_intensity(BACKGROUND_YIELD)


def read_gds_layers(path: str) -> tuple:
    """Return ``(polygons_by_layer, num_layers)`` for a reference GDS.

    ``polygons_by_layer`` maps a layer index to a list of ``(N, 2)`` float64
    arrays in design coordinates (nm, 1 nm/px at the reference scale).

    **Only datatype 0 is read.** A GDS layer number is paired with a datatype,
    and readers conventionally treat ``(layer, 0)`` as the drawn geometry while
    other datatypes carry annotations, fill, or DRC markers. The organizer-side
    renderer filters ``datatype=0``; reading every datatype instead would paint
    geometry the reference does not actually contain, and would silently break
    the bit-identical property against that renderer. A non-zero datatype is
    counted and reported rather than dropped in silence.
    """
    gdstk = _require_gdstk()
    try:
        lib = gdstk.read_gds(path)
    except Exception as exc:  # noqa: BLE001
        raise GdsError(f"could not read GDS {path!r}: {exc!r}") from exc

    cells = lib.top_level()
    if not cells:
        raise GdsError(f"{path!r} has no top-level cell")

    # A reference may legitimately span more than one top-level cell; merge
    # them rather than silently using the first, so a multi-cell file does not
    # produce a half-empty raster.
    by_layer: dict[int, list] = {}
    skipped_datatypes: set = set()
    for cell in cells:
        for poly in cell.get_polygons():
            if int(poly.datatype) != 0:
                # Not drawn geometry -- do not paint it.
                skipped_datatypes.add((int(poly.layer), int(poly.datatype)))
                continue
            by_layer.setdefault(int(poly.layer), []).append(
                np.asarray(poly.points, dtype=np.float64))
    if not by_layer:
        extra = (f" (only non-zero datatypes present: {sorted(skipped_datatypes)})"
                 if skipped_datatypes else "")
        raise GdsError(f"{path!r} contains no datatype-0 polygons{extra}")

    num_layers = max(by_layer) + 1
    return by_layer, num_layers


def rasterize_layers(polygons_by_layer: dict, num_layers: int,
                     size=REF_SIZE,
                     layer_intensities: dict | None = None,
                     offset: tuple = (0.0, 0.0),
                     min_layer: int = 0) -> np.ndarray:
    """Paint layers bottom-to-top onto a uint8 canvas.

    Painter's algorithm with direct assignment, matching the organizer-side
    renderer: a higher layer overwrites a lower one where they overlap, because
    a real SEM only sees the top surface.

    ``size`` is either an int (square canvas -- the reference convention) or an
    explicit ``(width, height)`` pair. The pair form matters for layout work: a
    mat is not square in general, and rasterizing a non-square mat onto a square
    canvas of ``max(w, h)`` then slicing ``[:h, :w]`` silently drops every
    polygon beyond the slice. Geometry present in the search image would then be
    missing from the reference clipped out of the same mat, and the two sides
    would stop describing the same scene.

    ``offset`` shifts design coordinates before rasterizing (used to render a
    sub-window). ``min_layer`` drops everything below an index, which models an
    acquisition that does not resolve that depth.
    """
    import cv2  # noqa: PLC0415  (already a hard dependency of the matcher)

    if isinstance(size, (tuple, list)):
        width, height = int(size[0]), int(size[1])
    else:
        width = height = int(size)
    if width <= 0 or height <= 0:
        raise GdsError(f"size must be positive, got {(width, height)}")
    canvas = np.full((height, width), background_intensity(), dtype=np.uint8)
    ox, oy = float(offset[0]), float(offset[1])

    for layer in range(min_layer, num_layers):
        polys = polygons_by_layer.get(layer)
        if not polys:
            continue
        if layer_intensities and layer in layer_intensities:
            intensity = int(layer_intensities[layer])
        else:
            intensity = yield_to_intensity(layer_yield(layer, num_layers))
        pts = [np.round(p - (ox, oy)).astype(np.int32) for p in polys]
        cv2.fillPoly(canvas, pts, color=intensity)
    return canvas


def render_reference(path: str, size: int = REF_SIZE,
                     layer_intensities: dict | None = None,
                     min_layer: int = 0) -> np.ndarray:
    """``.gds`` -> grayscale reference raster, ready for the matcher.

    This is the single entry point the Phase 3 read path needs.
    """
    polys, num_layers = read_gds_layers(path)
    return rasterize_layers(polys, num_layers, size=size,
                            layer_intensities=layer_intensities,
                            min_layer=min_layer)


# ---------------------------------------------------------------------------
# Search-resolution rasterization
# ---------------------------------------------------------------------------
#
# The search frame is 1000x1000 at 10 nm/px, so the design must also be
# renderable at that coarser scale: a 1 nm/px raster of the search FOV would be
# 10000x10000, which is both wasteful and the wrong sampling for matching.
#
# The scaling is done by shrinking the DESIGN COORDINATES before rasterizing,
# not by downsampling a fine raster afterwards. Those differ: raster-then-resize
# averages antialiased edges, whereas coordinate-scaling re-rasterizes the
# polygons at the target sampling, which is what a coarser capture actually
# does. For a 10x step the difference is visible on 1-2 px features.

PIXEL_SIZE_REF_NM = 1
PIXEL_SIZE_SEARCH_NM = 10
SCALE_FACTOR = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM


def rasterize_layers_at(polygons_by_layer: dict, num_layers: int, size: int,
                        nm_per_px: float, **kwargs) -> np.ndarray:
    """Rasterize design geometry sampled at ``nm_per_px``.

    ``nm_per_px=1`` gives the reference convention; ``nm_per_px=10`` gives the
    search convention. Coordinates are divided by ``nm_per_px`` before painting
    so polygons are re-sampled at the target pitch.
    """
    if nm_per_px <= 0:
        raise GdsError(f"nm_per_px must be positive, got {nm_per_px}")
    if nm_per_px == 1:
        return rasterize_layers(polygons_by_layer, num_layers, size=size, **kwargs)
    scaled = {L: [p / float(nm_per_px) for p in polys]
              for L, polys in polygons_by_layer.items()}
    return rasterize_layers(scaled, num_layers, size=size, **kwargs)


def render_reference_at(path: str, size: int, nm_per_px: float,
                        layer_intensities: dict | None = None,
                        min_layer: int = 0) -> np.ndarray:
    """``.gds`` -> raster at an arbitrary sampling (reference or search)."""
    polys, num_layers = read_gds_layers(path)
    return rasterize_layers_at(polys, num_layers, size, nm_per_px,
                               layer_intensities=layer_intensities,
                               min_layer=min_layer)
