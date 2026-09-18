"""Phase 3: emit real multi-layer GDSII geometry for a Drift-Sense site.

Phase 3 asks for a **reference CAD file**, not a reference image, and the deck
shows a **stacked** design: its staged view reads "layers 0 to 1 up to word
line", "0 to 3 up to bit line metal", "0 to 5 up to storage capacitor", "all 8
layers = the full GDS".

The Phase 1/2 generator already models a layer stack internally, but only four
deep and one of those four is a flat background constant
(``generator/src/patterns/dram.py`` returns ``{substrate, word_line, bit_line,
storage_contact}``). This module builds the **8-layer** stack the deck names, as
actual ``gdstk`` polygons, at the same geometry the raster generator draws --
same presets, same pitch/width fields, same position jitter and gap-collapse
model -- so a GDS generated here describes the *same* site the image generator
would rasterize.

Layer numbering (0 = most buried), chosen to match the deck's staged view
exactly at the two named cut points (1 = word line, 3 = bit line metal,
5 = storage capacitor):

    DRAM                           FinFET
    0  active / diffusion          0  fin
    1  word_line (poly gate)       1  gate (poly)
    2  bit_line_contact            2  spacer
    3  bit_line_metal              3  contact
    4  storage_contact             4  via0
    5  storage_capacitor           5  metal1
    6  via1                        6  via1
    7  metal2_strap                7  metal2

Brightness is **never** assigned here. A GDS layer is an integer and nothing
more; the render side derives grey level from the layer's position in the
stack (see ``driftsense/gds.py``). That is what makes "infer the per-layer
brightness" a meaningful task instead of a lookup.

Nothing in this module is used by ``register.py`` or by the shipped inference
path. It is generator-side: it produces the training artefacts.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# layer numbering
# ---------------------------------------------------------------------------

DRAM_LAYERS = (
    "active", "word_line", "bit_line_contact", "bit_line_metal",
    "storage_contact", "storage_capacitor", "via1", "metal2_strap",
)
FINFET_LAYERS = (
    "fin", "gate", "spacer", "contact", "via0", "metal1", "via1", "metal2",
)
NUM_LAYERS = 8

# Geometry knobs, mirroring the organizer-side CAD generator's constants so the
# two stacks have the same shape of layout. Illustrative of real topology, not
# a fab-exact process.
POSITION_JITTER_NM = 1.5
WIDTH_JITTER_FRACTION = 0.10
STRAP_PERIOD = 4            # one metal2 strap every N bit lines / fin columns
STRAP_WIDTH_FACTOR = 2.5    # strap width relative to the line width
CAPACITOR_FILL_FACTOR = 1.8  # capacitor radius relative to the storage via
VIA1_RADIUS_FACTOR = 0.6
SPACER_WIDTH_FRACTION = 0.15
ACTIVE_FILL_FRACTION = 0.88
METAL1_HEIGHT_FRACTION = 0.35
CONTACT_ASPECT_RATIO = 1.6


def _require_gdstk():
    try:
        import gdstk  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Phase 3 GDS emission needs 'gdstk' (pinned in requirements.txt); "
            f"it is not importable. Original error: {exc!r}"
        ) from exc
    return gdstk


def _positions(size_nm: float, pitch_nm: float,
               rng: np.random.Generator) -> np.ndarray:
    """Line centres at `pitch_nm` with the upstream jitter model."""
    out = []
    pos = float(rng.uniform(0, pitch_nm))
    while pos < size_nm:
        out.append(pos)
        pos += pitch_nm + float(rng.normal(0, POSITION_JITTER_NM))
    return np.asarray(out, dtype=np.float64)


def _jittered_widths(n: int, width_nm: float,
                     rng: np.random.Generator) -> np.ndarray:
    w = width_nm * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION, size=n))
    return np.clip(w, width_nm * 0.5, width_nm * 1.5)


def _tablet(cx: float, cy: float, radius: float, layer: int, aspect: float):
    """A capsule/tablet contact -- what the CAD generator uses for vias.

    Built as a rectangle with fully rounded short ends, which gdstk expresses
    as a polygon with filleted corners; approximated here by an octagon-with-
    caps outline so it needs no boolean ops.
    """
    import gdstk  # noqa: PLC0415

    half_short = max(radius / aspect, 1e-6)   # across the capsule
    half_long = radius * aspect               # along it
    # A stadium: two semicircular caps joined by straight sides.
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    xs = np.cos(angles) * half_short
    ys = np.sin(angles) * half_long
    pts = np.column_stack([cx + xs, cy + ys])
    return gdstk.Polygon(pts, layer=layer)


def build_dram_gds(size_nm: float, preset: dict, collapse_threshold_nm: float,
                   rng: np.random.Generator, contact_aspect_ratio: float = CONTACT_ASPECT_RATIO):
    """Build the 8-layer DRAM stack as a gdstk Cell."""
    gdstk = _require_gdstk()
    cell = gdstk.Cell("DRAM")

    wl_pos = _positions(size_nm, preset["word_line_pitch_nm"], rng)
    bl_pos = _positions(size_nm, preset["bit_line_pitch_nm"], rng)
    wl_w = _jittered_widths(len(wl_pos), preset["word_line_width_nm"], rng)
    bl_w = _jittered_widths(len(bl_pos), preset["bit_line_width_nm"], rng)

    # -- layer 0: active/diffusion islands in the cells between the lines.
    # Deliberately a strict interior fraction of each cell so it never touches
    # a neighbour -- growing past the gap *is* touching the line's edge.
    for i in range(len(wl_pos) - 1):
        lo_y = wl_pos[i] + wl_w[i] / 2.0
        hi_y = wl_pos[i + 1] - wl_w[i + 1] / 2.0
        if hi_y <= lo_y:
            continue
        cy = (lo_y + hi_y) / 2.0
        ry = (hi_y - lo_y) / 2.0 * ACTIVE_FILL_FRACTION
        for j in range(len(bl_pos) - 1):
            lo_x = bl_pos[j] + bl_w[j] / 2.0
            hi_x = bl_pos[j + 1] - bl_w[j + 1] / 2.0
            if hi_x <= lo_x:
                continue
            cx = (lo_x + hi_x) / 2.0
            rx = (hi_x - lo_x) / 2.0 * ACTIVE_FILL_FRACTION
            cell.add(gdstk.rectangle((cx - rx, cy - ry), (cx + rx, cy + ry), layer=0))

    # -- layer 1: word lines (horizontal)
    for i, y in enumerate(wl_pos):
        half = wl_w[i] / 2.0
        cell.add(gdstk.rectangle((0.0, y - half), (size_nm, y + half), layer=1))
        if i + 1 < len(wl_pos):
            nxt = wl_pos[i + 1] - wl_w[i + 1] / 2.0
            gap = nxt - (y + half)
            if gap <= collapse_threshold_nm:
                cell.add(gdstk.rectangle((0.0, y + half), (size_nm, nxt), layer=1))

    # -- layer 3: bit lines (vertical)
    for j, x in enumerate(bl_pos):
        half = bl_w[j] / 2.0
        cell.add(gdstk.rectangle((x - half, 0.0), (x + half, size_nm), layer=3))
        if j + 1 < len(bl_pos):
            nxt = bl_pos[j + 1] - bl_w[j + 1] / 2.0
            gap = nxt - (x + half)
            if gap <= collapse_threshold_nm:
                cell.add(gdstk.rectangle((x + half, 0.0), (nxt, size_nm), layer=3))

    # -- layers 2, 4, 5: checkerboard vias + the capacitor above each storage via
    base_r = preset["contact_diameter_nm"] / 2.0
    for i, wy in enumerate(wl_pos):
        for j, bx in enumerate(bl_pos):
            r = max(base_r * (1.0 + float(rng.normal(0, WIDTH_JITTER_FRACTION))), 1.0)
            storage = (i + j) % 2 == 0
            layer = 4 if storage else 2
            cell.add(_tablet(bx, wy, r, layer, contact_aspect_ratio))
            if storage:
                cell.add(_tablet(bx, wy, r * CAPACITOR_FILL_FACTOR, 5, contact_aspect_ratio))

    # -- layers 6, 7: via1 + the metal2 strap over a sparse subset of bit lines
    strap_half = (preset["bit_line_width_nm"] * STRAP_WIDTH_FACTOR) / 2.0
    via1_r = base_r * VIA1_RADIUS_FACTOR
    for j, bx in enumerate(bl_pos):
        if j % STRAP_PERIOD != 0:
            continue
        cell.add(gdstk.rectangle((bx - strap_half, 0.0), (bx + strap_half, size_nm), layer=7))
        for wy in wl_pos:
            cell.add(_tablet(bx, wy, via1_r, 6, contact_aspect_ratio))

    return cell


def build_finfet_gds(size_nm: float, preset: dict, collapse_threshold_nm: float,
                     rng: np.random.Generator, contact_aspect_ratio: float = CONTACT_ASPECT_RATIO):
    """Build the 8-layer FinFET stack as a gdstk Cell."""
    gdstk = _require_gdstk()
    cell = gdstk.Cell("FINFET")

    fin_pos = _positions(size_nm, preset["fin_pitch_nm"], rng)
    gate_pos = _positions(size_nm, preset["gate_pitch_nm"], rng)
    fin_w = _jittered_widths(len(fin_pos), preset["fin_width_nm"], rng)
    gate_w = _jittered_widths(len(gate_pos), preset["gate_length_nm"], rng)

    # -- layer 0: fins (vertical)
    for j, x in enumerate(fin_pos):
        half = fin_w[j] / 2.0
        cell.add(gdstk.rectangle((x - half, 0.0), (x + half, size_nm), layer=0))

    # -- layer 1: gates (horizontal), layer 2: spacers flanking each gate
    spacer_w = preset["gate_length_nm"] * SPACER_WIDTH_FRACTION
    for i, y in enumerate(gate_pos):
        half = gate_w[i] / 2.0
        cell.add(gdstk.rectangle((0.0, y - half), (size_nm, y + half), layer=1))
        if i + 1 < len(gate_pos):
            nxt = gate_pos[i + 1] - gate_w[i + 1] / 2.0
            gap = nxt - (y + half)
            if gap <= collapse_threshold_nm:
                cell.add(gdstk.rectangle((0.0, y + half), (size_nm, nxt), layer=1))
        cell.add(gdstk.rectangle((0.0, y - half - spacer_w), (size_nm, y - half), layer=2))
        cell.add(gdstk.rectangle((0.0, y + half), (size_nm, y + half + spacer_w), layer=2))

    # -- layer 3: source/drain contacts in the gaps between gates
    half_c = max(preset["contact_size_nm"] / 2.0, 1.0)
    for i, fx in enumerate(fin_pos):
        for j in range(len(gate_pos) - 1):
            if (i + j) % 2 != 0:
                continue
            mid_y = (gate_pos[j] + gate_pos[j + 1]) / 2.0
            cell.add(gdstk.rectangle((fx - half_c, mid_y - half_c),
                                     (fx + half_c, mid_y + half_c), layer=3))

    # -- layers 4, 5: via0 over each contact, then an M1 band
    m1_half = half_c * METAL1_HEIGHT_FRACTION
    for i, fx in enumerate(fin_pos):
        for j in range(len(gate_pos) - 1):
            if (i + j) % 2 != 0:
                continue
            mid_y = (gate_pos[j] + gate_pos[j + 1]) / 2.0
            cell.add(_tablet(fx, mid_y, half_c * 0.5, 4, contact_aspect_ratio))
            cell.add(gdstk.rectangle((fx - half_c, mid_y - m1_half),
                                     (fx + half_c, mid_y + m1_half), layer=5))

    # -- layers 6, 7: via1 + M2, routed orthogonally to M1
    m2_half = (preset["fin_width_nm"] * STRAP_WIDTH_FACTOR) / 2.0
    for i, fx in enumerate(fin_pos):
        if i % STRAP_PERIOD != 0:
            continue
        cell.add(gdstk.rectangle((fx - m2_half, 0.0), (fx + m2_half, size_nm), layer=7))
        for gy in gate_pos:
            cell.add(_tablet(fx, gy, half_c * 0.4, 6, contact_aspect_ratio))

    return cell


STRIP_LAYER_INDEX = 1   # routing lives on the same level as the word line/gate
STRIP_ROUTING_PITCH_NM = 220.0
STRIP_ROUTING_WIDTH_NM = 9.0


def clip_cell_to_rect(cell, w: float, h: float, num_layers: int = NUM_LAYERS):
    """Clip a cell to the rectangle ``(0,0)-(w,h)``.

    A mat cell is built over a square field of ``max(w, h)`` so its pattern
    pitch is laid out correctly, but only the ``(w, h)`` window of that field is
    actually the mat. Leaving the polygons that fall outside means the design
    contains geometry the raster never paints -- and since a reference is
    clipped from the design, a window near the mat edge picks up polygons the
    search image does not have. Measured as 14% of pixels differing at the true
    match before this clip.
    """
    gdstk = _require_gdstk()
    rect = gdstk.rectangle((0.0, 0.0), (float(w), float(h)))
    out = gdstk.Cell(f"CLIP_{cell.name}")
    for layer in range(num_layers):
        polys = cell.get_polygons(layer=layer, datatype=0)
        if not polys:
            continue
        for c in gdstk.boolean(polys, [rect], "and", layer=layer, datatype=0):
            out.add(c)
    return out


def build_strip_cell(x0: float, y0: float, w: float, h: float,
                     routing_pitch_nm: float = STRIP_ROUTING_PITCH_NM,
                     routing_width_nm: float = STRIP_ROUTING_WIDTH_NM,
                     rng: np.random.Generator | None = None,
                     layer: int = STRIP_LAYER_INDEX):
    """A separator-strip region as real design geometry, in canvas coords.

    The strips between mats carry sparse peripheral routing. This has to be
    **design geometry on a GDS layer**, not paint applied to the search raster
    only, for one reason: the reference is clipped from the design, so anything
    the search image shows that the design does not contain makes the two sides
    describe different scenes. Where a window straddles a mat/strip boundary,
    a raster-only strip texture put content in the search that the reference
    could never have (measured: 17.8% of pixels differing at the true match).

    Emitting it as polygons keeps one geometry source for both sides. The cell
    is built in CANVAS coordinates because the strip is a canvas-level region,
    not a mat-local one.
    """
    gdstk = _require_gdstk()
    cell = gdstk.Cell(f"STRIP_{int(x0)}_{int(y0)}")
    rng = rng if rng is not None else np.random.default_rng(0)
    half = routing_width_nm / 2.0

    # One direction per strip: horizontal strips get vertical routing lines and
    # vice versa, so the routing reads as peripheral wiring rather than as a
    # second device array.
    horizontal = w >= h
    if horizontal:
        pos = float(x0) + float(rng.uniform(0, routing_pitch_nm))
        while pos < x0 + w:
            cell.add(gdstk.rectangle((pos - half, y0), (pos + half, y0 + h),
                                     layer=layer))
            pos += routing_pitch_nm
    else:
        pos = float(y0) + float(rng.uniform(0, routing_pitch_nm))
        while pos < y0 + h:
            cell.add(gdstk.rectangle((x0, pos - half), (x0 + w, pos + half),
                                     layer=layer))
            pos += routing_pitch_nm
    return cell


def build_gds(architecture: str, size_nm: float, preset: dict,
              collapse_threshold_nm: float, rng: np.random.Generator,
              contact_aspect_ratio: float = CONTACT_ASPECT_RATIO):
    """Dispatch on architecture. `architecture` is 'dram' or 'finfet'."""
    kind = architecture.lower()
    if kind == "dram":
        return build_dram_gds(size_nm, preset, collapse_threshold_nm, rng,
                              contact_aspect_ratio)
    if kind == "finfet":
        return build_finfet_gds(size_nm, preset, collapse_threshold_nm, rng,
                                contact_aspect_ratio)
    raise ValueError(f"unknown architecture {architecture!r}; expected dram or finfet")


def clip_to_window(cell, x0: float, y0: float, size_nm: float, num_layers: int = NUM_LAYERS):
    """Clip a single cell to a window and translate it to the origin.

    Only correct when the whole window lies inside that one cell. Use
    ``clip_multi_cell`` when the window may straddle a mat boundary.
    """
    gdstk = _require_gdstk()
    window = gdstk.rectangle((x0, y0), (x0 + size_nm, y0 + size_nm))
    out = gdstk.Cell("REF")
    for layer in range(num_layers):
        polys = cell.get_polygons(layer=layer, datatype=0)
        if not polys:
            continue
        for poly in polys:
            clipped = gdstk.boolean([poly], [window], "and", layer=layer,
                                    datatype=0)
            for c in clipped:
                c.translate(-x0, -y0)
                out.add(c)
    return out


def clip_multi_cell(mats, x0: float, y0: float, size_nm: float,
                    num_layers: int = NUM_LAYERS):
    """Clip the *canvas* to a window, merging every mat that overlaps it.

    This is what the reference GDS must be built with. Each mat's cell holds
    polygons in its own local ``(0,0)-(w,h)`` frame, so they are shifted into
    canvas coordinates, clipped, then shifted back to be relative to
    ``(x0, y0)``.

    Clipping only the mat that *contains* the window origin is wrong whenever
    the window straddles a boundary: the neighbouring mat's geometry is present
    in the rendered search canvas but absent from the reference, so the two
    sides stop describing the same scene (measured as NCC ~0.42 at the true
    location instead of ~0.9). A window over a pure strip legitimately
    contributes no polygons for that area -- a real design has empty space too.
    """
    gdstk = _require_gdstk()
    window = gdstk.rectangle((x0, y0), (x0 + size_nm, y0 + size_nm))
    out = gdstk.Cell("REF")
    x1, y1 = x0 + size_nm, y0 + size_nm
    for mat in mats:
        mx0, my0, mw, mh = mat["x0"], mat["y0"], mat["w"], mat["h"]
        # bounding-box reject: no overlap means nothing to contribute
        if mx0 + mw <= x0 or mx0 >= x1 or my0 + mh <= y0 or my0 >= y1:
            continue
        cell = mat["cell"]
        for layer in range(num_layers):
            polys = cell.get_polygons(layer=layer, datatype=0)
            if not polys:
                continue
            moved = []
            for poly in polys:
                p = poly.copy()
                p.translate(mx0, my0)
                moved.append(p)
            for c in gdstk.boolean(moved, [window], "and",
                                   layer=layer, datatype=0):
                c.translate(-x0, -y0)
                out.add(c)
    return out


def layer_population(cell, num_layers: int = NUM_LAYERS) -> dict:
    """Polygon count per layer -- used by tests and the manifest."""
    counts = {}
    for layer in range(num_layers):
        counts[layer] = len(cell.get_polygons(layer=layer, datatype=0))
    return counts
