"""
Drift-Sense synthetic data explorer -- Streamlit app for Hugging Face Spaces.

Design-to-SEM registration: the Reference is a real multi-layer GDSII CAD
file (a design database, never imaged), and the Search is the noisy SEM
capture of the same scene at a coarser pixel size. This is a thin UI shell
around src/cad_pipeline.py -- every image/GDS shown here comes from the
exact same generator students use from the CLI (generate_cad_dataset.py),
nothing here is a separate/approximated reimplementation.

Two tabs: "Generate" is the interactive single-sample explorer; "Bookmarks"
holds samples saved from Generate (full parameter snapshots, re-rendered on
demand) with a bulk .zip export for organizer review.
"""

import json
import os
import tempfile
import time
import zipfile
from io import BytesIO

import cv2
import gdstk
import numpy as np
import streamlit as st

from src.cad_pipeline import (
    CadGenerationParams, build_cad_geometry, render_cad_sample, FINE_CANVAS_SIZE_PX, MAX_SEARCH_ROTATION_DEG,
)
from src.cad.render import rasterize_cell_layers_colored, rasterize_full_canvas_colored
from src.cad import yield_model
from baseline_solution.zncc import zncc_match, shift_report
from curated_bookmarks import generate_curated_bookmarks, load_curated_bookmarks_from_disk

st.set_page_config(page_title="Drift-Sense Synthetic Data Explorer", layout="wide")
st.title("Drift-Sense: Synthetic Dataset Explorer")
st.caption(
    "Reference: a real multi-layer GDSII CAD file (1000x1000 nm design window). "
    "Search: 1000x1000 px @ 10 nm/px SEM capture (10 um FOV). A design-to-SEM registration "
    "problem, not image-to-image -- find where the Reference CAD's footprint sits in the "
    "noisy Search image. Since a CAD file is a design database, not an imaged capture, only "
    "the Search image ever goes through SEM acquisition noise."
)

CAD_LAYER_NAMES = {
    "dram": [
        "active", "word_line", "bit_line_contact", "bit_line_metal", "storage_contact",
        "storage_capacitor", "via1", "metal2_strap",
    ],
    "finfet": ["fin", "gate", "spacer", "contact", "via0", "metal1", "via1", "metal2"],
}

if "bookmarks" not in st.session_state:
    # Fresh session -- load the persisted curated set from disk if one
    # exists, so a restart doesn't come back empty. "Generate all" (in the
    # Bookmarks tab) regenerates and re-persists this file on demand.
    st.session_state.bookmarks = load_curated_bookmarks_from_disk() or []
tab_generate, tab_bookmarks = st.tabs(["Generate", f"Bookmarks ({len(st.session_state.bookmarks)})"])


GT_COLOR_BGR = (0, 255, 0)      # green -- the ideal/true reference location
PRED_COLOR_BGR = (0, 0, 255)    # red -- a matcher's found location


def _draw_crosshair(img, cx, cy, color, size=18, thickness=2):
    """Draw a "+" crosshair centered at (cx, cy) -- used instead of a
    bounding box so overlapping GT/found markers stay readable and the
    marker reads as "this exact point", not "this exact box" (gt_box is an
    axis-aligned approximation anyway once rotation is involved)."""
    cx, cy = int(round(cx)), int(round(cy))
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)


@st.cache_resource(show_spinner="Building CAD geometry...")
def _cached_cad_geometry(
    kind, seed, mat_size, strip_width, collapse_threshold, psp, psr, nmp, lwb, corner_r, manual_center=None,
    contact_aspect_ratio=1.6, contact_thickness_factor=1.0, contact_angle_deg=90.0, center_bias=False,
):
    geom_params = CadGenerationParams(
        collapse_threshold_nm=collapse_threshold,
        mat_size_nm=mat_size, strip_width_nm=strip_width,
        polygon_scale_prob=psp, polygon_scale_range=psr, no_match_prob=nmp,
        linewidth_bias_nm=lwb, corner_rounding_px=corner_r,
        contact_aspect_ratio=contact_aspect_ratio,
        contact_thickness_factor=contact_thickness_factor, contact_angle_deg=contact_angle_deg,
    )
    rng = np.random.default_rng(seed)
    return build_cad_geometry(kind, rng, geom_params, manual_center_nm=manual_center, center_bias=center_bias)


@st.cache_resource(show_spinner="Rendering full-canvas CAD viewer...")
def _cached_full_canvas_colored(
    kind, seed, mat_size, strip_width, collapse_threshold, psp, psr, nmp, lwb, corner_r, manual_center=None,
    contact_aspect_ratio=1.6, contact_thickness_factor=1.0, contact_angle_deg=90.0, min_layer=0, display_px=1000,
):
    geom = _cached_cad_geometry(
        kind, seed, mat_size, strip_width, collapse_threshold, psp, psr, nmp, lwb, corner_r, manual_center,
        contact_aspect_ratio, contact_thickness_factor, contact_angle_deg,
    )
    full = rasterize_full_canvas_colored(
        geom["mats"], geom["strip_rects"], FINE_CANVAS_SIZE_PX, geom["num_layers"], min_layer=min_layer,
    )
    return cv2.resize(full, (display_px, display_px), interpolation=cv2.INTER_AREA)


def _gds_bytes(cell: gdstk.Cell) -> bytes:
    lib = gdstk.Library()
    lib.add(cell)
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as tmp:
        lib.write_gds(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data


def _full_canvas_cell(mats: list, num_layers: int) -> gdstk.Cell:
    """Merge every mat's real (as-designed) polygons into one Cell at their
    canvas coordinates -- the entire Search-side CAD as a single exportable
    .gds, the vector counterpart of rasterize_full_canvas_colored (minus the
    strip routing texture, which isn't real device geometry).

    Uses `design_cell`, not `fab_cell` -- a GDS is a design database, so
    even "the whole die's CAD" is the same ideal, undistorted geometry as
    the Reference crop; fabrication/imaging distortion only ever shows up
    in the rendered SEM images, never in an exported design file.
    """
    out = gdstk.Cell("SEARCH_SIDE_CAD")
    for mat in mats:
        mx0, my0 = mat["x0"], mat["y0"]
        for layer in range(num_layers):
            for poly in mat["design_cell"].get_polygons(layer=layer, datatype=0):
                poly.translate(mx0, my0)
                out.add(poly)
    return out


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _get_intensities(prefix: str, num_layers: int) -> dict:
    """Read per-layer intensity from session_state, falling back to the
    yield-model default for any layer whose slider hasn't been touched yet
    (e.g. the target dropdown hasn't been switched to it in this session).
    """
    out = {}
    for i in range(num_layers):
        default = yield_model.yield_to_intensity(yield_model.layer_yield(i, num_layers))
        out[i] = st.session_state.get(f"{prefix}_intensity_{i}", default)
    return out


def _randomize():
    """Re-roll everything a generated sample varies on: a fresh seed (new
    mat/strip layout, new crop location, new match/no-match draw), new SEM
    acquisition settings, and new augmentation parameters. Architecture kind
    and array/strip sizing are left alone -- those are structural choices,
    not something a single "give me a random sample" click should disturb.

    Every knob is drawn from a *moderate* sub-band of its slider's full
    range, not the full range itself -- sliders go all the way to their
    physical extremes (e.g. dose down to 20, detector noise up to 20) so a
    student can deliberately construct a torture-test sample, but several
    independent knobs each landing near their extreme at once (low dose +
    high detector noise + large beam spot + large drift, say) compounds
    into a sample that's unusably noisy end to end. Randomize should hand
    back a *plausible* draw every time, not occasionally an unreadable one.
    """
    rng = np.random.default_rng()
    st.session_state.cad_seed = int(rng.integers(0, 2_000_000_000))
    st.session_state.cad_dose = float(rng.uniform(120.0, 900.0))
    st.session_state.cad_beam_spot = float(rng.uniform(2.0, 9.0))
    st.session_state.cad_shear = float(rng.uniform(0.0, 2.5))
    st.session_state.cad_drift = float(rng.uniform(0.0, 1.2))
    st.session_state.cad_detector_noise = float(rng.uniform(0.0, 7.0))
    st.session_state.cad_astigmatism = float(rng.uniform(0.8, 1.6))
    st.session_state.cad_rotation = float(rng.uniform(0.0, 8.0))
    st.session_state.cad_barrel = float(rng.uniform(-0.05, 0.05))
    st.session_state.cad_vignette = float(rng.uniform(0.0, 0.3))
    st.session_state.cad_gamma = float(rng.uniform(0.7, 1.5))
    st.session_state.cad_charging_prob = float(rng.uniform(0.0, 1.5))
    st.session_state.cad_charging_intensity = float(rng.uniform(0.0, 1.0))
    st.session_state.cad_speckle = float(rng.uniform(0.0, 0.3))
    st.session_state.cad_salt_pepper = float(rng.uniform(0.0, 0.015))
    st.session_state.cad_psp = float(rng.uniform(0.0, 0.4))
    st.session_state.cad_psr = float(rng.uniform(0.0, 0.25))
    st.session_state.cad_lwb = float(rng.uniform(-8.0, 8.0))
    st.session_state.cad_corner_r = float(rng.uniform(0.0, 8.0))


if "cad_seed" not in st.session_state:
    st.session_state.cad_seed = 7


def _rslider(label: str, lo: float, hi: float, default: float, step: float, key: str, **kwargs) -> float:
    """A slider whose key can also be written by the Randomize callback.
    Passing an explicit `value=` *and* having a callback set session_state
    for the same key triggers a Streamlit warning ("created with a default
    value but also had its value set via the Session State API") -- so for
    any randomizable slider, seed session_state once instead and never pass
    `value` to st.slider at all.
    """
    st.session_state.setdefault(key, default)
    return st.slider(label, lo, hi, step=step, key=key, **kwargs)


def _multi_slider(
    label: str, lo: float, hi: float, default: float, step: float, name: str, active_prefixes: list, **kwargs,
) -> float:
    """One on-screen slider that fans its value out to every currently
    checked target's own key (e.g. `cad_dose` and/or `ref_dose`) -- lets
    checking both Apply-to boxes drive identical settings for Search and
    Reference from a single control, while each side's own key stays the
    actual source of truth read everywhere else (rendering, bookmarks,
    JSON, randomize). Seeds the widget from the first active target's
    current value so switching which boxes are checked doesn't silently
    reset anything.
    """
    widget_key = f"shared_{name}"
    if widget_key not in st.session_state:
        seed = default
        for p in active_prefixes:
            if f"{p}_{name}" in st.session_state:
                seed = st.session_state[f"{p}_{name}"]
                break
        st.session_state[widget_key] = seed
    value = st.slider(label, lo, hi, step=step, key=widget_key, **kwargs)
    for p in active_prefixes:
        st.session_state[f"{p}_{name}"] = value
    return value


def _render_operation_expanders(active_prefixes: list) -> None:
    """Fabrication-distortion / SEM-acquisition / motion / lens / charging
    controls. `active_prefixes` is the subset of ["cad", "ref"] currently
    checked in "Apply to" -- every slider here writes into all of them at
    once via `_multi_slider`, so checking both fans identical settings out
    to Search's and Reference's own independent stores.
    """
    with st.expander("Fabrication distortion (design -> real device)", expanded=True):
        st.caption("Applied when turning the design into a real, imaged device.")
        _multi_slider("Polygon-scale outlier probability", 0.0, 1.0, 0.10, 0.05, "psp", active_prefixes)
        _multi_slider("Polygon-scale outlier range (+/- fraction)", 0.0, 0.5, 0.10, 0.05, "psr", active_prefixes)
        _multi_slider("Linewidth/CD bias (nm)", -20.0, 20.0, 0.0, 1.0, "lwb", active_prefixes)
        _multi_slider("Corner rounding radius (nm)", 0.0, 20.0, 0.0, 1.0, "corner_r", active_prefixes)

    with st.expander("SEM acquisition noise", expanded=True):
        _multi_slider("Dose (higher = cleaner)", 20.0, 2000.0, 200.0, 20.0, "dose", active_prefixes)
        _multi_slider("Beam spot size (nm)", 1.0, 20.0, 5.0, 0.5, "beam_spot", active_prefixes)
        _multi_slider("Detector noise sigma", 0.0, 20.0, 5.0, 0.5, "detector_noise", active_prefixes)

    with st.expander("Motion, drift & rotation", expanded=True):
        _multi_slider("Raster drift/shear (px)", 0.0, 5.0, 1.5, 0.1, "shear", active_prefixes)
        _multi_slider("Row jitter (px)", 0.0, 3.0, 0.5, 0.1, "drift", active_prefixes)
        _multi_slider(
            "Stage/wafer rotation (+/- degrees)", 0.0, MAX_SEARCH_ROTATION_DEG, 0.0, 1.0, "rotation", active_prefixes,
            help="Whole-image rotation simulating the die being placed at a slight angle on the stage.",
        )

    with st.expander("Lens & environmental distortion"):
        _multi_slider(
            "Beam astigmatism ratio", 0.5, 2.5, 1.0, 0.05, "astigmatism", active_prefixes,
            help="1.0 = round beam spot. >1.0 stretches it into an ellipse (sigmaY = sigmaX * ratio), "
                 "modeling a real, common SEM aberration -- directional blur instead of uniform blur.",
        )
        _multi_slider(
            "Barrel(+)/pincushion(-) distortion", -0.15, 0.15, 0.0, 0.01, "barrel", active_prefixes,
            help="Radial lens-style distortion from imperfect beam-scan linearity.",
        )
        _multi_slider("Vignette strength", 0.0, 1.0, 0.0, 0.05, "vignette", active_prefixes)
        _multi_slider("Gamma (contrast curve)", 0.4, 2.5, 1.0, 0.05, "gamma", active_prefixes)

    with st.expander("Charging & impulse noise"):
        _multi_slider(
            "Charging streaks (per 100 rows)", 0.0, 5.0, 0.0, 0.25, "charging_prob", active_prefixes,
            help="Bright horizontal streaks from local sample charging on insulating regions.",
        )
        _multi_slider("Charging streak intensity", 0.0, 3.0, 0.0, 0.1, "charging_intensity", active_prefixes)
        _multi_slider(
            "Speckle noise sigma (multiplicative)", 0.0, 1.0, 0.0, 0.05, "speckle", active_prefixes,
            help="out = img * (1 + N(0, sigma)) -- noise magnitude scales with brightness, unlike "
                 "additive detector noise.",
        )
        _multi_slider(
            "Salt-and-pepper probability", 0.0, 0.05, 0.0, 0.005, "salt_pepper", active_prefixes,
            help="Fraction of pixels forced to pure black/white -- dead/hot pixels or discharge events.",
        )


_NO_AUGMENTATION_BASELINE = {
    "psp": 0.0, "psr": 0.0, "lwb": 0.0, "corner_r": 0.0,
    "dose": 2000.0, "beam_spot": 1.0, "detector_noise": 0.0,
    "shear": 0.0, "drift": 0.0, "rotation": 0.0,
    "astigmatism": 1.0, "barrel": 0.0, "vignette": 0.0, "gamma": 1.0,
    "charging_prob": 0.0, "charging_intensity": 0.0, "speckle": 0.0, "salt_pepper": 0.0,
}


def _reset_operations():
    """Reset every augmentation/noise/distortion slider in the Operations
    panel -- both Search's and Reference's stored values, plus whichever
    shared on-screen widgets are currently displayed -- back to a 'nearly
    no augmentation' baseline (clean dose, round beam, zero noise/drift/
    rotation/distortion). Layer intensities and CAD-structure fields are
    untouched; this only resets the Operations panel's sliders.
    """
    for name, value in _NO_AUGMENTATION_BASELINE.items():
        for key in (f"cad_{name}", f"ref_{name}", f"shared_{name}"):
            if key in st.session_state:
                st.session_state[key] = value


def _read_operation_params(op_prefix: str, **overrides) -> CadGenerationParams:
    """Build a CadGenerationParams from whatever is in session_state under
    `op_prefix` (falling back to each slider's own default), so Search's
    and Reference's operation panels can share one read path even though
    only one is ever rendered in the sidebar at a time. `overrides` sets
    fields that aren't part of the operations sliders (e.g. mat_size_nm).
    """
    g = st.session_state.get
    params = CadGenerationParams(
        polygon_scale_prob=g(f"{op_prefix}_psp", 0.10),
        polygon_scale_range=g(f"{op_prefix}_psr", 0.10),
        linewidth_bias_nm=g(f"{op_prefix}_lwb", 0.0),
        corner_rounding_px=g(f"{op_prefix}_corner_r", 0.0),
        dose_search=g(f"{op_prefix}_dose", 200.0),
        beam_spot_size_nm=g(f"{op_prefix}_beam_spot", 5.0),
        shear_amplitude_px=g(f"{op_prefix}_shear", 1.5),
        drift_jitter_px=g(f"{op_prefix}_drift", 0.5),
        detector_noise_sigma_search=g(f"{op_prefix}_detector_noise", 5.0),
        astigmatism_ratio=g(f"{op_prefix}_astigmatism", 1.0),
        search_rotation_deg=g(f"{op_prefix}_rotation", 0.0),
        barrel_distortion_k=g(f"{op_prefix}_barrel", 0.0),
        vignette_strength=g(f"{op_prefix}_vignette", 0.0),
        gamma=g(f"{op_prefix}_gamma", 1.0),
        charging_streak_prob=g(f"{op_prefix}_charging_prob", 0.0),
        charging_streak_intensity=g(f"{op_prefix}_charging_intensity", 0.0),
        speckle_sigma=g(f"{op_prefix}_speckle", 0.0),
        salt_pepper_prob=g(f"{op_prefix}_salt_pepper", 0.0),
    )
    for field_name, value in overrides.items():
        setattr(params, field_name, value)
    return params


def _current_bookmark_snapshot() -> dict:
    """Build a bookmark from whatever is currently in session_state --
    reading state directly (not local script variables) so this works
    identically whether called from a button callback (pre-rerun) or from
    the main body.
    """
    kind = st.session_state.get("cad_kind", "dram")
    num_layers = len(CAD_LAYER_NAMES.get(kind, []))
    default_intensity = {
        i: yield_model.yield_to_intensity(yield_model.layer_yield(i, num_layers)) for i in range(num_layers)
    }
    is_manual = st.session_state.get("crop_mode") == "Manual (pick center)"
    return {
        "id": f"{int(time.time() * 1000)}_{st.session_state.get('cad_seed')}",
        "kind": kind,
        "seed": st.session_state.get("cad_seed"),
        "manual_center": [
            st.session_state.get("manual_cx", FINE_CANVAS_SIZE_PX / 2.0),
            st.session_state.get("manual_cy", FINE_CANVAS_SIZE_PX / 2.0),
        ] if is_manual else None,
        "structure": {
            "mat_size_nm": st.session_state.get("cad_mat_size", 2600.0),
            "strip_width_nm": st.session_state.get("cad_strip_width", 320.0),
            "collapse_threshold_nm": st.session_state.get("cad_collapse", 10.0),
            "no_match_prob": st.session_state.get("cad_nmp", 0.08),
            "contact_aspect_ratio": st.session_state.get("cad_contact_aspect", 1.6),
            "contact_thickness_factor": st.session_state.get("cad_contact_thickness", 1.0),
            "contact_angle_deg": st.session_state.get("cad_contact_angle", 90),
        },
        "layer_visibility": {
            "search_min_layer": st.session_state.get("search_min_layer", 0),
            "reference_min_layer": st.session_state.get("ref_min_layer", 0),
        },
        "fab_distortion": {
            "polygon_scale_prob": st.session_state.get("cad_psp", 0.10),
            "polygon_scale_range": st.session_state.get("cad_psr", 0.10),
            "linewidth_bias_nm": st.session_state.get("cad_lwb", 0.0),
            "corner_rounding_px": st.session_state.get("cad_corner_r", 0.0),
        },
        "sem_acquisition": {
            "dose_search": st.session_state.get("cad_dose", 200.0),
            "beam_spot_size_nm": st.session_state.get("cad_beam_spot", 5.0),
            "shear_amplitude_px": st.session_state.get("cad_shear", 1.5),
            "drift_jitter_px": st.session_state.get("cad_drift", 0.5),
            "detector_noise_sigma_search": st.session_state.get("cad_detector_noise", 5.0),
            "astigmatism_ratio": st.session_state.get("cad_astigmatism", 1.0),
            "search_rotation_deg": st.session_state.get("cad_rotation", 0.0),
            "barrel_distortion_k": st.session_state.get("cad_barrel", 0.0),
            "vignette_strength": st.session_state.get("cad_vignette", 0.0),
            "gamma": st.session_state.get("cad_gamma", 1.0),
            "charging_streak_prob": st.session_state.get("cad_charging_prob", 0.0),
            "charging_streak_intensity": st.session_state.get("cad_charging_intensity", 0.0),
            "speckle_sigma": st.session_state.get("cad_speckle", 0.0),
            "salt_pepper_prob": st.session_state.get("cad_salt_pepper", 0.0),
        },
        "reference_layer_intensities": {
            i: st.session_state.get(f"ref_intensity_{i}", default_intensity[i]) for i in range(num_layers)
        },
        "search_layer_intensities": {
            i: st.session_state.get(f"search_intensity_{i}", default_intensity[i]) for i in range(num_layers)
        },
        "reference_ops_enabled": st.session_state.get("apply_ref", False),
        "reference_fab_distortion": {
            "polygon_scale_prob": st.session_state.get("ref_psp", 0.10),
            "polygon_scale_range": st.session_state.get("ref_psr", 0.10),
            "linewidth_bias_nm": st.session_state.get("ref_lwb", 0.0),
            "corner_rounding_px": st.session_state.get("ref_corner_r", 0.0),
        },
        "reference_sem_acquisition": {
            "dose_search": st.session_state.get("ref_dose", 200.0),
            "beam_spot_size_nm": st.session_state.get("ref_beam_spot", 5.0),
            "shear_amplitude_px": st.session_state.get("ref_shear", 1.5),
            "drift_jitter_px": st.session_state.get("ref_drift", 0.5),
            "detector_noise_sigma_search": st.session_state.get("ref_detector_noise", 5.0),
            "astigmatism_ratio": st.session_state.get("ref_astigmatism", 1.0),
            "search_rotation_deg": st.session_state.get("ref_rotation", 0.0),
            "barrel_distortion_k": st.session_state.get("ref_barrel", 0.0),
            "vignette_strength": st.session_state.get("ref_vignette", 0.0),
            "gamma": st.session_state.get("ref_gamma", 1.0),
            "charging_streak_prob": st.session_state.get("ref_charging_prob", 0.0),
            "charging_streak_intensity": st.session_state.get("ref_charging_intensity", 0.0),
            "speckle_sigma": st.session_state.get("ref_speckle", 0.0),
            "salt_pepper_prob": st.session_state.get("ref_salt_pepper", 0.0),
        },
    }


def _bookmark_current():
    st.session_state.bookmarks.append(_current_bookmark_snapshot())
    st.session_state["_just_bookmarked"] = True


def _sample_from_bookmark(bm: dict) -> tuple:
    """Rebuild geometry + render a sample from a bookmark's parameter
    snapshot, reusing the same cached geometry builder the Generate tab
    uses -- a bookmark stores parameters, not pixels, so it always
    regenerates the exact same sample (same seed, same everything).
    """
    manual_center = tuple(bm["manual_center"]) if bm.get("manual_center") else None
    contact_aspect_ratio = bm["structure"].get("contact_aspect_ratio", 1.6)
    contact_thickness_factor = bm["structure"].get("contact_thickness_factor", 1.0)
    contact_angle_deg = bm["structure"].get("contact_angle_deg", 90.0)
    center_bias = bm.get("center_bias", False)
    geom = _cached_cad_geometry(
        bm["kind"], bm["seed"], bm["structure"]["mat_size_nm"], bm["structure"]["strip_width_nm"],
        bm["structure"]["collapse_threshold_nm"], bm["fab_distortion"]["polygon_scale_prob"],
        bm["fab_distortion"]["polygon_scale_range"], bm["structure"]["no_match_prob"],
        bm["fab_distortion"]["linewidth_bias_nm"], bm["fab_distortion"]["corner_rounding_px"],
        manual_center, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, center_bias,
    )
    render_params = CadGenerationParams(
        mat_size_nm=bm["structure"]["mat_size_nm"], strip_width_nm=bm["structure"]["strip_width_nm"],
        collapse_threshold_nm=bm["structure"]["collapse_threshold_nm"],
        no_match_prob=bm["structure"]["no_match_prob"],
        contact_aspect_ratio=contact_aspect_ratio,
        contact_thickness_factor=contact_thickness_factor, contact_angle_deg=contact_angle_deg,
        polygon_scale_prob=bm["fab_distortion"]["polygon_scale_prob"],
        polygon_scale_range=bm["fab_distortion"]["polygon_scale_range"],
        linewidth_bias_nm=bm["fab_distortion"]["linewidth_bias_nm"],
        corner_rounding_px=bm["fab_distortion"]["corner_rounding_px"],
        **bm["sem_acquisition"],
    )
    reference_render_params = None
    if bm.get("reference_ops_enabled"):
        reference_render_params = CadGenerationParams(
            polygon_scale_prob=bm["reference_fab_distortion"]["polygon_scale_prob"],
            polygon_scale_range=bm["reference_fab_distortion"]["polygon_scale_range"],
            linewidth_bias_nm=bm["reference_fab_distortion"]["linewidth_bias_nm"],
            corner_rounding_px=bm["reference_fab_distortion"]["corner_rounding_px"],
            **bm["reference_sem_acquisition"],
        )
    layer_visibility = bm.get("layer_visibility", {})
    sample = render_cad_sample(
        geom, render_params,
        layer_intensities={int(k): v for k, v in bm["search_layer_intensities"].items()},
        reference_layer_intensities={int(k): v for k, v in bm["reference_layer_intensities"].items()},
        reference_render_params=reference_render_params,
        search_min_layer=layer_visibility.get("search_min_layer", 0),
        reference_min_layer=layer_visibility.get("reference_min_layer", 0),
    )
    return geom, sample


def _bookmark_metadata(bm: dict, geom: dict, sample: dict) -> dict:
    return {
        "id": bm["id"],
        "architecture_kind": bm["kind"],
        "num_layers": geom["num_layers"],
        "seed": bm["seed"],
        "manual_center": bm.get("manual_center"),
        "match_found": sample["match_found"],
        "gt_x": sample["gt_x"], "gt_y": sample["gt_y"], "gt_box": sample["gt_box"],
        "structure": bm["structure"],
        "layer_visibility": bm.get("layer_visibility"),
        "fabrication_distortion": bm["fab_distortion"],
        "sem_acquisition": bm["sem_acquisition"],
        "reference_layer_intensities": bm["reference_layer_intensities"],
        "search_layer_intensities": bm["search_layer_intensities"],
        "reference_ops_enabled": bm.get("reference_ops_enabled", False),
        "reference_fabrication_distortion": bm.get("reference_fab_distortion"),
        "reference_sem_acquisition": bm.get("reference_sem_acquisition"),
    }


def _build_bookmarks_zip(bookmarks: list) -> bytes:
    """Bulk-export every bookmark: its Reference/Search CAD (.gds), its
    rendered images (.png), a slim ground-truth file, and the full
    parameter/metadata JSON -- one subfolder per bookmark.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, bm in enumerate(bookmarks):
            geom, sample = _sample_from_bookmark(bm)
            prefix = f"{i + 1:02d}_{bm['kind']}_seed{bm['seed']}"

            zf.writestr(f"{prefix}/reference.gds", _gds_bytes(sample["reference_cell"]))
            full_cell = _full_canvas_cell(geom["mats"], geom["num_layers"])
            zf.writestr(f"{prefix}/search.gds", _gds_bytes(full_cell))
            zf.writestr(f"{prefix}/reference.png", _png_bytes(sample["reference_preview"]))
            zf.writestr(f"{prefix}/search.png", _png_bytes(sample["search_img"]))

            gt = {
                "match_found": sample["match_found"],
                "gt_x": sample["gt_x"], "gt_y": sample["gt_y"], "gt_box": sample["gt_box"],
            }
            zf.writestr(f"{prefix}/gt.json", json.dumps(gt, indent=2))
            zf.writestr(f"{prefix}/metadata.json", json.dumps(_bookmark_metadata(bm, geom, sample), indent=2))
    return buf.getvalue()


with tab_generate:
    st.subheader("1. CAD structure")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        cad_kind = st.selectbox("Architecture kind", ["dram", "finfet"], key="cad_kind")
        cad_mat_size = st.slider("Array block (mat) size (nm)", 800.0, 5000.0, 2600.0, 100.0, key="cad_mat_size")
    with scol2:
        cad_strip_width = st.slider("Separator strip width (nm)", 80.0, 800.0, 320.0, 20.0, key="cad_strip_width")
        cad_collapse_threshold = st.slider("Pattern-collapse threshold (nm)", 0.0, 20.0, 10.0, 1.0, key="cad_collapse")
    with scol3:
        cad_no_match_prob = st.slider(
            "P(no match -- Reference cut from an unrelated canvas)", 0.0, 1.0, 0.08, 0.02, key="cad_nmp",
            help="A reference-selection choice, not a distortion -- whether this sample's Reference is cut "
                 "from the same canvas as the Search (a real match) or an unrelated one (no match). "
                 "Ignored in Manual crop mode below.",
        )

    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        cad_contact_aspect = st.slider(
            "Contact/via tablet aspect ratio", 1.0, 3.0, 1.6, 0.1, key="cad_contact_aspect",
            help="1.0 = round contacts/vias. Higher = more elongated capsule/tablet shape -- every contact "
                 "and via layer (bit_line_contact, storage_contact, storage_capacitor, via1, metal2_strap's "
                 "vias for DRAM; contact, via0, via1 for FinFET) uses this shape.",
        )
    with tcol2:
        cad_contact_thickness = st.slider(
            "Contact/via tablet thickness", 0.3, 3.0, 1.0, 0.1, key="cad_contact_thickness",
            help="Scales the capsule's short (thickness) dimension -- independent from the aspect ratio, "
                 "which only controls how much longer the long axis is relative to this thickness.",
        )
    with tcol3:
        cad_contact_angle = st.selectbox(
            "Contact/via tablet angle", [0, 30, 45, 60, 90], index=4, key="cad_contact_angle",
            help="Angle of the tablet's long axis, measured from the word-line axis -- 0deg = elongated "
                 "along word-lines, 90deg = elongated along bit-lines, anything else is diagonal.",
        )

    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        crop_mode = st.selectbox(
            "Reference crop location", ["Auto (straddle a boundary)", "Manual (pick center)"], key="crop_mode",
            help="Auto lets the generator pick a location for you (always straddling a mat/strip separator). "
                 "Manual instead lets you choose exactly where in the Search-side CAD the Reference is cut "
                 "from -- the generator no longer decides the crop for you.",
        )
    manual_center = None
    if crop_mode == "Manual (pick center)":
        with ccol2:
            manual_cx = st.slider(
                "Reference center X (nm)", 0.0, float(FINE_CANVAS_SIZE_PX), float(FINE_CANVAS_SIZE_PX) / 2.0, 50.0,
                key="manual_cx",
            )
        with ccol3:
            manual_cy = st.slider(
                "Reference center Y (nm)", 0.0, float(FINE_CANVAS_SIZE_PX), float(FINE_CANVAS_SIZE_PX) / 2.0, 50.0,
                key="manual_cy",
            )
        manual_center = (manual_cx, manual_cy)
        st.caption("Manual mode always produces a real match (P(no match) above is ignored) -- the green box "
                   "in the CAD viewer and Search image below tracks your chosen center as you drag the sliders.")

    rbcol1, rbcol2 = st.columns(2)
    with rbcol1:
        st.button("Randomize sample", on_click=_randomize, use_container_width=True)
    with rbcol2:
        st.button("Bookmark this sample", on_click=_bookmark_current, use_container_width=True)
    if st.session_state.pop("_just_bookmarked", False):
        st.toast("Bookmarked -- see the Bookmarks tab.", icon="🔖")

    # Layer count/names depend only on architecture kind, not on any geometry
    # build -- resolving them here (instead of from cad_geom) lets the sidebar
    # render its per-layer intensity sliders and the fabrication-distortion
    # controls *before* we need to call _cached_cad_geometry with their values.
    layer_names = CAD_LAYER_NAMES.get(cad_kind, [])
    num_layers = len(layer_names)

    with st.sidebar:
        st.header("Operations")
        st.caption("Apply to:")
        acol1, acol2 = st.columns(2)
        with acol1:
            apply_search = st.checkbox("Search Image", value=True, key="apply_search")
        with acol2:
            apply_ref = st.checkbox("Reference Image", value=False, key="apply_ref")
        if not apply_search and not apply_ref:
            apply_search = True
            st.session_state.apply_search = True
            st.caption("At least one target must stay selected -- defaulted back to Search Image.")
        elif apply_search and apply_ref:
            st.caption("Both selected -- every slider below writes the same value into Search's and "
                       "Reference's own settings at once.")

        st.button("Reset to no augmentation", on_click=_reset_operations, use_container_width=True)

        # Intensity keys have always used "search"/"ref" (not "cad"/"ref")
        # -- kept as-is here for backward compatibility with existing
        # bookmarks and the JSON panel below.
        intensity_prefixes = (["search"] if apply_search else []) + (["ref"] if apply_ref else [])
        ops_prefixes = (["cad"] if apply_search else []) + (["ref"] if apply_ref else [])

        target_label = " + ".join(
            label for label, on in [("Search", apply_search), ("Reference", apply_ref)] if on
        )
        with st.expander(f"Layer intensity ({target_label})", expanded=True):
            st.caption("Secondary-electron yield, 0-255. Reference and Search each keep their own choice here -- "
                       "useful for simulating a calibration mismatch between the design tool and the imaging tool.")
            _multi_slider(
                "Hide layers below index (0 = show all)", 0, num_layers - 1, 0, 1, "min_layer", intensity_prefixes,
                help="A close-FOV capture resolves deeper/more-buried layers (layer 0 = most buried) than a "
                     "coarser, wide-FOV one -- raise this to simulate that side's layers collapsing out of "
                     "resolution. Search and Reference keep independent values, same as intensity above.",
            )
            for i in range(num_layers):
                default_intensity = yield_model.yield_to_intensity(yield_model.layer_yield(i, num_layers))
                label = layer_names[i] if i < len(layer_names) else f"layer {i}"
                _multi_slider(
                    f"L{i}: {label}", 0, 255, int(default_intensity), 1, f"intensity_{i}", intensity_prefixes,
                )

        if apply_ref:
            st.caption("Reference Image is checked, so it now gets its own full imaging/distortion pipeline "
                       "below (instead of staying the exact, undistorted design) -- uncheck it to go back to "
                       "the pristine design with only the intensity choice above.")
        else:
            st.caption("Only Search is checked -- the Reference stays the exact, undistorted design; only "
                       "intensity (above) affects it. Check Reference Image too to reach its own controls.")
        _render_operation_expanders(ops_prefixes)

        st.divider()
        show_gt_box = st.checkbox("Show ground-truth box", value=True, key="show_gt")

    cad_geom = _cached_cad_geometry(
        cad_kind, st.session_state.cad_seed, cad_mat_size, cad_strip_width, cad_collapse_threshold,
        st.session_state.get("cad_psp", 0.10), st.session_state.get("cad_psr", 0.10), cad_no_match_prob,
        st.session_state.get("cad_lwb", 0.0), st.session_state.get("cad_corner_r", 0.0), manual_center,
        cad_contact_aspect, cad_contact_thickness, cad_contact_angle,
    )

    ref_layer_intensities = _get_intensities("ref", num_layers)
    search_layer_intensities = _get_intensities("search", num_layers)
    search_min_layer = st.session_state.get("search_min_layer", 0)
    reference_min_layer = st.session_state.get("ref_min_layer", 0)

    render_params = _read_operation_params(
        "cad", no_match_prob=cad_no_match_prob, mat_size_nm=cad_mat_size,
        strip_width_nm=cad_strip_width, collapse_threshold_nm=cad_collapse_threshold,
        contact_aspect_ratio=cad_contact_aspect,
        contact_thickness_factor=cad_contact_thickness, contact_angle_deg=cad_contact_angle,
    )
    reference_render_params = _read_operation_params("ref") if apply_ref else None
    cad_sample = render_cad_sample(
        cad_geom, render_params,
        layer_intensities=search_layer_intensities, reference_layer_intensities=ref_layer_intensities,
        reference_render_params=reference_render_params,
        search_min_layer=search_min_layer, reference_min_layer=reference_min_layer,
    )

    # Identifies exactly what this sample is, so a ZNCC result computed for
    # one sample never gets displayed/drawn against a different one after
    # the user moves a slider -- see the staleness check below.
    sample_signature = json.dumps({
        "kind": cad_kind, "seed": st.session_state.cad_seed, "manual_center": manual_center,
        "render_params": render_params.as_dict(),
        "reference_render_params": reference_render_params.as_dict() if reference_render_params else None,
        "ref_intensities": ref_layer_intensities, "search_intensities": search_layer_intensities,
        "search_min_layer": search_min_layer, "reference_min_layer": reference_min_layer,
    }, sort_keys=True, default=str)
    zncc_result = None
    if st.session_state.get("zncc_signature") == sample_signature:
        zncc_result = st.session_state.get("zncc_result")

    st.subheader("2. GDS viewer (layer-colored, no intensity applied yet -- always the clean design)")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.caption("Reference (the clipped crop that becomes reference.gds) -- always straddles a separator")
        ref_colored = rasterize_cell_layers_colored(
            cad_geom["reference_cell"], 1000, num_layers, min_layer=reference_min_layer,
        )
        st.image(cv2.cvtColor(ref_colored, cv2.COLOR_BGR2RGB), use_container_width=True)
    with vcol2:
        st.caption("Entire Search-side CAD, as designed -- green box marks where the Reference is cut from")
        full_colored = _cached_full_canvas_colored(
            cad_kind, st.session_state.cad_seed, cad_mat_size, cad_strip_width, cad_collapse_threshold,
            st.session_state.get("cad_psp", 0.10), st.session_state.get("cad_psr", 0.10), cad_no_match_prob,
            st.session_state.get("cad_lwb", 0.0), st.session_state.get("cad_corner_r", 0.0), manual_center,
            cad_contact_aspect, cad_contact_thickness, cad_contact_angle, search_min_layer,
        )
        full_colored_display = full_colored.copy()
        if cad_sample["match_found"]:
            fx0, fy0, fw, fh = cad_sample["gt_box"]
            _draw_crosshair(full_colored_display, fx0 + fw / 2.0, fy0 + fh / 2.0, GT_COLOR_BGR)
        st.image(cv2.cvtColor(full_colored_display, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption("Legend: " + " | ".join(
        f"layer {i} ({layer_names[i] if i < len(layer_names) else '?'})" for i in range(num_layers)
    ) + " | dark gray = separator")

    cad_search_display = cv2.cvtColor(cad_sample["search_img"], cv2.COLOR_GRAY2BGR)
    if cad_sample["match_found"] and show_gt_box:
        x0, y0, w, h = cad_sample["gt_box"]
        _draw_crosshair(cad_search_display, x0 + w / 2.0, y0 + h / 2.0, GT_COLOR_BGR)
    if zncc_result is not None:
        _draw_crosshair(cad_search_display, zncc_result["x"], zncc_result["y"], PRED_COLOR_BGR)

    st.subheader("3. SEM images (rendered with the operations panel's settings)")
    st.caption("Green = ideal/true (ground-truth) location. Red = a matcher's found location, after running ZNCC below.")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        ref_caption = (
            "Reference (its own independent imaging/distortion settings applied)" if apply_ref
            else "Reference (rendered with the Reference intensity choice -- never SEM-imaged)"
        )
        st.caption(ref_caption)
        st.image(cad_sample["reference_preview"], clamp=True, use_container_width=True)
    with dcol2:
        st.caption("Search (10 nm/px, noisy SEM capture)")
        st.image(cv2.cvtColor(cad_search_display, cv2.COLOR_BGR2RGB), use_container_width=True)

    if cad_sample["match_found"]:
        st.markdown(f"**Ground truth center:** `({cad_sample['gt_x']:.1f}, {cad_sample['gt_y']:.1f})` px "
                    f"&nbsp;&nbsp;|&nbsp;&nbsp; **Kind:** `{cad_kind}` &nbsp;&nbsp;|&nbsp;&nbsp; **Seed:** `{st.session_state.cad_seed}`")
    else:
        st.warning(f"**NO MATCH** -- this Reference CAD was cut from an unrelated canvas. "
                   f"&nbsp;&nbsp;|&nbsp;&nbsp; **Kind:** `{cad_kind}` &nbsp;&nbsp;|&nbsp;&nbsp; **Seed:** `{st.session_state.cad_seed}`")

    def _run_zncc(sample=cad_sample, signature=sample_signature):
        result = zncc_match(sample["reference_preview"], sample["search_img"])
        st.session_state["zncc_result"] = result
        st.session_state["zncc_signature"] = signature

    if cad_sample["match_found"]:
        st.button("Run ZNCC baseline on this sample", on_click=_run_zncc, key="run_zncc_btn")
        if zncc_result is not None:
            zncc_shift = shift_report(zncc_result["x"], zncc_result["y"], cad_sample["gt_x"], cad_sample["gt_y"])
            st.markdown(
                f"**ZNCC prediction (red box):** `({zncc_result['x']:.2f}, {zncc_result['y']:.2f})` at scale {zncc_result['scale']}, "
                f"score `{zncc_result['score']:.4f}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**shift (predicted - actual):** `dx={zncc_shift['dx']:+.2f} dy={zncc_shift['dy']:+.2f}` "
                f"&nbsp;&nbsp;|&nbsp;&nbsp; **distance:** `{zncc_shift['distance_px']:.2f} px`"
            )
            if zncc_shift["distance_px"] <= 5.0:
                st.success("Matched within 5 px tolerance.")
            else:
                st.warning(
                    "Missed (>5 px off) -- confidence scores on this task stay high even when wrong: the "
                    "multi-mat field means a naive matcher can lock onto the wrong mat entirely, not just the "
                    "wrong periodic repeat within one mat. See baseline_solution/cad_infer.py."
                )

    with st.expander("Sample parameters (JSON)", expanded=False):
        st.json({
            "kind": cad_kind,
            "seed": st.session_state.cad_seed,
            "match_found": cad_sample["match_found"],
            "gt_x": cad_sample["gt_x"], "gt_y": cad_sample["gt_y"], "gt_box": cad_sample["gt_box"],
            "structure": {
                "mat_size_nm": cad_mat_size, "strip_width_nm": cad_strip_width,
                "collapse_threshold_nm": cad_collapse_threshold,
                "no_match_prob": cad_no_match_prob,
                "contact_aspect_ratio": cad_contact_aspect,
                "contact_thickness_factor": cad_contact_thickness, "contact_angle_deg": cad_contact_angle,
            },
            "layer_visibility": {"search_min_layer": search_min_layer, "reference_min_layer": reference_min_layer},
            "crop": {"mode": crop_mode, "manual_center_nm": manual_center},
            "fabrication_distortion": {
                "polygon_scale_prob": render_params.polygon_scale_prob,
                "polygon_scale_range": render_params.polygon_scale_range,
                "linewidth_bias_nm": render_params.linewidth_bias_nm,
                "corner_rounding_px": render_params.corner_rounding_px,
            },
            "sem_acquisition": {
                "dose_search": render_params.dose_search, "beam_spot_size_nm": render_params.beam_spot_size_nm,
                "shear_amplitude_px": render_params.shear_amplitude_px, "drift_jitter_px": render_params.drift_jitter_px,
                "detector_noise_sigma_search": render_params.detector_noise_sigma_search,
                "astigmatism_ratio": render_params.astigmatism_ratio,
                "search_rotation_deg": render_params.search_rotation_deg,
                "barrel_distortion_k": render_params.barrel_distortion_k,
                "vignette_strength": render_params.vignette_strength,
                "gamma": render_params.gamma,
                "charging_streak_prob": render_params.charging_streak_prob,
                "charging_streak_intensity": render_params.charging_streak_intensity,
                "speckle_sigma": render_params.speckle_sigma,
                "salt_pepper_prob": render_params.salt_pepper_prob,
            },
            "reference_layer_intensities": ref_layer_intensities,
            "search_layer_intensities": search_layer_intensities,
            "reference_ops_enabled": apply_ref,
            "reference_fabrication_distortion": {
                "polygon_scale_prob": reference_render_params.polygon_scale_prob,
                "polygon_scale_range": reference_render_params.polygon_scale_range,
                "linewidth_bias_nm": reference_render_params.linewidth_bias_nm,
                "corner_rounding_px": reference_render_params.corner_rounding_px,
            } if reference_render_params else None,
            "reference_sem_acquisition": {
                "dose_search": reference_render_params.dose_search,
                "beam_spot_size_nm": reference_render_params.beam_spot_size_nm,
                "shear_amplitude_px": reference_render_params.shear_amplitude_px,
                "drift_jitter_px": reference_render_params.drift_jitter_px,
                "detector_noise_sigma_search": reference_render_params.detector_noise_sigma_search,
                "astigmatism_ratio": reference_render_params.astigmatism_ratio,
                "search_rotation_deg": reference_render_params.search_rotation_deg,
                "barrel_distortion_k": reference_render_params.barrel_distortion_k,
                "vignette_strength": reference_render_params.vignette_strength,
                "gamma": reference_render_params.gamma,
                "charging_streak_prob": reference_render_params.charging_streak_prob,
                "charging_streak_intensity": reference_render_params.charging_streak_intensity,
                "speckle_sigma": reference_render_params.speckle_sigma,
                "salt_pepper_prob": reference_render_params.salt_pepper_prob,
            } if reference_render_params else None,
        })

    st.subheader("4. Download")
    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
    with gcol1:
        st.download_button(
            "CAD - Reference (.gds)", _gds_bytes(cad_sample["reference_cell"]), "reference.gds",
            "application/octet-stream", key="dl_cad_reference", use_container_width=True,
        )
    with gcol2:
        full_cell = _full_canvas_cell(cad_geom["mats"], num_layers)
        st.download_button(
            "CAD - Search (.gds)", _gds_bytes(full_cell), "search.gds",
            "application/octet-stream", key="dl_cad_search", use_container_width=True,
        )
    with gcol3:
        st.download_button(
            "SEM Image - Reference (.png)", _png_bytes(cad_sample["reference_preview"]), "reference.png",
            "image/png", key="dl_sem_reference", use_container_width=True,
        )
    with gcol4:
        st.download_button(
            "SEM Image - Search (.png)", _png_bytes(cad_sample["search_img"]), "search.png",
            "image/png", key="dl_sem_search", use_container_width=True,
        )


with tab_bookmarks:
    st.subheader("Bookmarked samples")
    st.caption(
        "\"Generate all\" builds 20 curated teaching samples, each combining 2-3 augmentations "
        "at once (rotation, drift, noise, CD bias, ...) -- run through vanilla ZNCC so you can "
        "see where plain intensity correlation degrades or fails outright, and why a real "
        "pipeline needs more than that. Saved to disk, so they reload automatically next time "
        "the app starts."
    )
    gcol1, gcol2 = st.columns([1, 3])
    with gcol1:
        generate_all_clicked = st.button(
            "Generate all (20 curated)", use_container_width=True, key="generate_all_curated",
        )
    if generate_all_clicked:
        progress_bar = st.progress(0.0, text="Starting...")

        def _on_progress(done, total, label):
            progress_bar.progress(done / total, text=f"[{done}/{total}] {label}")

        st.session_state.bookmarks = generate_curated_bookmarks(progress_callback=_on_progress)
        progress_bar.empty()
        st.rerun()

    bookmarks = st.session_state.bookmarks

    if not bookmarks:
        st.info('No bookmarks yet -- click "Generate all" above, or use "Bookmark this sample" in the Generate tab.')
    else:
        hcol1, hcol2, hcol3 = st.columns([2, 1, 1])
        with hcol1:
            st.caption(f"{len(bookmarks)} bookmarked sample(s)")
        with hcol2:
            st.download_button(
                "Download all as .zip", _build_bookmarks_zip(bookmarks), "bookmarked_samples.zip",
                "application/zip", key="dl_bookmarks_zip", use_container_width=True,
            )
        with hcol3:
            if st.button("Clear all", use_container_width=True):
                st.session_state.bookmarks = []
                st.rerun()

        remove_id = None
        for bm in bookmarks:
            geom, sample = _sample_from_bookmark(bm)
            with st.container(border=True):
                title = bm.get("label") or f"{bm['kind']} seed={bm['seed']}"
                st.markdown(f"**{title}**")
                if bm.get("challenge"):
                    st.caption(bm["challenge"])

                cols = st.columns([1, 1, 1, 1.2])
                with cols[0]:
                    ref_colored_bm = rasterize_cell_layers_colored(sample["reference_cell"], 1000, geom["num_layers"])
                    st.image(cv2.cvtColor(ref_colored_bm, cv2.COLOR_BGR2RGB), caption="Reference CAD", use_container_width=True)
                with cols[1]:
                    st.image(sample["reference_preview"], clamp=True, caption="Reference image (from CAD)", use_container_width=True)
                with cols[2]:
                    search_disp = cv2.cvtColor(sample["search_img"], cv2.COLOR_GRAY2BGR)
                    if sample["match_found"]:
                        bx0, by0, bw, bh = sample["gt_box"]
                        _draw_crosshair(search_disp, bx0 + bw / 2.0, by0 + bh / 2.0, GT_COLOR_BGR)
                    zncc_info = bm.get("zncc")
                    if zncc_info and "template_w" in zncc_info:
                        _draw_crosshair(search_disp, zncc_info["x"], zncc_info["y"], PRED_COLOR_BGR)
                    st.image(cv2.cvtColor(search_disp, cv2.COLOR_BGR2RGB), caption="Search SEM image", use_container_width=True)
                with cols[3]:
                    st.markdown(f"**{bm['kind']}** &nbsp; seed=`{bm['seed']}`")
                    gt_str = f"({sample['gt_x']:.1f}, {sample['gt_y']:.1f})" if sample["match_found"] else "NO MATCH"
                    st.caption(f"GT (green): {gt_str} &nbsp; | &nbsp; strip width: {bm['structure']['strip_width_nm']:.0f}nm")
                    if zncc_info:
                        st.caption(
                            f"Vanilla ZNCC (red): dist={zncc_info['distance_px']:.1f}px, score={zncc_info['score']:.3f}"
                        )
                    with st.expander("Params (JSON)"):
                        st.json(_bookmark_metadata(bm, geom, sample))
                    if st.button("Remove", key=f"remove_bm_{bm['id']}"):
                        remove_id = bm["id"]
        if remove_id is not None:
            st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b["id"] != remove_id]
            st.rerun()
