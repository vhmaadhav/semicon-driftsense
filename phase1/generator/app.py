"""
Drift-Sense synthetic data explorer -- Streamlit app for Hugging Face Spaces.

This is a thin UI shell around the real generator in `src/`: every image
shown here comes from the exact same `generate_sample()` pipeline students
use from the CLI, so nothing here is a separate/approximated reimplementation.
"""

import io
import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline import GenerationParams, generate_sample, generate_sample_family, SEARCH_VARIANTS
from src.presets import PRESETS, get_preset
from src.patterns.dram import generate_dram_canvas
from src.patterns.finfet import generate_finfet_canvas
from baseline_solution.zncc import zncc_match, shift_report
import visualize_layers as layer_viz

st.set_page_config(page_title="Drift-Sense Synthetic Data Explorer", layout="wide")
st.title("Drift-Sense: Synthetic Dataset Explorer")
st.caption(
    "Reference: 1000x1000 px @ 1 nm/px (1 um FOV). "
    "Search: 1000x1000 px @ 10 nm/px (10 um FOV). "
    "The reference's footprint in the search image is 100x100 px -- the '10x shrink' "
    "in the problem statement falls directly out of that pixel-size ratio."
)

if "seed" not in st.session_state:
    st.session_state.seed = 42

tab_explorer, tab_family, tab_layers, tab_eval = st.tabs(
    ["Single-sample explorer", "Sample family (5 search variants)", "Intermediate layers", "Baseline evaluation"]
)

with st.sidebar:
    st.header("Structure")
    architecture = st.selectbox("Architecture preset", list(PRESETS.keys()))
    feature_scale = st.slider(
        "Feature size scale", 0.5, 2.0, 1.0, 0.05,
        help="Scales every pitch/width in the preset proportionally -- explore other process nodes.",
    )

    st.header("SEM imaging physics")
    beam_spot_size_nm = st.slider(
        "Beam spot size (nm)", 1.0, 20.0, 5.0, 0.5,
        help="Single physical PSF blur applied before downsampling. Larger spot = "
             "more smearing of dense structures once downsampled into the search image.",
    )
    collapse_threshold_nm = st.slider(
        "Pattern-collapse threshold (nm)", 0.0, 20.0, 10.0, 1.0,
        help="Gaps/lines below this (structural defect, not imaging) probabilistically bridge. "
             "Default 10 nm = exactly 1 px at the search image's 10 nm/px resolution.",
    )

    st.header("Acquisition noise")
    dose_reference = st.slider("Reference dose (higher = cleaner)", 100.0, 5000.0, 2000.0, 100.0)
    dose_search = st.slider("Search dose (higher = cleaner)", 20.0, 2000.0, 200.0, 20.0)
    shear_amplitude_px = st.slider("Search raster drift/shear (px)", 0.0, 5.0, 1.5, 0.1)
    drift_jitter_px = st.slider("Search row jitter (px)", 0.0, 3.0, 0.5, 0.1)

    st.header("Distortion & polygon scaling")
    linewidth_bias_nm = st.slider(
        "Linewidth/CD bias (nm)", -10.0, 10.0, 0.0, 0.5,
        help="Deterministic global over/under-exposure bias applied to every drawn feature.",
    )
    corner_rounding_px = st.slider(
        "Corner rounding (px)", 0.0, 6.0, 0.0, 0.5,
        help="Morphological rounding of polygon corners -- real litho/etch never draws sharp corners.",
    )
    astigmatism_ratio = st.slider(
        "Beam astigmatism ratio", 0.5, 2.0, 1.0, 0.05,
        help="Elliptical beam spot (sigmaY = sigmaX * ratio) -- directional blur.",
    )
    barrel_distortion_k = st.slider(
        "Barrel(+)/pincushion(-) distortion", -0.15, 0.15, 0.0, 0.01,
        help="Radial lens-style distortion from imperfect beam-scan linearity.",
    )
    vignette_strength = st.slider("Vignette strength", 0.0, 1.0, 0.0, 0.05)
    gamma = st.slider("Gamma (contrast curve)", 0.4, 2.5, 1.0, 0.05)
    charging_streak_prob = st.slider(
        "Charging streaks (per 100 rows)", 0.0, 5.0, 0.0, 0.25,
        help="Bright horizontal streaks from local sample charging on insulating regions.",
    )
    charging_streak_intensity = st.slider("Charging streak intensity", 0.0, 3.0, 0.0, 0.1)
    speckle_sigma = st.slider(
        "Speckle noise sigma (multiplicative)", 0.0, 1.0, 0.0, 0.05,
        help="out = img * (1 + N(0, sigma)) -- noise magnitude scales with brightness, unlike additive detector noise.",
    )
    salt_pepper_prob = st.slider(
        "Salt-and-pepper probability", 0.0, 0.05, 0.0, 0.005,
        help="Fraction of pixels forced to pure black/white -- dead/hot pixels or discharge events.",
    )

    st.header("Die layout (multi-region)")
    mat_size_nm = st.slider(
        "Array block (mat) size (nm)", 800.0, 5000.0, 2600.0, 100.0,
        help="Size of each independently-generated structure block before a separating strip.",
    )
    strip_width_nm = st.slider(
        "Separator strip width (nm)", 80.0, 800.0, 320.0, 20.0,
        help="Width of the peripheral/routing material band between array blocks.",
    )
    boundary_bias = st.slider(
        "P(reference crop straddles a boundary)", 0.0, 1.0, 0.35, 0.05,
        help="Probability the Reference crop is deliberately placed across a mat/strip edge "
             "instead of sampled uniformly at random.",
    )

    st.header(" ")
    show_gt_box = st.checkbox("Show ground-truth box", value=True)
    if st.button("Regenerate (new seed)"):
        st.session_state.seed = np.random.randint(0, 2_000_000_000)


def build_params():
    return GenerationParams(
        beam_spot_size_nm=beam_spot_size_nm,
        collapse_threshold_nm=collapse_threshold_nm,
        dose_reference=dose_reference,
        dose_search=dose_search,
        shear_amplitude_px=shear_amplitude_px,
        drift_jitter_px=drift_jitter_px,
        linewidth_bias_nm=linewidth_bias_nm,
        corner_rounding_px=corner_rounding_px,
        astigmatism_ratio=astigmatism_ratio,
        barrel_distortion_k=barrel_distortion_k,
        vignette_strength=vignette_strength,
        gamma=gamma,
        charging_streak_prob=charging_streak_prob,
        charging_streak_intensity=charging_streak_intensity,
        speckle_sigma=speckle_sigma,
        salt_pepper_prob=salt_pepper_prob,
        mat_size_nm=mat_size_nm,
        strip_width_nm=strip_width_nm,
        boundary_bias=boundary_bias,
    )


def to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


preset_overrides = None
if feature_scale != 1.0:
    base_preset = get_preset(architecture)
    preset_overrides = {
        k: v * feature_scale for k, v in base_preset.items() if k.endswith("_nm")
    }
    st.sidebar.caption(f"Scaled preset: {preset_overrides}")

with tab_explorer:
    rng = np.random.default_rng(st.session_state.seed)
    params = build_params()
    sample = generate_sample(architecture, rng, params, preset_overrides)

    search_display = cv2.cvtColor(sample["search_img"], cv2.COLOR_GRAY2BGR)
    if show_gt_box:
        x0, y0, w, h = sample["gt_box"]
        cv2.rectangle(
            search_display,
            (int(round(x0)), int(round(y0))),
            (int(round(x0 + w)), int(round(y0 + h))),
            (0, 0, 255),
            2,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reference (1 nm/px)")
        st.image(sample["reference_img"], clamp=True, use_container_width=True)
    with col2:
        st.subheader("Search (10 nm/px)")
        st.image(cv2.cvtColor(search_display, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown(
        f"**Ground truth center in search image:** `({sample['gt_x']:.1f}, {sample['gt_y']:.1f})` px "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; **Architecture:** `{architecture}` &nbsp;&nbsp;|&nbsp;&nbsp; **Seed:** `{st.session_state.seed}`"
    )

    if st.checkbox("Run ZNCC matcher on this sample", value=False):
        match = zncc_match(sample["reference_img"], sample["search_img"])
        shift = shift_report(match["x"], match["y"], sample["gt_x"], sample["gt_y"])
        st.markdown(
            f"**ZNCC prediction:** `({match['x']:.2f}, {match['y']:.2f})` at scale {match['scale']}, "
            f"score `{match['score']:.4f}` &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**shift (predicted - actual):** `dx={shift['dx']:+.2f} dy={shift['dy']:+.2f}` "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; **distance:** `{shift['distance_px']:.2f} px`"
        )

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("Download reference.png", to_png_bytes(sample["reference_img"]), "reference.png", "image/png")
    with dl2:
        st.download_button("Download search.png", to_png_bytes(sample["search_img"]), "search.png", "image/png")
    with dl3:
        metadata = {
            "architecture": architecture,
            "gt_x": sample["gt_x"],
            "gt_y": sample["gt_y"],
            "gt_box": sample["gt_box"],
            "seed": st.session_state.seed,
            **sample["params"],
        }
        st.download_button("Download metadata.json", json.dumps(metadata, indent=2), "metadata.json", "application/json")


FAMILY_ARCHS = ["dram_1x", "finfet_10nm"]
N_FAMILY_SAMPLES = 10


@st.cache_data(show_spinner="Generating sample family...")
def _cached_family(architecture: str, sample_idx: int):
    base_seed = 5000 + sample_idx * 101 + (0 if architecture.startswith("dram") else 50000)
    return generate_sample_family(architecture, base_seed, GenerationParams())


with tab_family:
    st.markdown(
        "One fixed **Reference** crop per sample, paired with **5 differently-imaged Search "
        "images** (same physical scene, different acquisition conditions: dose, drift, "
        "speckle/impulse noise, charging). The point: the same Reference should still be "
        "findable across all of them -- pick a sample and switch the dropdown."
    )
    fc1, fc2 = st.columns(2)
    with fc1:
        family_arch = st.selectbox("Architecture", FAMILY_ARCHS, key="family_arch")
    with fc2:
        sample_idx = st.selectbox("Sample #", list(range(N_FAMILY_SAMPLES)), key="family_sample_idx")

    family = _cached_family(family_arch, sample_idx)
    variant_labels = [v["label"] for v in family["search_variants"]]
    variant_label = st.selectbox("Search image variant", variant_labels, key="family_variant")
    variant = next(v for v in family["search_variants"] if v["label"] == variant_label)

    show_gt_family = st.checkbox("Show ground-truth box", value=True, key="family_show_gt")
    search_disp = cv2.cvtColor(variant["search_img"], cv2.COLOR_GRAY2BGR)
    if show_gt_family:
        x0, y0, w, h = family["gt_box"]
        cv2.rectangle(search_disp, (int(round(x0)), int(round(y0))), (int(round(x0 + w)), int(round(y0 + h))), (0, 0, 255), 2)

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.subheader("Reference (fixed)")
        st.image(family["reference_img"], clamp=True, use_container_width=True)
    with fcol2:
        st.subheader(f"Search -- variant: {variant_label}")
        st.image(cv2.cvtColor(search_disp, cv2.COLOR_BGR2RGB), use_container_width=True)

    match = zncc_match(family["reference_img"], variant["search_img"])
    shift = shift_report(match["x"], match["y"], family["gt_x"], family["gt_y"])
    st.markdown(
        f"**Ground truth:** `({family['gt_x']:.1f}, {family['gt_y']:.1f})` &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**ZNCC prediction:** `({match['x']:.2f}, {match['y']:.2f})` score `{match['score']:.4f}` "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; **shift (predicted-actual):** `dx={shift['dx']:+.2f} dy={shift['dy']:+.2f}` "
        f"distance=`{shift['distance_px']:.2f} px`"
    )
    if shift["distance_px"] <= 5.0:
        st.success("Matched within 5 px tolerance.")
    else:
        st.warning("Matcher missed this one (>5 px off) -- a real limitation to improve on.")


LAYER_ORDER = {"dram": ["substrate", "word_line", "bit_line", "storage_contact"], "finfet": ["substrate", "fin", "gate", "contact"]}
LAYER_GENERATORS = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}

with tab_layers:
    st.markdown(
        "The flattened grayscale image is what a real SEM actually images (it only sees the "
        "top exposed surface) -- but the generator draws it up from separate structural layers. "
        "This view shows those layers individually, false-colored, plus a simple exploded stack."
    )
    layer_arch = st.selectbox("Architecture", list(PRESETS.keys()), key="layer_arch")
    layer_seed = st.slider("Seed", 0, 50, 3, key="layer_seed")
    layer_size = st.slider("Patch size (px)", 200, 800, 500, 50, key="layer_size")

    layer_preset = get_preset(layer_arch)
    layer_kind = layer_preset["kind"]
    layer_rng = np.random.default_rng(layer_seed)
    canvas, layers = LAYER_GENERATORS[layer_kind](layer_size, layer_preset, 10.0, layer_rng, return_layers=True)

    order = LAYER_ORDER[layer_kind]
    cols = st.columns(len(order))
    for col, name in zip(cols, order):
        colored = layer_viz.false_color(layers[name], layer_viz._LAYER_COLORS[name])
        with col:
            st.image(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), caption=name, use_container_width=True)

    st.subheader("Exploded stack")
    layers_bgr = [layer_viz.false_color(layers[n], layer_viz._LAYER_COLORS[n]) for n in order]
    exploded = layer_viz.build_exploded_stack(layers_bgr, [n.replace("_", " ") for n in order])
    st.image(cv2.cvtColor(exploded, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Flattened composite (what SEM actually images)")
    st.image(canvas, clamp=True, use_container_width=True)


with tab_eval:
    st.markdown(
        "Precision-Recall of the ZNCC baseline matcher across noise levels. ZNCC score is used "
        "as the acceptance threshold; a prediction is a true positive if within the tolerance of "
        "ground truth. Run via `python baseline_solution/evaluate.py`."
    )
    pr_path = "baseline_solution/eval_results/pr_curves.png"
    trend_path = "baseline_solution/eval_results/ap_vs_noise.png"
    if os.path.exists(pr_path) and os.path.exists(trend_path):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.image(pr_path, caption="Precision-Recall by noise level", use_container_width=True)
        with ecol2:
            st.image(trend_path, caption="AP / accuracy vs noise level", use_container_width=True)
    else:
        st.info(
            "No evaluation results found yet. Run:\n\n"
            "`python baseline_solution/evaluate.py --samples-per-level 40 --tolerance-px 5`\n\n"
            "then reload this page."
        )
