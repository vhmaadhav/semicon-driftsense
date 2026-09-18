"""
Curated demo/teaching bookmarks for the Bookmarks tab -- 20 deliberately
designed samples, one per augmentation category, so a viewer can see
exactly what each distortion looks like in isolation before facing them
combined. Generated on demand (the app's "Generate all" button) and cached
to disk (BOOKMARKS_STORE_PATH) so a fresh app launch loads the same set
instead of starting empty.

Design rules (per the organizer's brief):
  - Reference gets a *slight blur only* -- a close-FOV design-review capture
    isn't noise-free in reality (there's always some point-spread), but it
    is never fabrication-distorted or acquisition-noisy the way the Search
    side is. See REFERENCE_BLUR_ONLY.
  - Every case combines 2-3 Search-side augmentations at once, not one
    variable in isolation -- vanilla ZNCC (correlating raw intensity/yield)
    already solves an isolated-variable case, which teaches nothing about
    why a real pipeline needs more than intensity correlation. Stacking
    geometric distortion (rotation, shear, drift) with acquisition/fab
    effects is what actually forces a feature/edge-based approach.
  - Whole-frame rotation is capped at MAX_SEARCH_ROTATION_DEG (10 degrees)
    -- a few degrees of stage/placement error is realistic, tens is not --
    and is always a *global* transform (one angle for the whole frame, not
    per-mat), never combined with per-cell structural variation.
  - Barrel distortion is never used anywhere, by request.
  - Separator strip width is swept across [80, 200] nm, one value per case.
  - Crop location is center-biased (center_bias=True): the Reference always
    straddles the boundary strip nearest the canvas center, not a random
    one -- keeps the ground truth away from the edges, and away from
    whichever peripheral mat happens to look most similar.
  - Each case is regenerated with a fresh seed (up to a few attempts) if a
    vanilla ZNCC run lands on a *confident but wrong* match -- a sign the
    array's periodicity made some other mat look just as good as the real
    one, which isn't the lesson that case is supposed to teach.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cad_pipeline import CadGenerationParams, build_cad_geometry, render_cad_sample, MAX_SEARCH_ROTATION_DEG
from src.cad import yield_model
from baseline_solution.zncc import zncc_match, shift_report

BOOKMARKS_STORE_PATH = Path(__file__).parent / "bookmarks_store.json"

# A clean Search-side baseline -- every case below overrides only the one
# or two fields that are "the point" of that case, everything else stays
# at this baseline so the isolated effect is unambiguous.
BASELINE_SEARCH = dict(
    dose_search=1500.0, beam_spot_size_nm=3.0,
    shear_amplitude_px=0.0, drift_jitter_px=0.0, detector_noise_sigma_search=0.0,
    astigmatism_ratio=1.0, search_rotation_deg=0.0, barrel_distortion_k=0.0,
    vignette_strength=0.0, gamma=1.0,
    charging_streak_prob=0.0, charging_streak_intensity=0.0,
    speckle_sigma=0.0, salt_pepper_prob=0.0,
    polygon_scale_prob=0.0, polygon_scale_range=0.0, linewidth_bias_nm=0.0, corner_rounding_px=0.0,
)

# The Reference's own render params -- slight blur only, nothing else.
REFERENCE_BLUR_ONLY = dict(
    dose_search=3000.0, beam_spot_size_nm=2.0,
    shear_amplitude_px=0.0, drift_jitter_px=0.0, detector_noise_sigma_search=0.0,
    astigmatism_ratio=1.0, search_rotation_deg=0.0, barrel_distortion_k=0.0,
    vignette_strength=0.0, gamma=1.0,
    charging_streak_prob=0.0, charging_streak_intensity=0.0,
    speckle_sigma=0.0, salt_pepper_prob=0.0,
    polygon_scale_prob=0.0, polygon_scale_range=0.0, linewidth_bias_nm=0.0, corner_rounding_px=0.0,
)

CURATED_CASES = [
    {"label": "Rotation + raster shear + row jitter", "kind": "dram", "overrides": {
        "search_rotation_deg": 6.0, "shear_amplitude_px": 4.0, "drift_jitter_px": 1.8,
    }, "challenge": "The wordline/bitline array sits at a real stage angle, and the raster scan "
                    "shears and jitters on top of that. Three independent geometric distortions "
                    "stack -- no single linear correction undoes all of them at once."},
    {"label": "Rotation + starved dose + detector noise", "kind": "finfet", "overrides": {
        "search_rotation_deg": 9.0, "dose_search": 50.0, "detector_noise_sigma_search": 8.0,
    }, "challenge": "Near-maximum stage rotation on top of a starved electron dose and heavy "
                    "detector noise. Fin/gate edges are both rotated and buried in grain."},
    {"label": "Rotation + shear + CD bias grow", "kind": "dram", "overrides": {
        "search_rotation_deg": 5.0, "shear_amplitude_px": 3.5, "linewidth_bias_nm": 12.0,
    }, "challenge": "The fabricated storage-node contacts run wider than the design, the raster "
                    "shears progressively top to bottom, and the frame is rotated on top of both."},
    {"label": "Rotation + row jitter + vignette", "kind": "finfet", "overrides": {
        "search_rotation_deg": 8.0, "drift_jitter_px": 2.2, "vignette_strength": 0.5,
    }, "challenge": "Row-to-row scan jitter and strong edge vignetting wash out the gate/contact "
                    "signal near the frame border, while the whole scene sits at an angle."},
    {"label": "Rotation + starved dose + polygon-scale outliers", "kind": "dram", "overrides": {
        "search_rotation_deg": 4.0, "dose_search": 70.0, "polygon_scale_prob": 0.5, "polygon_scale_range": 0.3,
    }, "challenge": "Local process variation scatters oversized/undersized contacts through the "
                    "array, shot noise from a starved dose roughens every pixel, and it's rotated too."},
    {"label": "Max rotation + shear + astigmatism", "kind": "finfet", "overrides": {
        "search_rotation_deg": 10.0, "shear_amplitude_px": 3.0, "astigmatism_ratio": 1.8,
    }, "challenge": "The full 10-degree rotation cap combined with raster shear and a strongly "
                    "elliptical beam -- fin edges are directionally smeared, sheared, AND rotated."},
    {"label": "Rotation + charging streaks + speckle", "kind": "dram", "overrides": {
        "search_rotation_deg": 7.0, "charging_streak_prob": 3.0, "charging_streak_intensity": 2.0,
        "speckle_sigma": 0.5,
    }, "challenge": "Bright charging streaks cut across the insulating separator rows, "
                    "multiplicative speckle roughens the whole array, and it's rotated besides."},
    {"label": "Rotation + shear + CD shrink", "kind": "finfet", "overrides": {
        "search_rotation_deg": 6.0, "shear_amplitude_px": 3.0, "linewidth_bias_nm": -10.0,
    }, "challenge": "Fins and gates etch narrower than drawn, the raster shears progressively "
                    "top to bottom, and the frame carries a real stage rotation on top."},
    {"label": "Rotation + detector noise + gamma expand", "kind": "dram", "overrides": {
        "search_rotation_deg": 9.0, "detector_noise_sigma_search": 10.0, "gamma": 0.6,
    }, "challenge": "A blown-out, overexposed capture with heavy sensor noise, rotated near the "
                    "10-degree cap -- low contrast and geometric misalignment together."},
    {"label": "Rotation + starved dose + CD bias grow", "kind": "finfet", "overrides": {
        "search_rotation_deg": 5.0, "dose_search": 60.0, "linewidth_bias_nm": 10.0,
    }, "challenge": "Fins and contacts print systematically oversized, the capture is shot-noise "
                    "starved, and a real stage tilt sits on top of both."},
    {"label": "Rotation + row jitter + salt-pepper", "kind": "dram", "overrides": {
        "search_rotation_deg": 8.0, "drift_jitter_px": 2.5, "salt_pepper_prob": 0.025,
    }, "challenge": "Row-to-row scan jitter and sparse impulse-noise pixels corrupt the array "
                    "independently of a near-maximum whole-frame rotation."},
    {"label": "Max rotation + vignette + speckle", "kind": "finfet", "overrides": {
        "search_rotation_deg": 10.0, "vignette_strength": 0.5, "speckle_sigma": 0.4,
    }, "challenge": "Full rotation cap, edge vignetting, and brightness-scaled speckle together -- "
                    "the true match may sit in the dimmest, noisiest part of the frame."},
    {"label": "Rotation + shear + detector noise", "kind": "dram", "overrides": {
        "search_rotation_deg": 8.0, "shear_amplitude_px": 4.0, "detector_noise_sigma_search": 8.0,
    }, "challenge": "A sheared raster scan and constant-variance sensor noise both distort the "
                    "bitline pattern, with a near-maximum stage rotation layered on top."},
    {"label": "Rotation + row jitter + charging streaks", "kind": "finfet", "overrides": {
        "search_rotation_deg": 7.0, "drift_jitter_px": 2.0, "charging_streak_prob": 2.5,
        "charging_streak_intensity": 1.8,
    }, "challenge": "Row-to-row jitter and bright charging streaks over the spacer regions both "
                    "corrupt the fin/gate pattern, on top of a real stage tilt."},
    {"label": "Rotation + starved dose + astigmatism", "kind": "dram", "overrides": {
        "search_rotation_deg": 9.0, "dose_search": 45.0, "astigmatism_ratio": 1.8,
    }, "challenge": "Low-dose shot noise and directional astigmatic blur degrade the signal along "
                    "with a near-maximum rotation -- three compounding acquisition problems."},
    {"label": "Rotation + shear + CD grow", "kind": "finfet", "overrides": {
        "search_rotation_deg": 9.0, "shear_amplitude_px": 2.5, "linewidth_bias_nm": 15.0,
    }, "challenge": "Systematically oversized fins/contacts on top of a sheared raster scan, "
                    "captured at a near-maximum stage angle."},
    {"label": "Rotation + row jitter + gamma crush", "kind": "dram", "overrides": {
        "search_rotation_deg": 7.0, "drift_jitter_px": 2.2, "gamma": 2.0,
    }, "challenge": "A dark, contrast-crushed capture with unstable row-to-row jitter, plus a "
                    "real whole-frame rotation on top."},
    {"label": "Max rotation + shear + detector noise", "kind": "finfet", "overrides": {
        "search_rotation_deg": 10.0, "shear_amplitude_px": 4.0, "detector_noise_sigma_search": 9.0,
    }, "challenge": "The hardest geometric combination: full rotation cap, strong raster shear, "
                    "and heavy sensor noise all applied to the same frame at once."},
    {"label": "Rotation + starved dose + salt-pepper", "kind": "dram", "overrides": {
        "search_rotation_deg": 6.0, "dose_search": 70.0, "salt_pepper_prob": 0.03,
    }, "challenge": "Sparse impulse-noise pixels and shot-noise grain from a starved dose combine "
                    "with a real stage rotation -- individually survivable, compounding together."},
    {"label": "Max rotation + shear + astigmatism", "kind": "finfet", "overrides": {
        "search_rotation_deg": 10.0, "shear_amplitude_px": 3.0, "astigmatism_ratio": 2.0,
    }, "challenge": "Directional beam blur and progressive raster shear both smear fin/gate "
                    "edges, and the whole capture carries the full 10-degree stage rotation."},
]

MAT_SIZE_NM = 2600.0
COLLAPSE_THRESHOLD_NM = 10.0
MAX_ATTEMPTS_PER_CASE = 12
# A confident-but-wrong ZNCC match above this score/distance combo signals
# the array's periodicity fooled the matcher onto a different, look-alike
# mat -- not the lesson a given case is supposed to teach, so retry with a
# fresh seed instead of keeping a misleading "gotcha" sample.
CONFUSION_SCORE_THRESHOLD = 0.7
CONFUSION_DISTANCE_PX = 20.0


def _default_intensities(num_layers: int) -> dict:
    return {i: yield_model.yield_to_intensity(yield_model.layer_yield(i, num_layers)) for i in range(num_layers)}


def _looks_periodicity_confused(zncc_result: dict, shift: dict) -> bool:
    return zncc_result["score"] > CONFUSION_SCORE_THRESHOLD and shift["distance_px"] > CONFUSION_DISTANCE_PX


def _build_one(case: dict, strip_width_nm: float, seed: int):
    search_kwargs = {**BASELINE_SEARCH, **case["overrides"]}
    params = CadGenerationParams(
        mat_size_nm=MAT_SIZE_NM, strip_width_nm=strip_width_nm, collapse_threshold_nm=COLLAPSE_THRESHOLD_NM,
        no_match_prob=0.0, **search_kwargs,
    )
    reference_render_params = CadGenerationParams(**REFERENCE_BLUR_ONLY)
    rng = np.random.default_rng(seed)
    geom = build_cad_geometry(case["kind"], rng, params, center_bias=True)
    intensities = _default_intensities(geom["num_layers"])
    sample = render_cad_sample(
        geom, params, layer_intensities=intensities, reference_layer_intensities=intensities,
        reference_render_params=reference_render_params,
    )
    zncc_result = zncc_match(sample["reference_preview"], sample["search_img"])
    shift = shift_report(zncc_result["x"], zncc_result["y"], sample["gt_x"], sample["gt_y"])
    return geom, sample, params, reference_render_params, intensities, zncc_result, shift


def _to_bookmark_dict(case: dict, index: int, seed: int, strip_width_nm: float, geom, sample, params, ref_params, intensities, zncc_result, shift) -> dict:
    slug = case["label"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(",", "").replace("-", "_")
    return {
        "id": f"curated_{index + 1:02d}_{slug}",
        "label": case["label"],
        "challenge": case["challenge"],
        "curated": True,
        "kind": case["kind"],
        "seed": seed,
        "manual_center": None,
        # Reconstruction must use the *same* crop-picking code path used at
        # generation time (center_bias=True), not just the same seed -- the
        # two paths consume a different number of RNG draws internally
        # (center_bias skips the "which strip" random draw), which would
        # desynchronize everything drawn afterward (jitter, strip texture,
        # search noise) even with an identical seed.
        "center_bias": True,
        "structure": {
            "mat_size_nm": params.mat_size_nm, "strip_width_nm": strip_width_nm,
            "collapse_threshold_nm": params.collapse_threshold_nm, "no_match_prob": params.no_match_prob,
            "contact_aspect_ratio": params.contact_aspect_ratio,
            "contact_thickness_factor": params.contact_thickness_factor, "contact_angle_deg": params.contact_angle_deg,
        },
        "layer_visibility": {"search_min_layer": 0, "reference_min_layer": 0},
        "fab_distortion": {
            "polygon_scale_prob": params.polygon_scale_prob, "polygon_scale_range": params.polygon_scale_range,
            "linewidth_bias_nm": params.linewidth_bias_nm, "corner_rounding_px": params.corner_rounding_px,
        },
        "sem_acquisition": {
            "dose_search": params.dose_search, "beam_spot_size_nm": params.beam_spot_size_nm,
            "shear_amplitude_px": params.shear_amplitude_px, "drift_jitter_px": params.drift_jitter_px,
            "detector_noise_sigma_search": params.detector_noise_sigma_search,
            "astigmatism_ratio": params.astigmatism_ratio, "search_rotation_deg": params.search_rotation_deg,
            "barrel_distortion_k": params.barrel_distortion_k, "vignette_strength": params.vignette_strength,
            "gamma": params.gamma, "charging_streak_prob": params.charging_streak_prob,
            "charging_streak_intensity": params.charging_streak_intensity,
            "speckle_sigma": params.speckle_sigma, "salt_pepper_prob": params.salt_pepper_prob,
        },
        "reference_layer_intensities": intensities,
        "search_layer_intensities": intensities,
        "reference_ops_enabled": True,
        "reference_fab_distortion": {
            "polygon_scale_prob": ref_params.polygon_scale_prob, "polygon_scale_range": ref_params.polygon_scale_range,
            "linewidth_bias_nm": ref_params.linewidth_bias_nm, "corner_rounding_px": ref_params.corner_rounding_px,
        },
        "reference_sem_acquisition": {
            "dose_search": ref_params.dose_search, "beam_spot_size_nm": ref_params.beam_spot_size_nm,
            "shear_amplitude_px": ref_params.shear_amplitude_px, "drift_jitter_px": ref_params.drift_jitter_px,
            "detector_noise_sigma_search": ref_params.detector_noise_sigma_search,
            "astigmatism_ratio": ref_params.astigmatism_ratio, "search_rotation_deg": ref_params.search_rotation_deg,
            "barrel_distortion_k": ref_params.barrel_distortion_k, "vignette_strength": ref_params.vignette_strength,
            "gamma": ref_params.gamma, "charging_streak_prob": ref_params.charging_streak_prob,
            "charging_streak_intensity": ref_params.charging_streak_intensity,
            "speckle_sigma": ref_params.speckle_sigma, "salt_pepper_prob": ref_params.salt_pepper_prob,
        },
        "zncc": {
            "x": zncc_result["x"], "y": zncc_result["y"], "score": zncc_result["score"], "scale": zncc_result["scale"],
            "template_w": zncc_result["template_w"], "template_h": zncc_result["template_h"],
            "distance_px": shift["distance_px"], "dx": shift["dx"], "dy": shift["dy"],
        },
    }


def generate_curated_bookmarks(base_seed: int = 1000, progress_callback=None) -> list:
    """Build all 20 curated cases (with rejection-sampling retries against
    periodicity-confused ZNCC matches), save them to BOOKMARKS_STORE_PATH,
    and return the list of bookmark dicts (same schema app.py's own
    bookmark snapshot uses, so the existing gallery/zip-export code needs
    no special-casing for these).
    """
    for case in CURATED_CASES:
        rot = case["overrides"].get("search_rotation_deg", 0.0)
        assert rot <= MAX_SEARCH_ROTATION_DEG, f"{case['label']!r} exceeds the {MAX_SEARCH_ROTATION_DEG} deg rotation cap"

    strip_widths = np.linspace(80.0, 200.0, len(CURATED_CASES))
    bookmarks = []
    for i, case in enumerate(CURATED_CASES):
        strip_width_nm = float(strip_widths[i])
        result = None
        for attempt in range(MAX_ATTEMPTS_PER_CASE):
            seed = base_seed + i * 1000 + attempt
            built = _build_one(case, strip_width_nm, seed)
            *_, zncc_result, shift = built
            if not _looks_periodicity_confused(zncc_result, shift):
                result = (seed, built)
                break
        if result is None:
            result = (seed, built)  # exhausted retries -- keep the last attempt rather than drop the case
        seed, (geom, sample, params, ref_params, intensities, zncc_result, shift) = result
        bookmarks.append(_to_bookmark_dict(
            case, i, seed, strip_width_nm, geom, sample, params, ref_params, intensities, zncc_result, shift,
        ))
        if progress_callback:
            progress_callback(i + 1, len(CURATED_CASES), case["label"])

    with open(BOOKMARKS_STORE_PATH, "w") as f:
        json.dump(bookmarks, f, indent=2)
    return bookmarks


def load_curated_bookmarks_from_disk() -> list | None:
    if not BOOKMARKS_STORE_PATH.exists():
        return None
    try:
        with open(BOOKMARKS_STORE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
