"""
Orchestrates one Drift-Sense sample:

  fine canvas (1 nm/px, 10000x10000)
    -> random 1000x1000 crop = Reference Image (native res)
    -> whole-canvas beam blur + 10x downsample + search-specific noise/drift
       = Search Image (1000x1000 @ 10 nm/px)
    -> ground truth = crop location, converted to search-image pixel coords

Physical calibration is fixed by the problem statement: both images are
1000x1000 px; reference is 1 nm/px (1 um FOV), search is 10 nm/px (10 um
FOV). The 10x relationship falls directly out of that pixel-size ratio --
no separate "shrink by 10x" resize step is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np

from src import sem_imaging
from src.presets import get_preset
from src.patterns.dram import generate_dram_canvas
from src.patterns.finfet import generate_finfet_canvas
from src.patterns.zones import generate_zone_canvas

REFERENCE_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1
PIXEL_SIZE_SEARCH_NM = 10
SCALE_FACTOR = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM  # 10
FINE_CANVAS_SIZE_PX = REFERENCE_SIZE_PX * SCALE_FACTOR  # 10000


@dataclass
class GenerationParams:
    beam_spot_size_nm: float = 5.0
    collapse_threshold_nm: float = 10.0
    dose_reference: float = 2000.0
    dose_search: float = 200.0
    shear_amplitude_px: float = 1.5
    drift_jitter_px: float = 0.5
    detector_noise_sigma_ref: float = 2.0
    detector_noise_sigma_search: float = 5.0

    # Astigmatism: beam-spot ellipticity (1.0 = round spot, no effect)
    astigmatism_ratio: float = 1.0
    # Vignetting: radial darkening toward frame edges (0 = none)
    vignette_strength: float = 0.0
    # Nonlinear contrast/gamma response (1.0 = no effect)
    gamma: float = 1.0
    # Barrel(+)/pincushion(-) geometric lens distortion (0 = none)
    barrel_distortion_k: float = 0.0
    # Charging-streak artifacts: expected streaks per 100 rows, and their
    # brightness scale (0 = none)
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    # Multiplicative (speckle-style) noise: out = img * (1 + N(0, sigma))
    speckle_sigma: float = 0.0
    # Impulse (salt-and-pepper) noise: fraction of pixels forced to 0/255
    salt_pepper_prob: float = 0.0

    # Large-scale zone composition: mat (array block) size and separating
    # strip (peripheral/routing material) width, both in nm. Set
    # mat_size_nm >= FINE_CANVAS size to effectively disable zoning (single
    # uniform mat filling the canvas).
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    # Probability that a sample's crop is deliberately biased to straddle a
    # mat/strip boundary rather than sampled uniformly at random.
    boundary_bias: float = 0.35

    # Deterministic global CD/etch bias applied to every drawn line/contact,
    # in nm ("polygon scaling" -- positive grows features, negative shrinks
    # them), on top of each pattern module's own per-instance jitter.
    linewidth_bias_nm: float = 0.0
    # Morphological corner-rounding radius (px) applied to the rendered
    # pattern mask -- real lithography/etch never produces perfectly sharp
    # polygon corners.
    corner_rounding_px: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


_GENERATORS = {
    "dram": generate_dram_canvas,
    "finfet": generate_finfet_canvas,
}


def generate_fine_canvas(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    preset_overrides: dict | None = None,
) -> np.ndarray:
    """Legacy single-mat canvas (one preset filling the whole fine canvas,
    no zone/strip composition). Kept for direct/simple use and tests;
    generate_sample uses generate_fine_canvas_zoned by default.
    """
    preset = get_preset(architecture)
    if preset_overrides:
        preset.update(preset_overrides)
    generator = _GENERATORS[preset["kind"]]
    return generator(
        FINE_CANVAS_SIZE_PX, preset, params.collapse_threshold_nm, rng,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
    )


def generate_fine_canvas_zoned(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
) -> dict:
    preset = get_preset(architecture)
    return generate_zone_canvas(
        FINE_CANVAS_SIZE_PX,
        preset["kind"],
        params.collapse_threshold_nm,
        rng,
        mat_size_nm=params.mat_size_nm,
        strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
    )


def _pick_crop_origin(zone_result: dict, params: GenerationParams, rng: np.random.Generator) -> tuple:
    max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    strip_rects = zone_result.get("strip_rects") or []

    if strip_rects and rng.random() < params.boundary_bias:
        sx, sy, sw, sh = strip_rects[int(rng.integers(0, len(strip_rects)))]
        scx, scy = sx + sw / 2.0, sy + sh / 2.0
        x0 = scx - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        y0 = scy - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        x0 = int(np.clip(x0, 0, max_offset))
        y0 = int(np.clip(y0, 0, max_offset))
        return x0, y0

    x0 = int(rng.integers(0, max_offset + 1))
    y0 = int(rng.integers(0, max_offset + 1))
    return x0, y0


def generate_sample(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    preset_overrides: dict | None = None,
) -> dict:
    if preset_overrides:
        fine_canvas = generate_fine_canvas(architecture, rng, params, preset_overrides)
        zone_result = {"strip_rects": []}
    else:
        zone_result = generate_fine_canvas_zoned(architecture, rng, params)
        fine_canvas = zone_result["canvas"]

    x0, y0 = _pick_crop_origin(zone_result, params, rng)
    crop = fine_canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_reference,
        rng=rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k * 0.3,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
    )

    search_img = sem_imaging.image_search(
        fine_canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
    )

    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR  # 100
    gt_x0 = x0 / SCALE_FACTOR
    gt_y0 = y0 / SCALE_FACTOR
    gt_cx = gt_x0 + box_w / 2.0
    gt_cy = gt_y0 + box_h / 2.0

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_h),
        "architecture": architecture,
        "params": params.as_dict(),
        "mat_rects": zone_result.get("mat_rects", []),
        "strip_rects": zone_result.get("strip_rects", []),
    }


# Named acquisition-condition variants for generate_sample_family(): the same
# physical scene (fixed structure + fixed Reference crop), imaged under
# different conditions -- demonstrates that one Reference is findable across
# a range of degraded Search images, not just the one it happened to be cut
# from. Each entry overrides a subset of GenerationParams fields.
SEARCH_VARIANTS = [
    {"label": "clean", "dose_search": 900.0, "shear_amplitude_px": 0.3, "drift_jitter_px": 0.15},
    {"label": "low_dose", "dose_search": 55.0, "shear_amplitude_px": 1.0, "drift_jitter_px": 0.4},
    {"label": "heavy_drift", "shear_amplitude_px": 4.5, "drift_jitter_px": 2.0},
    {"label": "speckle_salt_pepper", "speckle_sigma": 0.3, "salt_pepper_prob": 0.012},
    {"label": "charging", "charging_streak_prob": 3.5, "charging_streak_intensity": 2.2},
]


def generate_sample_family(
    architecture: str,
    base_seed: int,
    params: GenerationParams,
    variants: list | None = None,
) -> dict:
    """One fixed structure + one fixed Reference crop, rendered as several
    Search images under different acquisition conditions (dose, drift,
    speckle/impulse noise, charging). All variants share the same
    fine canvas, crop location, and ground truth -- only how the Search was
    *imaged* differs, which is the point: the same Reference should still be
    findable across all of them.
    """
    variants = variants if variants is not None else SEARCH_VARIANTS

    struct_rng = np.random.default_rng(base_seed)
    zone_result = generate_fine_canvas_zoned(architecture, struct_rng, params)
    fine_canvas = zone_result["canvas"]
    x0, y0 = _pick_crop_origin(zone_result, params, struct_rng)
    crop = fine_canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

    ref_rng = np.random.default_rng(base_seed * 7919 + 1)
    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_reference,
        rng=ref_rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k * 0.3,
    )

    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR
    gt_x0 = x0 / SCALE_FACTOR
    gt_y0 = y0 / SCALE_FACTOR
    gt_cx = gt_x0 + box_w / 2.0
    gt_cy = gt_y0 + box_h / 2.0

    search_variants = []
    for i, overrides in enumerate(variants):
        label = overrides.get("label", f"variant_{i}")
        field_overrides = {k: v for k, v in overrides.items() if k != "label"}
        variant_params = replace(params, **field_overrides)
        variant_rng = np.random.default_rng(base_seed * 1_000_003 + i + 1)
        search_img = sem_imaging.image_search(
            fine_canvas,
            pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
            pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
            spot_size_nm=variant_params.beam_spot_size_nm,
            dose=variant_params.dose_search,
            rng=variant_rng,
            shear_amplitude_px=variant_params.shear_amplitude_px,
            drift_jitter_px=variant_params.drift_jitter_px,
            detector_noise_sigma=variant_params.detector_noise_sigma_search,
            astigmatism_ratio=variant_params.astigmatism_ratio,
            vignette_strength=variant_params.vignette_strength,
            gamma=variant_params.gamma,
            barrel_distortion_k=variant_params.barrel_distortion_k,
            charging_streak_prob=variant_params.charging_streak_prob,
            charging_streak_intensity=variant_params.charging_streak_intensity,
            speckle_sigma=variant_params.speckle_sigma,
            salt_pepper_prob=variant_params.salt_pepper_prob,
        )
        search_variants.append({
            "label": label,
            "search_img": search_img,
            "params": variant_params.as_dict(),
        })

    return {
        "reference_img": reference_img,
        "search_variants": search_variants,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_h),
        "architecture": architecture,
        "base_seed": base_seed,
        "mat_rects": zone_result.get("mat_rects", []),
        "strip_rects": zone_result.get("strip_rects", []),
    }
