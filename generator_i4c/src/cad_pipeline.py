"""
Phase 3: orchestrates one CAD-reference Drift-Sense sample.

  fine canvas (1 nm/px, 10000x10000), composed of CAD-built mats
    -> pick a 1000x1000 crop fully inside one mat
    -> clip+translate that mat's real polygons to the crop window
       = Reference (a .gds file -- ideal design geometry, no SEM noise:
         a CAD file is a design database, not an imaged capture)
    -> whole-canvas beam blur + 10x downsample + search noise/drift
       = Search Image (1000x1000 @ 10 nm/px), exactly as in Phase 1/2
    -> ground truth = crop location, in Search-image pixel coords

The reference render and the search image both come from the SAME
rasterized mat canvas, so "the Reference really is in the Search image at
gt_box" holds by construction, not by re-deriving matching geometry twice.

Task shape change from Phase 1/2: the Reference is no longer an imaged
raster (no rotation, no acquisition noise -- those modeled a capture
process that no longer applies to a design file). Only the Search image is
still a noisy SEM capture.

Geometry (mat layout, crop location, clipped reference polygons) and
rendering (per-layer intensity choice -> raster -> SEM imaging) are
deliberately separate steps (build_cad_geometry / render_cad_sample) so an
interactive tool can let a user inspect the CAD and choose per-layer
intensities *before* paying for a re-render, without re-rolling the random
layout every time a slider moves. generate_cad_sample() below is a
one-shot convenience wrapper over both for scripts that don't need that.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from src import sem_imaging
from src.cad.cad_zones import build_cad_mats, rasterize_zone_canvas
from src.cad.render import rasterize_cell, clip_multi_mat_reference
from src.cad.fab_distortion import apply_fab_distortion
from src.cad import yield_model

REFERENCE_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1
PIXEL_SIZE_SEARCH_NM = 10
SCALE_FACTOR = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM
FINE_CANVAS_SIZE_PX = REFERENCE_SIZE_PX * SCALE_FACTOR

# Search-image rotation is capped at this magnitude (real stage/placement
# error is a few degrees, not tens) -- callers (the app's slider, the
# curated-bookmark generator) should never exceed it.
MAX_SEARCH_ROTATION_DEG = 10.0

# Rotating the fine canvas in place and cropping back to the same size
# leaves flat background-colored triangles in the corners -- an obvious,
# free "this frame is rotated" tell that has nothing to do with the actual
# registration challenge. Instead: rasterize a *larger* padded canvas (real
# device content across the whole padded area, not just fill), rotate that,
# then crop the center FINE_CANVAS_SIZE_PX back out -- every pixel in the
# final frame is then real content, sourced correctly, no matter the angle.
# 1.25 covers up to ~17 degrees (cos(t)+sin(t) <= 1.25) with headroom
# above the enforced 10-degree cap.
_ROTATION_PAD_FACTOR = 1.25
PADDED_CANVAS_SIZE_PX = int(round(FINE_CANVAS_SIZE_PX * _ROTATION_PAD_FACTOR))
_PAD_MARGIN_PX = (PADDED_CANVAS_SIZE_PX - FINE_CANVAS_SIZE_PX) // 2


@dataclass
class CadGenerationParams:
    collapse_threshold_nm: float = 10.0

    # Search-image acquisition (unchanged in spirit from Phase 1/2 -- the
    # Search is still an imaged SEM capture).
    beam_spot_size_nm: float = 5.0
    dose_search: float = 200.0
    shear_amplitude_px: float = 1.5
    drift_jitter_px: float = 0.5
    detector_noise_sigma_search: float = 5.0
    # Beam-spot ellipticity (1.0 = round spot, no effect) -- see
    # sem_imaging.gaussian_psf_blur for the physical rationale.
    astigmatism_ratio: float = 1.0

    # Whole-Search-image rotation (+/- degrees), simulating the die/wafer
    # being placed at a slight angle on the stage. Applied only to the
    # Search capture -- the Reference is a design file, so it never picks up
    # placement error the way an imaged frame can. 0 disables it entirely.
    search_rotation_deg: float = 0.0

    # Lens/scan-style and environmental acquisition artifacts, same family
    # as Phase 1/2's -- see sem_imaging.image_search for the physical
    # rationale of each.
    barrel_distortion_k: float = 0.0
    vignette_strength: float = 0.0
    gamma: float = 1.0
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    speckle_sigma: float = 0.0
    salt_pepper_prob: float = 0.0

    # Large-scale zone composition (same idea as Phase 1).
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0

    # Structural (not fabrication distortion): how elongated/thick the
    # tablet/capsule-shaped contacts and vias are, and at what angle -- a
    # real design choice, so it shapes design_cell and fab_cell alike.
    contact_aspect_ratio: float = 1.6
    contact_thickness_factor: float = 1.0
    contact_angle_deg: float = 90.0

    # Fabrication/imaging distortion (see src/cad/fab_distortion.py) --
    # per-polygon size outliers, a deterministic global CD/etch bias
    # (positive grows, negative shrinks), and vector-level corner rounding
    # via gdstk's fillet(). These are effects of turning a design into a
    # real, imaged device, so they only ever shape the Search-side render
    # (`fab_cell` in each mat) -- the Reference GDS always stays the exact,
    # undistorted design (`design_cell`), same as a real design database.
    polygon_scale_prob: float = 0.10
    polygon_scale_range: float = 0.10
    linewidth_bias_nm: float = 0.0
    corner_rounding_px: float = 0.0

    # Probability the Reference CAD comes from a different, independently
    # generated canvas/mat -- a genuine "no match" case.
    no_match_prob: float = 0.08

    def as_dict(self) -> dict:
        return asdict(self)


def _pick_boundary_crop_origin(strip_rects: list, rng: np.random.Generator, center_bias: bool = False) -> tuple:
    """Pick a 1000x1000 crop window that always straddles a mat/strip
    boundary -- every chosen Reference CAD must include a separator, not
    just be entirely one mat's device geometry. Centers on a chosen strip
    region (with jitter), matching Phase 1's boundary-bias math but
    unconditional here rather than probabilistic.

    `center_bias=True` picks the strip nearest the canvas center instead of
    a uniformly random one -- still straddles a real boundary (so the strip
    width still matters and shows up), but keeps the ground truth away from
    the canvas edges. Useful for curated/demo samples where you want the
    match to land somewhere easy to eyeball in a small thumbnail, and where
    "the right one" should read as the central, not peripheral, candidate
    when the array's periodicity throws up look-alikes elsewhere.
    """
    if not strip_rects:
        raise ValueError("no strip regions to straddle -- check mat_size_nm/strip_width_nm")
    max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    if center_bias:
        canvas_center = FINE_CANVAS_SIZE_PX / 2.0
        sx, sy, sw, sh = min(
            strip_rects,
            key=lambda r: (r[0] + r[2] / 2.0 - canvas_center) ** 2 + (r[1] + r[3] / 2.0 - canvas_center) ** 2,
        )
    else:
        sx, sy, sw, sh = strip_rects[int(rng.integers(0, len(strip_rects)))]
    scx, scy = sx + sw / 2.0, sy + sh / 2.0
    x0 = scx - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
    y0 = scy - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
    x0 = int(np.clip(x0, 0, max_offset))
    y0 = int(np.clip(y0, 0, max_offset))
    return x0, y0


def build_cad_geometry(
    architecture_kind: str, rng: np.random.Generator, params: CadGenerationParams,
    manual_center_nm: tuple | None = None, center_bias: bool = False,
) -> dict:
    """Everything that's fixed once a seed is picked: mat layout, which mat
    the reference comes from, its clipped polygons. No intensities chosen
    yet, no raster produced yet -- render_cad_sample() does that part,
    cheaply and repeatably, from this.

    `manual_center_nm`, if given, is an (x, y) point in full-canvas nm
    coordinates -- e.g. picked by a user directly on the Search-side CAD
    viewer -- and the crop window is centered there instead of the
    automatic boundary-straddling pick. A user-chosen point is always
    somewhere in the real canvas, so `no_match_prob` is skipped entirely in
    this mode: manual selection means "give me the reference that's really
    here", not a random draw that might come from an unrelated canvas.
    `center_bias` (ignored when `manual_center_nm` is given) picks the
    boundary strip nearest canvas center instead of a random one -- see
    `_pick_boundary_crop_origin`.
    """
    geometry = build_cad_mats(
        FINE_CANVAS_SIZE_PX, architecture_kind, params.collapse_threshold_nm, rng,
        mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
        polygon_scale_prob=params.polygon_scale_prob, polygon_scale_range=params.polygon_scale_range,
        linewidth_bias_nm=params.linewidth_bias_nm, corner_rounding_px=params.corner_rounding_px,
        contact_aspect_ratio=params.contact_aspect_ratio,
        contact_thickness_factor=params.contact_thickness_factor, contact_angle_deg=params.contact_angle_deg,
    )
    mats, strip_rects, num_layers = geometry["mats"], geometry["strip_rects"], geometry["num_layers"]

    if manual_center_nm is not None:
        max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
        cx, cy = manual_center_nm
        x0 = int(np.clip(cx - REFERENCE_SIZE_PX / 2.0, 0, max_offset))
        y0 = int(np.clip(cy - REFERENCE_SIZE_PX / 2.0, 0, max_offset))
        match_found = True
    else:
        x0, y0 = _pick_boundary_crop_origin(strip_rects, rng, center_bias=center_bias)
        match_found = not (params.no_match_prob > 0 and rng.random() < params.no_match_prob)

    if match_found:
        ref_mats, ref_x0, ref_y0 = mats, x0, y0
        neg_mats, neg_strip_rects = None, None
    else:
        # Reference comes from an independently-generated canvas -- it does
        # not appear anywhere in this sample's Search image.
        neg_geometry = build_cad_mats(
            FINE_CANVAS_SIZE_PX, architecture_kind, params.collapse_threshold_nm, rng,
            mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
            polygon_scale_prob=params.polygon_scale_prob, polygon_scale_range=params.polygon_scale_range,
            contact_aspect_ratio=params.contact_aspect_ratio,
        contact_thickness_factor=params.contact_thickness_factor, contact_angle_deg=params.contact_angle_deg,
        )
        neg_mats, neg_strip_rects = neg_geometry["mats"], neg_geometry["strip_rects"]
        ref_mats = neg_mats
        ref_x0, ref_y0 = _pick_boundary_crop_origin(neg_strip_rects, rng)

    reference_cell = clip_multi_mat_reference(ref_mats, ref_x0, ref_y0, REFERENCE_SIZE_PX, num_layers)

    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR
    if match_found:
        gt_x0, gt_y0 = x0 / SCALE_FACTOR, y0 / SCALE_FACTOR
        gt_box = (gt_x0, gt_y0, box_w, box_h)
        gt_cx, gt_cy = gt_x0 + box_w / 2.0, gt_y0 + box_h / 2.0
    else:
        gt_box = None
        gt_cx = gt_cy = None

    # A fresh rng, seeded from this build, so the strip routing texture is
    # reproducible across repeated render_cad_sample() calls on this same
    # geometry (strip texture doesn't depend on layer_intensities at all,
    # only on this draw).
    strip_rng_seed = int(rng.integers(0, 2**31 - 1))

    return {
        "architecture_kind": architecture_kind,
        "mats": mats,
        "strip_rects": strip_rects,
        "strip_rng_seed": strip_rng_seed,
        "match_found": match_found,
        "x0": x0, "y0": y0,
        "reference_cell": reference_cell,
        "num_layers": num_layers,
        "gt_x": gt_cx, "gt_y": gt_cy, "gt_box": gt_box,
        # only set when match_found is False, since then the Reference's
        # mats are unrelated to the Search and need their own viewer data
        "neg_mats": neg_mats, "neg_strip_rects": neg_strip_rects,
    }


def render_cad_sample(
    geometry: dict, params: CadGenerationParams,
    layer_intensities: dict | None = None, reference_layer_intensities: dict | None = None,
    reference_render_params: CadGenerationParams | None = None,
    search_min_layer: int = 0, reference_min_layer: int = 0,
) -> dict:
    """Render a sample from previously-built geometry: rasterize mats with
    the given (or default, yield-model-derived) per-layer intensities, run
    the Search image through the same SEM imaging as Phase 1/2, and
    rasterize the clipped reference. Cheap (~15-20ms) -- safe to call on
    every slider change.

    `layer_intensities` drives the Search-side render; `reference_layer_intensities`
    drives the Reference preview and defaults to `layer_intensities` when not
    given -- letting a caller intentionally choose *different* per-layer
    brightness for the two, e.g. to simulate a calibration mismatch between
    the design tool and the imaging tool, rather than assuming both always
    render identically just because they show the same physical structure.

    `reference_render_params`, if given, turns the Reference from a plain
    intensity-only raster into its *own* independently-imaged/distorted
    capture -- its own fabrication distortion (polygon_scale_prob/range,
    linewidth_bias_nm, corner_rounding_px) and its own acquisition pipeline
    (sem_imaging.image_reference, using this params object's dose/spot
    size/noise/rotation fields), completely independent from the Search
    side's `params`. Left as None (the default), the Reference stays the
    exact, undistorted design -- only `reference_layer_intensities` affects
    it, matching the original Phase 3 design-database semantics.

    `search_min_layer`/`reference_min_layer` exclude layers below that
    index from each side's render -- a close-FOV Reference can resolve
    deeper/more-buried layers than a coarser, wide-FOV Search capture, so
    these default independently rather than sharing one setting.
    """
    if reference_layer_intensities is None:
        reference_layer_intensities = layer_intensities

    # The Search always reflects the *true* scene (geometry["mats"]),
    # regardless of match_found -- only the Reference switches to unrelated
    # geometry (geometry["ref_mat"], possibly drawn from neg_mats) in the
    # no-match case. Search never renders from neg_mats.
    fine_canvas = rasterize_zone_canvas(
        FINE_CANVAS_SIZE_PX, geometry["mats"], geometry["strip_rects"],
        np.random.default_rng(geometry["strip_rng_seed"]), layer_intensities, min_layer=search_min_layer,
    )

    if reference_render_params is None:
        reference_preview = rasterize_cell(
            geometry["reference_cell"], REFERENCE_SIZE_PX, geometry["num_layers"],
            layer_intensities=reference_layer_intensities, min_layer=reference_min_layer,
        )
    else:
        ref_rng = np.random.default_rng(geometry["strip_rng_seed"] + 3)  # independent stream
        rp = reference_render_params
        ref_cell = geometry["reference_cell"]
        if (rp.polygon_scale_prob > 0 and rp.polygon_scale_range > 0) or abs(rp.linewidth_bias_nm) >= 1e-9 or rp.corner_rounding_px >= 0.5:
            ref_cell = apply_fab_distortion(
                ref_cell, geometry["num_layers"], ref_rng,
                polygon_scale_prob=rp.polygon_scale_prob, polygon_scale_range=rp.polygon_scale_range,
                linewidth_bias_nm=rp.linewidth_bias_nm, corner_rounding_px=rp.corner_rounding_px,
            )
        reference_raster = rasterize_cell(
            ref_cell, REFERENCE_SIZE_PX, geometry["num_layers"], layer_intensities=reference_layer_intensities,
            min_layer=reference_min_layer,
        )
        reference_preview = sem_imaging.image_reference(
            reference_raster, pixel_size_nm=PIXEL_SIZE_REF_NM,
            spot_size_nm=rp.beam_spot_size_nm, dose=rp.dose_search, rng=ref_rng,
            detector_noise_sigma=rp.detector_noise_sigma_search, drift_jitter_px=rp.drift_jitter_px,
            shear_amplitude_px=rp.shear_amplitude_px, astigmatism_ratio=rp.astigmatism_ratio,
            vignette_strength=rp.vignette_strength, gamma=rp.gamma, barrel_distortion_k=rp.barrel_distortion_k,
            charging_streak_prob=rp.charging_streak_prob, charging_streak_intensity=rp.charging_streak_intensity,
            speckle_sigma=rp.speckle_sigma, salt_pepper_prob=rp.salt_pepper_prob,
        )
        if rp.search_rotation_deg > 0:
            angle = ref_rng.uniform(-rp.search_rotation_deg, rp.search_rotation_deg)
            center = (REFERENCE_SIZE_PX / 2.0, REFERENCE_SIZE_PX / 2.0)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            reference_preview = cv2.warpAffine(
                reference_preview, rot_matrix, (REFERENCE_SIZE_PX, REFERENCE_SIZE_PX),
                borderValue=float(yield_model.background_intensity()),
            )

    # Whole-frame rotation, simulating the die/wafer sitting at a slight
    # angle on the stage -- physically, only an *imaged* frame can pick this
    # up, so it's applied to the fine canvas that feeds the Search render,
    # never to the Reference (a design file has no "placement" to be off
    # by). Rotating here, before the crop's ground-truth point is converted
    # to Search-pixel coordinates, keeps "the Reference really is in the
    # Search image at gt_box" true by construction: gt_x/gt_y are derived
    # from the *same* rotation applied to the *same* canvas.
    gt_x, gt_y, gt_box = geometry["gt_x"], geometry["gt_y"], geometry["gt_box"]
    if params.search_rotation_deg > 0:
        rot_rng = np.random.default_rng(geometry["strip_rng_seed"] + 2)  # independent stream
        angle = rot_rng.uniform(-params.search_rotation_deg, params.search_rotation_deg)

        # Build a larger, independently-seeded canvas with the SAME
        # structural params (mat size, strip width, contact shape, ...) --
        # not to vary anything, only to give the rotation real device
        # content to source the crop's corners from. Paste the real,
        # already-rasterized fine_canvas into its center before rotating,
        # so every pixel that lands inside the original footprint is still
        # exactly the true scene; only the thin margin rotation can expose
        # slivers of at the crop edges comes from this separately-drawn
        # stand-in geometry (unavoidable -- there's no "real" content beyond
        # the canvas that was actually built -- but it uses the identical
        # process parameters, so it is not visually distinguishable filler).
        pad_rng = np.random.default_rng(geometry["strip_rng_seed"] + 4)
        padded_geometry = build_cad_mats(
            PADDED_CANVAS_SIZE_PX, geometry["architecture_kind"], params.collapse_threshold_nm, pad_rng,
            mat_size_nm=params.mat_size_nm, strip_width_nm=params.strip_width_nm,
            polygon_scale_prob=params.polygon_scale_prob, polygon_scale_range=params.polygon_scale_range,
            linewidth_bias_nm=params.linewidth_bias_nm, corner_rounding_px=params.corner_rounding_px,
            contact_aspect_ratio=params.contact_aspect_ratio,
            contact_thickness_factor=params.contact_thickness_factor, contact_angle_deg=params.contact_angle_deg,
        )
        padded_canvas = rasterize_zone_canvas(
            PADDED_CANVAS_SIZE_PX, padded_geometry["mats"], padded_geometry["strip_rects"],
            np.random.default_rng(geometry["strip_rng_seed"] + 4), layer_intensities, min_layer=search_min_layer,
        )
        m = _PAD_MARGIN_PX
        padded_canvas[m:m + FINE_CANVAS_SIZE_PX, m:m + FINE_CANVAS_SIZE_PX] = fine_canvas

        center = (PADDED_CANVAS_SIZE_PX / 2.0, PADDED_CANVAS_SIZE_PX / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_padded = cv2.warpAffine(
            padded_canvas, rot_matrix, (PADDED_CANVAS_SIZE_PX, PADDED_CANVAS_SIZE_PX),
            borderValue=float(yield_model.background_intensity()),
        )
        # _ROTATION_PAD_FACTOR keeps this crop's corners inside the rotated
        # square (not the rotation's own border fill) for any angle up to
        # ~17 degrees -- comfortably above MAX_SEARCH_ROTATION_DEG (10).
        fine_canvas = rotated_padded[m:m + FINE_CANVAS_SIZE_PX, m:m + FINE_CANVAS_SIZE_PX]

        if geometry["match_found"]:
            # gt point shifted into padded-canvas coordinates, rotated about
            # the PADDED canvas's center, then shifted back by the same
            # margin -- i.e. expressed relative to the final CROPPED frame's
            # own origin, not the original pre-rotation canvas.
            fx = geometry["x0"] + REFERENCE_SIZE_PX / 2.0 + m
            fy = geometry["y0"] + REFERENCE_SIZE_PX / 2.0 + m
            rx = rot_matrix[0, 0] * fx + rot_matrix[0, 1] * fy + rot_matrix[0, 2] - m
            ry = rot_matrix[1, 0] * fx + rot_matrix[1, 1] * fy + rot_matrix[1, 2] - m
            gt_x, gt_y = rx / SCALE_FACTOR, ry / SCALE_FACTOR
            # The true rotated footprint is no longer axis-aligned; gt_box
            # stays an axis-aligned approximation of the same size, centered
            # on the rotated point -- fine for a location-tolerance check,
            # not a pixel-exact mask.
            box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR
            gt_box = (gt_x - box_w / 2.0, gt_y - box_h / 2.0, box_w, box_h)

    search_rng = np.random.default_rng(geometry["strip_rng_seed"] + 1)  # independent stream from strip texture
    search_img = sem_imaging.image_search(
        fine_canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=search_rng,
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

    return {
        "reference_cell": geometry["reference_cell"],
        "reference_preview": reference_preview,
        "search_img": search_img,
        "gt_x": gt_x, "gt_y": gt_y, "gt_box": gt_box,
        "match_found": geometry["match_found"],
        "architecture_kind": geometry["architecture_kind"],
        "num_layers": geometry["num_layers"],
        "params": params.as_dict(),
    }


def generate_cad_sample(architecture_kind: str, rng: np.random.Generator, params: CadGenerationParams) -> dict:
    """One-shot convenience wrapper (build geometry + render immediately)
    for scripts that don't need the split -- generate_cad_dataset.py, tests."""
    geometry = build_cad_geometry(architecture_kind, rng, params)
    return render_cad_sample(geometry, params)
