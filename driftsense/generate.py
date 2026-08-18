"""Corrected-ground-truth sample generation for Drift-Sense.

Shared by the deliverable CLI (generate_dataset.py) and the development
tooling (scripts/gen_data.py) so both produce byte-identical data.

The upstream generator (generator/) writes ground truth in pre-imaging
coordinates and then warps the search frame with raster drift and barrel
distortion without updating the label. This module re-runs the same imaging
steps while capturing the realised warp parameters, then maps the label
through them -- see correct_gt().
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

from src import sem_imaging  # noqa: E402
from src.pipeline import (  # noqa: E402
    FINE_CANVAS_SIZE_PX, PIXEL_SIZE_REF_NM, PIXEL_SIZE_SEARCH_NM,
    REFERENCE_SIZE_PX, SCALE_FACTOR, GenerationParams,
    generate_fine_canvas_zoned, _pick_crop_origin,
)
from src.presets import PRESETS  # noqa: E402

SEARCH_SIZE_PX = REFERENCE_SIZE_PX  # 1000; search frame is also 1000x1000
BOX_PX = REFERENCE_SIZE_PX // SCALE_FACTOR  # 100

# Fixed-condition presets. `low`/`medium`/`high`/`severe` mirror the noise
# levels in generator/baseline_solution/evaluate.py so our numbers stay
# comparable to the published ZNCC baseline; `default` is the dataclass
# default (which is what upstream generate_dataset.py emits).
NOISE_PRESETS = {
    "default": {},
    "low": {"dose_search": 800.0, "detector_noise_sigma_search": 2.0,
            "shear_amplitude_px": 0.5, "drift_jitter_px": 0.2},
    "medium": {"dose_search": 200.0, "detector_noise_sigma_search": 5.0,
               "shear_amplitude_px": 1.5, "drift_jitter_px": 0.5},
    "high": {"dose_search": 60.0, "detector_noise_sigma_search": 8.0,
             "shear_amplitude_px": 2.5, "drift_jitter_px": 1.0,
             "speckle_sigma": 0.15},
    "severe": {"dose_search": 25.0, "detector_noise_sigma_search": 12.0,
               "shear_amplitude_px": 4.0, "drift_jitter_px": 1.8,
               "speckle_sigma": 0.3, "salt_pepper_prob": 0.01},
}


def sample_random_params(rng: np.random.Generator) -> dict:
    """Draw acquisition conditions for one sample.

    Ranges span roughly `low` through `severe` above, so a model trained on
    this split sees the whole degradation envelope rather than one operating
    point. Optics terms (astigmatism, vignetting, gamma, barrel, charging)
    are drawn too -- upstream leaves them at their no-op defaults, which
    would let a model overfit to a perfectly-corrected column.
    """
    return {
        "dose_search": float(np.exp(rng.uniform(np.log(25.0), np.log(900.0)))),
        "dose_reference": float(np.exp(rng.uniform(np.log(800.0), np.log(3000.0)))),
        "detector_noise_sigma_search": float(rng.uniform(1.5, 12.0)),
        "detector_noise_sigma_ref": float(rng.uniform(1.0, 4.0)),
        "shear_amplitude_px": float(rng.uniform(0.2, 4.5)),
        "drift_jitter_px": float(rng.uniform(0.1, 2.0)),
        "beam_spot_size_nm": float(rng.uniform(3.5, 8.0)),
        "astigmatism_ratio": float(rng.uniform(1.0, 1.35)),
        "vignette_strength": float(rng.uniform(0.0, 0.35)),
        "gamma": float(rng.uniform(0.75, 1.35)),
        # Kept modest: this is residual scan nonlinearity after calibration,
        # and it also desynchronises reference/search geometry (the reference
        # only receives 0.3x of it), which is a difficulty knob, not noise.
        "barrel_distortion_k": float(rng.uniform(-0.02, 0.02)),
        "charging_streak_prob": float(rng.choice([0.0, 0.0, 1.0, 3.5])),
        "charging_streak_intensity": float(rng.uniform(0.0, 2.2)),
        "speckle_sigma": float(rng.choice([0.0, 0.0, 0.15, 0.3])),
        "salt_pepper_prob": float(rng.choice([0.0, 0.0, 0.004, 0.012])),
        "linewidth_bias_nm": float(rng.uniform(-4.0, 4.0)),
        "corner_rounding_px": float(rng.uniform(0.0, 3.0)),
        "collapse_threshold_nm": float(rng.uniform(8.0, 14.0)),
        "mat_size_nm": float(rng.uniform(1800.0, 4000.0)),
        "strip_width_nm": float(rng.uniform(200.0, 500.0)),
    }


# ---------------------------------------------------------------------------
# Traced search imaging: same steps as sem_imaging.image_search, but the
# geometric warps are applied by us so their exact parameters can be reused
# to place the ground-truth point.
# ---------------------------------------------------------------------------

def image_search_traced(full_canvas: np.ndarray, p: GenerationParams,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    factor = int(round(PIXEL_SIZE_SEARCH_NM / PIXEL_SIZE_REF_NM))
    img = sem_imaging.gaussian_psf_blur(
        full_canvas, p.beam_spot_size_nm, PIXEL_SIZE_REF_NM, p.astigmatism_ratio)
    img = sem_imaging.downsample_area_average(img, factor)

    # --- raster drift, with the realised per-row shift captured ---
    h = img.shape[0]
    rows = np.arange(h)
    shear = p.shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = (rng.normal(0, p.drift_jitter_px, size=h)
              if p.drift_jitter_px > 0 else np.zeros(h))
    row_shift = (shear + jitter).astype(np.float32)
    if p.shear_amplitude_px != 0 or p.drift_jitter_px != 0:
        w = img.shape[1]
        map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    else:
        row_shift = np.zeros(h, dtype=np.float32)

    img = sem_imaging.apply_barrel_distortion(img, p.barrel_distortion_k)

    # Photometric only from here -- no effect on geometry / ground truth.
    img = sem_imaging.add_shot_noise(img, p.dose_search, rng)
    img = sem_imaging.add_detector_noise(img, p.detector_noise_sigma_search, rng)
    img = sem_imaging.add_speckle_noise(img, p.speckle_sigma, rng)
    img = sem_imaging.add_salt_and_pepper_noise(img, p.salt_pepper_prob, rng)
    img = sem_imaging.apply_vignette(img, p.vignette_strength)
    img = sem_imaging.apply_gamma(img, p.gamma)
    img = sem_imaging.add_charging_streaks(
        img, p.charging_streak_prob, p.charging_streak_intensity, rng)
    return img, row_shift, p.barrel_distortion_k


def _invert_barrel(nx: float, ny: float, k: float) -> tuple[float, float]:
    """Invert the normalised radial map r_src = r_dst * (1 + k*r_dst^2).

    apply_barrel_distortion samples the source at map(dst), so a feature
    sitting at source radius r_src is *displayed* at the r_dst solving that
    cubic. Newton from r_dst = r_src converges in a few steps for |k| small.
    """
    if k == 0.0:
        return nx, ny
    r_src = float(np.hypot(nx, ny))
    if r_src < 1e-12:
        return nx, ny
    r = r_src
    for _ in range(50):
        f = r * (1.0 + k * r * r) - r_src
        df = 1.0 + 3.0 * k * r * r
        if abs(df) < 1e-12:
            break
        step = f / df
        r -= step
        if abs(step) < 1e-12:
            break
    scale = r / r_src
    return nx * scale, ny * scale


def correct_gt(px: float, py: float, row_shift: np.ndarray, k: float) -> tuple[float, float]:
    """Map a label point from pre-imaging search coords to where the pattern
    actually lands after drift + barrel.

    Drift: remap reads out(y, x) = in(y, x + row_shift[y]) and leaves rows
    alone, so a feature at input column px is displayed at px - row_shift[py].
    Barrel: invert the radial map (see _invert_barrel).
    """
    row = int(np.clip(round(py), 0, len(row_shift) - 1))
    ax, ay = px - float(row_shift[row]), py

    if k == 0.0:
        return ax, ay
    c = (SEARCH_SIZE_PX - 1) / 2.0
    nb_x, nb_y = _invert_barrel((ax - c) / c, (ay - c) / c, k)
    return nb_x * c + c, nb_y * c + c


def build_one(job: tuple) -> list[dict]:
    """Render one fine canvas -> one search image + `crops` reference crops.

    The canvas costs ~1s (10000^2 px) while an extra reference crop costs
    ~50ms, so crops>1 multiplies the number of training pairs at almost no
    extra cost. Every crop shares the search image but has its own crop
    location, ground truth and reference-side imaging noise. Splits used for
    evaluation should stay at crops=1 so each sample is an independent scene.
    """
    idx, entropy, architectures, noise, out_dirs, crops, store_templates = job
    ref_dir, search_dir = out_dirs

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    architecture = architectures[int(rng.integers(0, len(architectures)))]

    overrides = sample_random_params(rng) if noise == "randomized" else dict(NOISE_PRESETS[noise])
    params = GenerationParams(**overrides)

    zone_result = generate_fine_canvas_zoned(architecture, rng, params)
    fine_canvas = zone_result["canvas"]

    search_img, row_shift, k = image_search_traced(fine_canvas, params, rng)
    search_rel = os.path.join("search", f"{idx:05d}.png")
    cv2.imwrite(os.path.join(search_dir, f"{idx:05d}.png"), search_img)

    rows = []
    for c in range(crops):
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

        gt_x = x0 / SCALE_FACTOR + BOX_PX / 2.0
        gt_y = y0 / SCALE_FACTOR + BOX_PX / 2.0
        gt_x_corr, gt_y_corr = correct_gt(gt_x, gt_y, row_shift, k)

        if store_templates:
            # Training only ever consumes the reference through an exact 10x
            # INTER_AREA downsample to BOX_PX (driftsense.dataset.build_sample,
            # and matching.make_template at inference), and that downsample
            # commutes with the dihedral augmentation applied before it. Storing
            # the result instead of the 1000px original is therefore lossless
            # for training and ~70x smaller: 748 KB -> ~10 KB per pair, which
            # takes a 60k-pair pool at 50 GB to a 300k-pair pool at 35 GB.
            # Eval splits must NOT use this -- evaluate.py and infer.py are
            # specified on full-resolution references.
            reference_img = cv2.resize(reference_img, (BOX_PX, BOX_PX),
                                       interpolation=cv2.INTER_AREA)

        stem = f"{idx:05d}" if crops == 1 else f"{idx:05d}_{c}"
        ref_rel = os.path.join("reference", f"{stem}.png")
        cv2.imwrite(os.path.join(ref_dir, f"{stem}.png"), reference_img)

        row = {
            "id": stem,
            "canvas_id": idx,
            "reference_path": ref_rel,
            "search_path": search_rel,
            "gt_x": gt_x, "gt_y": gt_y,
            "gt_x_corr": gt_x_corr, "gt_y_corr": gt_y_corr,
            "gt_box_x": x0 / SCALE_FACTOR, "gt_box_y": y0 / SCALE_FACTOR,
            "gt_box_w": BOX_PX, "gt_box_h": BOX_PX,
            "crop_x0_fine": x0, "crop_y0_fine": y0,
            "architecture": architecture,
            "noise_profile": noise,
            "reference_px": reference_img.shape[0],
            "sample_entropy": entropy,
            "label_shift_px": float(np.hypot(gt_x_corr - gt_x, gt_y_corr - gt_y)),
        }
        row.update(params.as_dict())
        rows.append(row)
    return rows


# Manifest columns: identifiers/ground truth first, then every
# GenerationParams field so a row fully reproduces its own sample.
BASE_FIELDS = [
    "id", "canvas_id", "reference_path", "search_path",
    "gt_x", "gt_y", "gt_x_corr", "gt_y_corr",
    "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
    "crop_x0_fine", "crop_y0_fine",
    "architecture", "noise_profile", "reference_px", "sample_entropy", "label_shift_px",
]
PARAM_FIELDS = list(GenerationParams().as_dict().keys())




def write_split(split_dir: str, num_canvases: int, seed: int, noise: str,
                architectures: list[str], workers: int = 5,
                crops_per_canvas: int = 1, progress_every: int = 25,
                store_templates: bool = False, start_index: int = 0) -> int:
    """Generate one split. Returns the number of (reference, search) pairs.

    Sample i is seeded from its own SeedSequence child, so a given (seed, i)
    always yields the same sample regardless of how many workers run.

    `store_templates` writes 100x100 templates in place of 1000x1000
    references -- lossless for training, ~7x less disk per pair. Never use it
    for a split that evaluate.py or infer.py will read.

    `start_index` appends to an existing split: canvases are still spawned from
    `seed` so indices [start_index, start_index+num_canvases) are the same
    samples they would have been in one big run, and the manifest is opened for
    append. This is what lets a pool grow while the GPU trains on what is
    already there.
    """
    import csv

    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    # Spawn the whole prefix and keep the tail, so canvas i is the same sample
    # whether it was produced in one run or appended in several.
    children = np.random.SeedSequence(seed).spawn(start_index + num_canvases)
    jobs = [
        (i, int(children[i].generate_state(1, dtype=np.uint64)[0]), architectures,
         noise, (ref_dir, search_dir), crops_per_canvas, store_templates)
        for i in range(start_index, start_index + num_canvases)
    ]

    manifest_path = os.path.join(split_dir, "manifest.csv")
    append = start_index > 0 and os.path.exists(manifest_path)
    pairs = done = 0
    with open(manifest_path, "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS + PARAM_FIELDS, extrasaction="ignore")
        if not append:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rows in ex.map(build_one, jobs, chunksize=4):
                writer.writerows(rows)
                done += 1
                pairs += len(rows)
                if progress_every and (done % progress_every == 0 or done == num_canvases):
                    print(f"  [{done}/{num_canvases} canvases, {pairs} pairs]", flush=True)
                    f.flush()
    return pairs


def make_pairs(entropy: int, architectures: list[str], noise: str,
               crops: int = 8) -> list[dict]:
    """In-memory version of build_one: one canvas -> one search frame and
    `crops` (reference, ground-truth) pairs, returned as arrays.

    Same generation path and same geometry-corrected ground truth as
    build_one, but nothing touches disk. Used by the streaming training
    dataset, where storing the images would cost ~0.7 MB per reference and
    the point is to never reuse a scene.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    architecture = architectures[int(rng.integers(0, len(architectures)))]

    overrides = sample_random_params(rng) if noise == "randomized" else dict(NOISE_PRESETS[noise])
    params = GenerationParams(**overrides)

    zone_result = generate_fine_canvas_zoned(architecture, rng, params)
    fine_canvas = zone_result["canvas"]
    search_img, row_shift, k = image_search_traced(fine_canvas, params, rng)

    out = []
    for _ in range(crops):
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
        gt_x = x0 / SCALE_FACTOR + BOX_PX / 2.0
        gt_y = y0 / SCALE_FACTOR + BOX_PX / 2.0
        gx, gy = correct_gt(gt_x, gt_y, row_shift, k)
        out.append({"reference": reference_img, "search": search_img,
                    "gt_x": gx, "gt_y": gy, "architecture": architecture})
    return out
