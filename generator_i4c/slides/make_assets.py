"""
One-off generator for slide deck illustration assets. Not part of the
teaching repo's core pipeline -- pulls real output from src/ so every image
in the deck is an authentic sample, not a mockup.

Run from repo root: python slides/make_assets.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from PIL import Image

from src.pipeline import (
    GenerationParams, generate_sample, generate_fine_canvas,
    PIXEL_SIZE_REF_NM, PIXEL_SIZE_SEARCH_NM, SCALE_FACTOR,
)
from src.structural_defects import maybe_collapse_gap
from src.presets import DRAM_PRESET_NAMES, FINFET_PRESET_NAMES
from src import sem_imaging

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)


def save(img, name):
    Image.fromarray(img).save(os.path.join(OUT, name))
    print("wrote", name, img.shape)


def upscale(img, factor):
    return cv2.resize(img, (img.shape[1] * factor, img.shape[0] * factor), interpolation=cv2.INTER_NEAREST)


def draw_gt_box(search_img, gt_box, color=(255, 90, 90), thickness=2):
    bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    x0, y0, w, h = (int(round(v)) for v in gt_box)
    cv2.rectangle(bgr, (x0, y0), (x0 + w, y0 + h), color, thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_mat_boundaries(search_img_rgb, mat_rects, color=(90, 200, 255)):
    """mat_rects are in fine-canvas (1 nm/px) coords; search image is 10x
    smaller (10 nm/px) -- scale down before drawing."""
    out = search_img_rgb.copy()
    for (x0, y0, w, h) in mat_rects:
        p0 = (int(round(x0 / SCALE_FACTOR)), int(round(y0 / SCALE_FACTOR)))
        p1 = (int(round((x0 + w) / SCALE_FACTOR)), int(round((y0 + h) / SCALE_FACTOR)))
        cv2.rectangle(out, p0, p1, color, 1)
    return out


# ---------------------------------------------------------------------------
# 1. Hero reference/search pairs (default realistic noise, default zoned
#    multi-mat canvas) -- one DRAM, one FinFET. Also a version with mat
#    boundaries + GT box overlaid, to show the crop sitting on an edge.
# ---------------------------------------------------------------------------
for arch, seed in [("dram_1x", 7), ("finfet_10nm", 11)]:
    rng = np.random.default_rng(seed)
    params = GenerationParams(boundary_bias=1.0)  # force an edge-straddling crop for the hero shot
    sample = generate_sample(arch, rng, params)
    save(sample["reference_img"], f"{arch}_reference.png")
    save(sample["search_img"], f"{arch}_search.png")
    with_gt = draw_gt_box(sample["search_img"], sample["gt_box"])
    save(with_gt, f"{arch}_search_gt.png")
    with_mats = draw_mat_boundaries(with_gt, sample["mat_rects"])
    save(with_mats, f"{arch}_search_annotated.png")

# ---------------------------------------------------------------------------
# 2. Galleries: 8 search images per architecture family, cycling through all
#    preset variants within that family, each its own multi-mat/strip die.
# ---------------------------------------------------------------------------
for family, names, base_seed in [("dram", DRAM_PRESET_NAMES, 1000), ("finfet", FINFET_PRESET_NAMES, 2000)]:
    for i in range(8):
        preset_name = names[i % len(names)]
        rng = np.random.default_rng(base_seed + i)
        params = GenerationParams(boundary_bias=0.35)
        sample = generate_sample(preset_name, rng, params)
        save(sample["search_img"], f"gallery_{family}_{i:02d}_{preset_name}.png")
        save(sample["reference_img"], f"gallery_{family}_{i:02d}_{preset_name}_ref.png")

# ---------------------------------------------------------------------------
# 3. Aliasing close-up: pick the densest preset (smallest pitch relative to
#    search resolution) so the aliasing effect reads clearly.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
params = GenerationParams()
sample = generate_sample("dram_dense", rng, params)
x0, y0, w, h = (int(round(v)) for v in sample["gt_box"])
sub_w_ref, sub_h_ref = 220, 220
ref = sample["reference_img"]
cy, cx = ref.shape[0] // 2, ref.shape[1] // 2
ref_patch = ref[cy - sub_h_ref // 2: cy + sub_h_ref // 2, cx - sub_w_ref // 2: cx + sub_w_ref // 2]
save(upscale(ref_patch, 3), "aliasing_reference_patch.png")

search_patch_w, search_patch_h = sub_w_ref // SCALE_FACTOR, sub_h_ref // SCALE_FACTOR
search = sample["search_img"]
sx0 = x0 + (w - search_patch_w) // 2
sy0 = y0 + (h - search_patch_h) // 2
search_patch = search[sy0:sy0 + search_patch_h, sx0:sx0 + search_patch_w]
save(upscale(search_patch, 30), "aliasing_search_patch.png")

# ---------------------------------------------------------------------------
# 4. Noise comparison: same crop, low dose (default) vs near-noiseless
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
fine = generate_fine_canvas("dram_1x", rng, GenerationParams())
crop = fine[4000:5000, 4000:5000]

rng_a = np.random.default_rng(100)
noisy = sem_imaging.image_reference(
    crop, pixel_size_nm=PIXEL_SIZE_REF_NM, spot_size_nm=5.0, dose=250.0,
    rng=rng_a, detector_noise_sigma=4.0, drift_jitter_px=0.1,
)
rng_b = np.random.default_rng(100)
clean = sem_imaging.image_reference(
    crop, pixel_size_nm=PIXEL_SIZE_REF_NM, spot_size_nm=5.0, dose=1e6,
    rng=rng_b, detector_noise_sigma=0.0, drift_jitter_px=0.0,
)
save(clean, "noise_clean.png")
save(noisy, "noise_noisy.png")

# ---------------------------------------------------------------------------
# 5. Drift comparison: shear applied to a high-contrast reference-res crop
#    so the effect is clearly visible.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(5)
fine2 = generate_fine_canvas("dram_1x", rng, GenerationParams())
drift_crop = fine2[4000:4700, 4000:4700]
drift_blurred = sem_imaging.gaussian_psf_blur(drift_crop, 5.0, PIXEL_SIZE_REF_NM)

rng_c = np.random.default_rng(9)
no_drift = sem_imaging.apply_raster_drift(drift_blurred, shear_amplitude_px=0, jitter_std_px=0, rng=rng_c)
rng_d = np.random.default_rng(9)
with_drift = sem_imaging.apply_raster_drift(drift_blurred, shear_amplitude_px=35, jitter_std_px=3, rng=rng_d)
save(no_drift, "drift_none.png")
save(with_drift, "drift_yes.png")

# ---------------------------------------------------------------------------
# 6. New distortion knobs: astigmatism, vignette+gamma, barrel distortion,
#    charging streaks -- each as a clean/affected pair on the same crop.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(21)
fine3 = generate_fine_canvas("finfet_10nm", rng, GenerationParams())
distort_crop = fine3[3000:3700, 3000:3700]

base = sem_imaging.gaussian_psf_blur(distort_crop, 5.0, PIXEL_SIZE_REF_NM, astigmatism_ratio=1.0)
save(base, "distort_base.png")

astig = sem_imaging.gaussian_psf_blur(distort_crop, 5.0, PIXEL_SIZE_REF_NM, astigmatism_ratio=2.4)
save(astig, "distort_astigmatism.png")

vign_gamma = sem_imaging.apply_gamma(sem_imaging.apply_vignette(base, 0.85), 1.8)
save(vign_gamma, "distort_vignette_gamma.png")

barrel = sem_imaging.apply_barrel_distortion(base, 0.12)
save(barrel, "distort_barrel.png")

rng_charge = np.random.default_rng(55)
charged = sem_imaging.add_charging_streaks(base, streak_prob=3.0, intensity=2.0, rng=rng_charge)
save(charged, "distort_charging.png")

# ---------------------------------------------------------------------------
# 7. Polygon scaling: linewidth bias + corner rounding, direct pattern calls
# ---------------------------------------------------------------------------
from src.patterns.dram import generate_dram_canvas
from src.presets import get_preset

preset = get_preset("dram_1x")
side = 400
rng_p1 = np.random.default_rng(30)
nominal = generate_dram_canvas(side, preset, 10.0, rng_p1)
save(nominal, "polygon_nominal.png")

rng_p2 = np.random.default_rng(30)
grown = generate_dram_canvas(side, preset, 10.0, rng_p2, linewidth_bias_nm=14.0)
save(grown, "polygon_grown.png")

rng_p3 = np.random.default_rng(30)
shrunk = generate_dram_canvas(side, preset, 10.0, rng_p3, linewidth_bias_nm=-14.0)
save(shrunk, "polygon_shrunk.png")

rng_p4 = np.random.default_rng(30)
rounded = generate_dram_canvas(side, preset, 10.0, rng_p4, corner_rounding_px=6.0)
save(rounded, "polygon_rounded.png")

# ---------------------------------------------------------------------------
# 8. Collapse demo data: real calls to maybe_collapse_gap for a spread of
#    gaps, deterministic seed, so the deck shows genuine function output.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
gaps = [20, 16, 12, 10, 8, 6, 4, 2]
results = [(g, maybe_collapse_gap(g, threshold_nm=10.0, rng=rng)) for g in gaps]
print("collapse demo results:", results)
with open(os.path.join(OUT, "collapse_results.txt"), "w") as f:
    for g, r in results:
        f.write(f"{g},{r}\n")

# ---------------------------------------------------------------------------
# 9. QR code linking to the live HF Space (requires: pip install qrcode)
# ---------------------------------------------------------------------------
try:
    import qrcode
    qr_img = qrcode.make("https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data")
    qr_img.resize((300, 300)).save(os.path.join(OUT, "hf_space_qr.png"))
    print("wrote hf_space_qr.png")
except ImportError:
    print("skipped hf_space_qr.png (pip install qrcode to regenerate)")

print("done")
