"""
SEM acquisition artifacts -- applied per-image, since the reference and
search captures happen under different conditions (careful/slow vs.
fast/wide-area). This is deliberately kept separate from
structural_defects.py, which models a property of the physical device
rather than of how it was imaged.

There is a single physical beam (`beam_spot_size_nm`), applied identically
to both images as a Gaussian PSF blur *before* any downsampling. The search
image's extra softness on dense structures comes naturally from the 10x
area-average downsample on top of that shared blur -- not from a separate
"search-only blur" fudge factor.
"""

import cv2
import numpy as np


def gaussian_psf_blur(
    img: np.ndarray,
    spot_size_nm: float,
    pixel_size_nm: float,
    astigmatism_ratio: float = 1.0,
) -> np.ndarray:
    """Gaussian beam-spot blur. `astigmatism_ratio` != 1.0 makes the spot
    elliptical (sigmaY = sigmaX * ratio) -- a real, common SEM aberration
    where the beam isn't perfectly round, which shows up as directional
    blurring (sharper along one scan axis than the other).
    """
    sigma_x = max(spot_size_nm / pixel_size_nm, 1e-6)
    sigma_y = max(sigma_x * astigmatism_ratio, 1e-6)
    k = int(2 * round(3 * max(sigma_x, sigma_y)) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma_x, sigmaY=sigma_y)


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial darkening toward the frame edges, from off-axis beam/detector
    collection efficiency falloff. `strength` in [0, 1]; 0 = no effect.
    """
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    falloff = 1.0 - strength * (r ** 2)
    out = img.astype(np.float64) * falloff
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Nonlinear contrast/brightness response curve (detector gain nonlinearity
    or contrast/brightness knob mis-calibration). gamma=1.0 is a no-op.
    """
    if gamma == 1.0:
        return img
    norm = img.astype(np.float64) / 255.0
    out = np.power(np.clip(norm, 0, 1), gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Radial lens-style distortion (barrel if k>0, pincushion if k<0) from
    imperfect beam-scan linearity/calibration. k=0.0 is a no-op.
    """
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + k * r2
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_charging_streaks(
    img: np.ndarray,
    streak_prob: float,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Occasional bright horizontal streaks from local sample charging
    (common on insulating/oxide regions under e-beam). `streak_prob` is the
    expected streaks per 100 rows; `intensity` scales streak brightness.
    """
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape
    out = img.astype(np.float64)
    expected = streak_prob * (h / 100.0)
    n_streaks = rng.poisson(max(expected, 0))
    for _ in range(n_streaks):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.uint8)


def downsample_area_average(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def apply_raster_drift(
    img: np.ndarray,
    shear_amplitude_px: float,
    jitter_std_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Progressive row-to-row shear (drift accumulating over scan time) plus
    per-row jitter (vibration), mimicking real raster-scan drift artifacts.
    """
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    h, w = img.shape
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std_px, size=h) if jitter_std_px > 0 else np.zeros(h)
    row_shift = (shear + jitter).astype(np.float32)

    map_x = (np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None])
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_shot_noise(img: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise. `dose` is a proxy for electron count/dwell time --
    higher dose (slower/careful scan) means less relative noise.
    """
    img_f = img.astype(np.float64)
    counts = np.clip(img_f / 255.0 * dose, 0, None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    noisy = noisy_counts / dose * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_detector_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img
    noisy = img.astype(np.float64) + rng.normal(0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_speckle_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative noise: out = img * (1 + N(0, sigma)). Distinct from
    the additive Gaussian detector noise above -- a stand-in for detector
    gain variation / coherent-interference-style artifacts, where noise
    magnitude scales with signal brightness rather than being constant.
    """
    if sigma <= 0:
        return img
    img_f = img.astype(np.float64)
    noise = rng.normal(0, sigma, size=img.shape)
    out = img_f * (1.0 + noise)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Impulse noise: a fraction `prob` of pixels are forced to 0 or 255 --
    a stand-in for dead/hot detector pixels or sudden discharge events,
    structurally different from the smooth noise models above.
    """
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def image_reference(
    crop: np.ndarray,
    pixel_size_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    detector_noise_sigma: float = 2.0,
    drift_jitter_px: float = 0.2,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    gamma: float = 1.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
) -> np.ndarray:
    img = gaussian_psf_blur(crop, spot_size_nm, pixel_size_nm, astigmatism_ratio)
    img = apply_raster_drift(img, shear_amplitude_px=0.0, jitter_std_px=drift_jitter_px, rng=rng)
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    return img


def image_search(
    full_canvas: np.ndarray,
    pixel_size_ref_nm: float,
    pixel_size_search_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    shear_amplitude_px: float = 1.5,
    drift_jitter_px: float = 0.5,
    detector_noise_sigma: float = 5.0,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    gamma: float = 1.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
) -> np.ndarray:
    factor = int(round(pixel_size_search_nm / pixel_size_ref_nm))
    blurred = gaussian_psf_blur(full_canvas, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio)
    downsampled = downsample_area_average(blurred, factor)
    drifted = apply_raster_drift(downsampled, shear_amplitude_px, drift_jitter_px, rng)
    distorted = apply_barrel_distortion(drifted, barrel_distortion_k)
    noisy = add_shot_noise(distorted, dose, rng)
    noisy = add_detector_noise(noisy, detector_noise_sigma, rng)
    noisy = add_speckle_noise(noisy, speckle_sigma, rng)
    noisy = add_salt_and_pepper_noise(noisy, salt_pepper_prob, rng)
    noisy = apply_vignette(noisy, vignette_strength)
    noisy = apply_gamma(noisy, gamma)
    noisy = add_charging_streaks(noisy, charging_streak_prob, charging_streak_intensity, rng)
    return noisy
