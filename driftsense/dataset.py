"""Dataset and augmentation for Drift-Sense.

Two things here are worth knowing:

* Labels come from `gt_x_corr`/`gt_y_corr`, not `gt_x`/`gt_y`. The upstream
  generator writes ground truth in pre-imaging coordinates and then warps the
  search frame (raster drift + barrel) without updating the label, so the raw
  columns do not point at the pattern. See scripts/gen_data.py.
* Training crops a window out of the search frame. The full 1000x1000 frame is
  ~4x the compute of a 512 window for no extra supervision (there is still
  exactly one positive), and the model is fully convolutional so it runs on
  the full frame at inference.
"""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from driftsense.model import SCALE, STRIDE, TEMPLATE_FEAT, TEMPLATE_SIZE

SEARCH_FULL = 1000


def load_manifest(split_dir: str) -> list[dict]:
    with open(os.path.join(split_dir, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_split_dir"] = split_dir
    return rows


def dihedral(img: np.ndarray, t: int) -> np.ndarray:
    """One of the 8 square symmetries. Images here are square, so the shape
    is preserved and point mapping stays simple."""
    k, flip = t % 4, t // 4
    if k:
        img = np.rot90(img, k)
    if flip:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


def dihedral_point(x: float, y: float, size: int, t: int) -> tuple[float, float]:
    """Same transform applied to a point. np.rot90 is counter-clockwise and
    maps (x, y) -> (y, size-1-x); fliplr maps (x, y) -> (size-1-x, y)."""
    k, flip = t % 4, t // 4
    for _ in range(k):
        x, y = y, size - 1 - x
    if flip:
        x = size - 1 - x
    return x, y


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent brightness/contrast/gamma/noise jitter.

    Applied separately to reference and search so the model cannot rely on
    them sharing an exposure -- in the real tool they never do.
    """
    x = img.astype(np.float32) / 255.0
    if rng.random() < 0.8:
        x = np.clip(x, 0, 1) ** float(np.exp(rng.normal(0, 0.22)))
    if rng.random() < 0.8:
        x = x * float(np.exp(rng.normal(0, 0.18))) + float(rng.normal(0, 0.06))
    if rng.random() < 0.3:
        x = x + rng.normal(0, float(rng.uniform(0.01, 0.05)), size=x.shape).astype(np.float32)

    # Multiplicative (speckle) noise. Added after failure analysis showed it is
    # the single strongest predictor of a wrong-repeat lock-on (standardised
    # effect +0.59 on the test split): because it scales with signal it is not
    # removed by the per-image standardisation the way additive noise is, and
    # the generator only ever emitted it at four discrete levels.
    if rng.random() < 0.4:
        x = x * (1.0 + rng.normal(0, float(rng.uniform(0.05, 0.40)),
                                  size=x.shape).astype(np.float32))
    # Impulse noise (dead/hot detector pixels, discharge events).
    if rng.random() < 0.15:
        hit = rng.random(x.shape) < float(rng.uniform(0.002, 0.015))
        salt = rng.random(x.shape) < 0.5
        x = np.where(hit & salt, 1.0, np.where(hit & ~salt, 0.0, x))

    return np.clip(x, 0, 1)


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per image -- the cheapest way to make the two
    frames comparable when they were acquired at different dose and gamma."""
    m, s = float(x.mean()), float(x.std())
    return (x - m) / max(s, 1e-5)


def gaussian_heatmap(h: int, w: int, ci: float, cj: float, sigma: float = 1.0) -> np.ndarray:
    """Soft target peak. A hard one-hot target makes the classification loss
    fight the offset head over sub-cell placement; a narrow Gaussian keeps the
    peak sharp while still giving gradient to the immediate neighbours."""
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    return np.exp(-((yy - ci) ** 2 + (xx - cj) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def build_sample(reference: np.ndarray, search: np.ndarray, gx: float, gy: float,
                 crop: int, train: bool, rng: np.random.Generator, resp: int) -> dict:
    """Turn one (reference, search, ground-truth) triple into a training tensor
    sample: dihedral + photometric augmentation, window crop, standardisation,
    and the response-grid heatmap/offset targets.

    Shared by the on-disk dataset and the streaming one so both paths are
    provably identical -- the streaming dataset differs only in where the
    images come from.
    """
    if train:
        t = int(rng.integers(0, 8))
        reference = dihedral(reference, t)
        search = dihedral(search, t)
        gx, gy = dihedral_point(gx, gy, SEARCH_FULL, t)

    # Reference -> template at the search frame's pixel size. INTER_AREA
    # matches the area-average downsampling the physical search image
    # already went through.
    if reference.shape[0] == TEMPLATE_SIZE and reference.shape[1] == TEMPLATE_SIZE:
        # Compact training pools (scripts/gen_data.py --store-templates) store
        # this downsample already applied, because it is the *only* thing the
        # reference is ever used for -- on this path and in matching.locate,
        # which correlates against the template too. An exact 10x area average
        # commutes with the 8 square symmetries (the block partition of a
        # 1000px frame into 10x10 cells is invariant under them), so applying
        # it before rather than after `dihedral` above is byte-identical --
        # verified over 20 images x 8 symmetries, max difference 0. It costs
        # ~70x less disk per pair, which is what makes a 300k-pair pool fit.
        template = reference
    else:
        template = cv2.resize(reference, (TEMPLATE_SIZE, TEMPLATE_SIZE),
                              interpolation=cv2.INTER_AREA)

    if train:
        template = photometric(template, rng)
        search = photometric(search, rng)
    else:
        template = template.astype(np.float32) / 255.0
        search = search.astype(np.float32) / 255.0

    bx, by = gx - TEMPLATE_SIZE / 2.0, gy - TEMPLATE_SIZE / 2.0  # box top-left

    if train and crop < SEARCH_FULL:
        # Window must contain the whole target box, else there is no positive
        # cell to supervise.
        lo_x = max(0.0, bx + TEMPLATE_SIZE - crop)
        hi_x = min(bx, SEARCH_FULL - crop)
        lo_y = max(0.0, by + TEMPLATE_SIZE - crop)
        hi_y = min(by, SEARCH_FULL - crop)
        ox = int(round(rng.uniform(min(lo_x, hi_x), max(lo_x, hi_x))))
        oy = int(round(rng.uniform(min(lo_y, hi_y), max(lo_y, hi_y))))
        ox = int(np.clip(ox, 0, SEARCH_FULL - crop))
        oy = int(np.clip(oy, 0, SEARCH_FULL - crop))
        search = search[oy:oy + crop, ox:ox + crop]
        bx, by = bx - ox, by - oy

    template = standardize(template)
    search = standardize(search)

    ci, cj = by / STRIDE, bx / STRIDE
    h = w = resp if train else (search.shape[0] // STRIDE - TEMPLATE_FEAT + 1)
    ti = int(np.clip(round(ci), 0, h - 1))
    tj = int(np.clip(round(cj), 0, w - 1))

    heat = gaussian_heatmap(h, w, ci, cj)
    offset = np.array([cj - tj, ci - ti], dtype=np.float32)  # (dx, dy) in stride units

    return {
        "template": torch.from_numpy(template[None].astype(np.float32)),
        "search": torch.from_numpy(search[None].astype(np.float32)),
        "heat": torch.from_numpy(heat[None]),
        "offset": torch.from_numpy(offset),
        "peak": torch.tensor([ti, tj], dtype=torch.long),
        "gt": torch.tensor([bx + TEMPLATE_SIZE / 2.0, by + TEMPLATE_SIZE / 2.0],
                           dtype=torch.float32),
    }


class DriftSenseDataset(Dataset):
    def __init__(self, split_dirs, crop: int = 512, train: bool = True, seed: int = 0,
                 limit: int | None = None, epoch: int = 0):
        self.rows = []
        for d in split_dirs:
            self.rows.extend(load_manifest(d))
        if limit:
            self.rows = self.rows[:limit]
        self.crop = crop
        self.train = train
        self.seed = seed
        self.epoch = epoch
        self.resp = crop // STRIDE - TEMPLATE_FEAT + 1

    def __len__(self):
        return len(self.rows)

    def reload(self, split_dirs, limit: int | None = None) -> int:
        """Re-read the manifests, picking up pool shards finished since startup.

        Generation runs at ~1/6 of the GPU's step rate on this machine, so a
        pool worth training on is still being written while training runs. This
        lets the training set grow between epochs instead of being fixed at
        whatever existed when the process started. Pair it with a fixed
        --samples-per-epoch so the number of steps per epoch (and therefore the
        one-cycle schedule) does not move underneath the scheduler.
        """
        rows = []
        for d in split_dirs:
            rows.extend(load_manifest(d))
        if limit:
            rows = rows[:limit]
        self.rows = rows
        return len(rows)

    def set_epoch(self, epoch: int):
        """Re-roll the augmentation for the next pass over the pool.

        Without this the per-sample rng is keyed on `idx` alone, so every epoch
        replays the *identical* dihedral, photometric jitter and search-window
        crop for each pair: a second pass over the pool is a literal repeat, not
        a new view. That is invisible when the pool is read once, and expensive
        when a long run makes several passes -- which is exactly the regime the
        streaming experiment showed matters (NIGHT_LOG.md: fresh scenes were
        worth +2.9 points over a replayed pool).

        The caller must build the DataLoader with `persistent_workers=False`,
        or this never reaches the workers -- the same trap that silently froze
        the streaming dataset. train.py does; `scripts/check_epoch_variety.py`
        checks it empirically rather than trusting the comment.
        """
        self.epoch = epoch

    def _read(self, row, key):
        path = os.path.join(row["_split_dir"], row[key])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        # epoch 0 reproduces the original single-pass seeding exactly.
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + self.epoch * 7_654_321 + idx) & 0xFFFFFFFF)
        return build_sample(self._read(row, "reference_path"),
                            self._read(row, "search_path"),
                            float(row["gt_x_corr"]), float(row["gt_y_corr"]),
                            self.crop, self.train, rng, self.resp)
