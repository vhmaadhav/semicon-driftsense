"""Infinite-data training: generate fresh scenes in the dataloader workers.

Why this exists
---------------
Both fixed-dataset training runs showed the same signature -- training loss
kept falling while held-out accuracy plateaued or declined. That is
overfitting to a finite pool of scenes (3 500 of them), not a capacity or
schedule limit. The obvious fix is more scenes, but storing them is the
bottleneck: a reference image is ~0.7 MB, so another 24 000 pairs would need
~17 GB of disk.

Generating instead of storing removes the constraint entirely. A 10000x10000
canvas costs ~0.6 s and yields 8 (reference, search) pairs, so one worker
produces ~12 samples/s and five produce ~59/s -- comfortably above the ~11
img/s the MPS training step consumes. The model therefore sees a *fresh scene
every time* and cannot memorise the training set.

Sample construction goes through driftsense.dataset.build_sample, the same
function the on-disk dataset uses, so the two paths are identical apart from
where the images come from.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from driftsense.dataset import build_sample
from driftsense.generate import PRESETS, PoseSpec, make_pairs
from driftsense.model import STRIDE, TEMPLATE_FEAT


class StreamingDriftSense(IterableDataset):
    """Yields freshly generated training samples, forever.

    `length` only defines what one "epoch" means for the scheduler and the
    progress display -- there is no fixed dataset to exhaust.
    """

    def __init__(self, length: int = 14000, crop: int = 512,
                 crops_per_canvas: int = 8, noise: str = "randomized",
                 architectures: list[str] | None = None, seed: int = 0,
                 epoch: int = 0, pose: "PoseSpec | None" = None,
                 pose_jitter: tuple[float, float] = (0.0, 0.0)):
        self.length = length
        self.crop = crop
        self.crops_per_canvas = crops_per_canvas
        self.noise = noise
        self.architectures = architectures or list(PRESETS.keys())
        self.seed = seed
        self.epoch = epoch
        self.resp = crop // STRIDE - TEMPLATE_FEAT + 1
        # Phase 2 pose distribution and the residual-error jitter applied when
        # canonicalising. Defaults reproduce the Phase 1 stream exactly.
        self.pose = pose or PoseSpec()
        self.pose_jitter = pose_jitter

    def __len__(self):
        return self.length

    def set_epoch(self, epoch: int):
        """Shift the seed stream so a resumed/next epoch draws new scenes.

        Only effective when the dataloader does *not* use persistent workers --
        see the `_pass` counter in __iter__ for why, and for the fallback that
        keeps the stream advancing when it does.
        """
        self.epoch = epoch

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1

        # Each (epoch, worker, pass) gets a disjoint, reproducible seed stream.
        #
        # `_pass` is not redundant with `epoch`. With persistent_workers=True the
        # worker processes are forked once and hold their own copy of this
        # object, so set_epoch() in the parent never reaches them: self.epoch
        # stays frozen at its value when the workers were forked, every epoch
        # re-draws the identical seed stream, and the whole point of streaming
        # -- never reusing a scene -- is silently lost. (Measured: with
        # persistent workers, epoch 1 reproduced epoch 0's scenes byte for byte.)
        #
        # A worker's copy *is* long-lived under exactly that setting, though, so
        # a counter stored on it survives between epochs and advances the stream
        # even when set_epoch cannot. Together the two keys cover both loader
        # configurations.
        self._pass = getattr(self, "_pass", 0) + 1
        base = np.random.SeedSequence([self.seed, self.epoch, wid, self._pass])
        rng = np.random.default_rng(base)

        # Split the nominal epoch length across workers.
        quota = self.length // max(nworkers, 1)
        produced = 0
        while produced < quota:
            entropy = int(rng.integers(0, 2 ** 63 - 1))
            pairs = make_pairs(entropy, self.architectures, self.noise,
                               crops=self.crops_per_canvas, pose=self.pose)
            for p in pairs:
                if produced >= quota:
                    break
                sample_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
                yield build_sample(p["reference"], p["search"], p["gt_x"], p["gt_y"],
                                   self.crop, True, sample_rng, self.resp,
                                   magnification=p.get("magnification", 10.0),
                                   rotation_deg=p.get("rotation_deg", 0.0),
                                   found=p.get("found", 1),
                                   pose_jitter=self.pose_jitter)
                produced += 1
