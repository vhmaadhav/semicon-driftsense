"""Siamese correlation network for Drift-Sense navigation-error recovery.

Design rationale
----------------
The layout is periodic, so *local* appearance cannot identify a site: dozens
of positions in the search frame look the same at patch level. Three things
actually break the tie, and the architecture is built around them.

1. Large-scale zone structure (array mats separated by flat routing strips).
   Only visible with a wide receptive field, so a context branch runs over
   the whole search frame and conditions the response map.
2. The arrangement of the decoy peaks themselves. The head is deliberately
   deep and dilated *over the response map*, so it sees the whole lattice of
   candidate peaks rather than scoring each in isolation.
3. Per-line CD/placement fingerprints, which survive the 10x downsample as
   low-amplitude intensity modulation. That is what the correlation branch
   picks up.

Layout is a SiamRPN++-style cross-correlation: a shared encoder embeds both
the template and the search frame, a grouped cross-correlation produces a
multi-channel match-evidence volume, and the head turns that (plus context)
into a centre heatmap and a sub-cell offset field.

Geometry is fixed by the problem statement and hard-coded rather than
searched: reference is 1 nm/px and search is 10 nm/px, so the reference
occupies exactly a 100x100 px box in the search frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

REF_SIZE = 1000       # reference frame, 1 nm/px
SEARCH_SIZE = 1000    # search frame, 10 nm/px
SCALE = 10            # pixel-size ratio -> reference footprint is 100x100 px
TEMPLATE_SIZE = REF_SIZE // SCALE   # 100
STRIDE = 4            # encoder total stride
TEMPLATE_FEAT = TEMPLATE_SIZE // STRIDE   # 25


def conv_bn(cin, cout, k=3, stride=1, dilation=1):
    pad = dilation * (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        pad = dilation
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class Encoder(nn.Module):
    """Shared embedding trunk, total stride 4.

    Kept deliberately *local* (modest receptive field): it runs on both the
    25x25 template and the full search frame, and a huge receptive field here
    would make the template embedding mostly padding. Global reasoning is the
    context branch's job instead.
    """

    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn(1, width // 2, k=7, stride=2),     # /2
            conv_bn(width // 2, width, k=3),
            conv_bn(width, width, k=3, stride=2),      # /4
        )
        self.body = nn.Sequential(
            ResBlock(width, dilation=1),
            ResBlock(width, dilation=2),
        )
        self.out = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
        )

    def forward(self, x):
        return self.out(self.body(self.stem(x)))


class ContextBranch(nn.Module):
    """Wide-receptive-field description of the search frame.

    Stacked dilations reach several hundred search pixels, which is the scale
    of the mat/strip composition -- the strongest globally unique cue in the
    layout, and the one plain correlation cannot see.
    """

    def __init__(self, cin=64, cout=32):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(cin, cout, 3, dilation=2),
            conv_bn(cout, cout, 3, dilation=4),
            conv_bn(cout, cout, 3, dilation=8),
            conv_bn(cout, cout, 3, dilation=16),
        )

    def forward(self, x):
        return self.body(x)


CORR_GROUPS = 8


def grouped_xcorr(search_feat: torch.Tensor, template_feat: torch.Tensor,
                  groups: int = CORR_GROUPS) -> torch.Tensor:
    """Correlate each sample's template over its own search embedding.

    Channels are split into `groups` bands and summed within each band, giving
    `groups` channels of match evidence (fully depthwise -- one channel per
    feature -- measures the same thing but was 4.5x slower to backprop through
    on MPS, where grouped-conv weight gradients scale badly with group count;
    at 512 groups it dominated the whole training step).

    Returns (B, groups, Hs-Ht+1, Ws-Wt+1).
    """
    b, c, _, _ = search_feat.shape
    ht, wt = template_feat.shape[-2:]
    outs = [
        F.conv2d(search_feat[i:i + 1],
                 template_feat[i].reshape(groups, c // groups, ht, wt),
                 groups=groups)
        for i in range(b)
    ]
    return torch.cat(outs, dim=0)


class DriftSenseNet(nn.Module):
    """Reference + search -> (centre heatmap, sub-cell offsets).

    Response cell (i, j) means "template top-left at search pixel
    (4j, 4i)", i.e. centre (4j + 50, 4i + 50). The offset head predicts a
    correction in units of the stride, so the final prediction is
    ((j + dx) * 4 + 50, (i + dy) * 4 + 50) and is continuous.
    """

    def __init__(self, width=64, ctx=32, head=64):
        super().__init__()
        self.encoder = Encoder(width)
        self.context = ContextBranch(width, ctx)
        # E1 efficiency cache: the template-branch embedding is identical for
        # every pose hypothesis of a pair (locate_phase2 canonicalizes the
        # SEARCH, so the template tensor is byte-identical across attempts).
        # Single-slot, keyed on the input bytes + device; never served in
        # training mode. use_template_cache=False restores the pre-E1
        # behaviour (every hypothesis re-encodes) -- the A/B benchmark's
        # "existing" baseline. See tests/test_search_feat_cache.py.
        self.use_template_cache = True
        self._tf_cache = None
        # Issue #21: the cache key covers the input bytes, not the parameter
        # state -- any weight change must invalidate it. load_state_dict is
        # the realistic mid-session mutation (e.g. two eval passes on one
        # model instance); optimizer steps mutate in training mode, where the
        # cache is never populated anyway.
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: setattr(module, "_tf_cache", None))
        self.corr_mix = nn.Sequential(
            nn.Conv2d(CORR_GROUPS, head, 1, bias=False),
            nn.BatchNorm2d(head),
            nn.ReLU(inplace=True),
        )
        # Dilated stack over the response map: lets the head judge a peak
        # against the surrounding lattice of decoy peaks, not on its own.
        self.head = nn.Sequential(
            conv_bn(head + ctx, head, 3),
            conv_bn(head, head, 3, dilation=2),
            conv_bn(head, head, 3, dilation=4),
            conv_bn(head, head, 3, dilation=8),
        )
        self.logit = nn.Conv2d(head, 1, 1)
        self.offset = nn.Conv2d(head, 2, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # The two prediction heads must NOT get the kaiming treatment: they are
        # 1x1 convs with one/two output channels, so fan_out is 1 and kaiming
        # hands them a std near sqrt(2). Summed over 64 inputs that puts the
        # initial logits around +/-11, saturating the sigmoid and making the
        # focal loss start in the thousands.
        for m in (self.logit, self.offset):
            nn.init.normal_(m.weight, std=0.01)
        # Low prior probability: one positive against ~10^4 negatives, so an
        # unbiased init spends the first epochs just unlearning the prior.
        nn.init.constant_(self.logit.bias, -4.0)
        nn.init.constant_(self.offset.bias, 0.0)

    def forward(self, reference: torch.Tensor, search: torch.Tensor,
                ref_feat: "torch.Tensor | None" = None) -> dict:
        """reference: (B,1,1000,1000) or a pre-downsampled (B,1,100,100)
        template. search: (B,1,H,W) with H,W multiples of the stride.

        Any reference that is not exactly TEMPLATE_SIZE square -- including a
        non-square one that happens to be TEMPLATE_SIZE *wide* -- is area-
        resized to the template size first, so both branches see one shape.

        ref_feat: a precomputed template-branch embedding, or None to encode
        here. Inference also consults a single-slot cache keyed on the input
        bytes (identical templates across pose hypotheses hit it); training
        never does."""
        if ref_feat is not None:
            tf = ref_feat
        elif self.training or not self.use_template_cache:
            # Pre-E1 behaviour: every call re-encodes, nothing is stored.
            if reference.shape[-2:] != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                reference = F.interpolate(
                    reference, size=(TEMPLATE_SIZE, TEMPLATE_SIZE),
                    mode="area")
            tf = self.encoder(reference)
        else:
            key = (reference.detach().cpu().numpy().tobytes(),
                   str(reference.device))
            if self._tf_cache is not None and self._tf_cache[0] == key:
                tf = self._tf_cache[1]
            else:
                if reference.shape[-2:] != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                    reference = F.interpolate(
                        reference, size=(TEMPLATE_SIZE, TEMPLATE_SIZE),
                        mode="area")
                tf = self.encoder(reference)
                self._tf_cache = (key, tf)

        sf = self.encoder(search)

        # L2-normalise so the match score reflects pattern agreement rather
        # than local contrast -- reference and search are captured at very
        # different dose/gamma, so raw magnitudes are not comparable.
        tf = F.normalize(tf, dim=1)
        sf = F.normalize(sf, dim=1)

        corr = grouped_xcorr(sf, tf)
        corr = self.corr_mix(corr)

        ctx = self.context(sf)
        # Align context to the response grid: response (i,j) has its centre at
        # feature (i + Ht//2, j + Wt//2). Taken from the actual template shape
        # rather than the constant, so a non-1000px reference still aligns.
        oi, oj = tf.shape[-2] // 2, tf.shape[-1] // 2
        ctx = ctx[:, :, oi:oi + corr.shape[-2], oj:oj + corr.shape[-1]]

        feat = self.head(torch.cat([corr, ctx], dim=1))
        return {"logit": self.logit(feat), "offset": self.offset(feat)}
