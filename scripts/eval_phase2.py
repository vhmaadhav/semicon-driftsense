#!/usr/bin/env python3
"""Score a split against the Phase 2 rubric.

Implements the published credit tiers so that a local number means the same
thing as a blind-set number: tiered localisation credit, pose credit
conditional on localisation, rejection F1 on the `found` flag, and the AUC of
the confidence column against per-pair correctness.

Five properties keep this honest, all of which it previously got wrong:

* **It decodes with the shipped config**, imported from `driftsense.config`
  rather than inherited from `locate_phase2`'s function defaults.
* **It scores the shipped statistic** -- `confidence` (legacy_min), the value
  `register.py` actually thresholds -- not the raw network `score`.
* **It reports rejection F1 reject-positive at the shipped threshold.** The
  lenient reading and the swept oracle optimum are printed alongside, labelled,
  because they are strictly more flattering and are not the planning number.
* **It masks declined present pairs** (added 2026-09-05). `register.py`
  zero-fills the pose/location columns of a declined answer, so a wrongly
  declined PRESENT pair earns zero localisation and zero pose on the graded
  CSV. This script used to score the raw predictions, crediting pairs the
  submitted output never claimed -- so its localisation and pose numbers were
  conditional on the pairs we happened to accept, and rose as the system got
  more timid.
* **It weights the A/B strata** when the split carries `phase2_set` labels
  (0.45 A + 0.55 B, the published weighting). Splits from our own generator
  carry no such labels, and are scored pooled with the header saying so.

The last two come from `driftsense.rubric`, the single shared implementation
`scripts/eval_ext.py` also calls -- the two scorers disagreed before, which is
why component deltas measured here could not be compared against an eval_ext
subtotal.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer as I
from driftsense.matching import locate_phase2
# The shipped decode config -- read from driftsense.config, never re-stated as a
# local literal, so this evaluator cannot drift away from what register.py runs.
# Previously this script relied on locate_phase2's *function defaults*, which
# happened to equal the SHIPPED_* values; that agreement was a coincidence, not
# a guarantee, and a change to config.py would have silently left this script
# measuring the old decode.
from driftsense.config import (SHIPPED_BAND, SHIPPED_SUBPIXEL_ROWS,
                               SHIPPED_THRESHOLD, SHIPPED_VERIFICATION)
from driftsense.rubric import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="reference machine has 4 cores")
    ap.add_argument("--out", default=None,
                    help="per-pair CSV (default: <split>/phase2_eval.csv). Keep it: "
                         "every severity/stratum breakdown is computed from this "
                         "file, not from a second inference pass.")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    d = pd.read_csv(os.path.join(a.split, "manifest.csv"))
    if a.limit:
        d = d.head(a.limit)
    model, device = I.load_model(a.weights)

    rows = []
    for _, r in d.iterrows():
        ref = I.read_gray(os.path.join(a.split, r.reference_path))
        sea = I.read_gray(os.path.join(a.split, r.search_path))
        t0 = time.perf_counter()
        res = locate_phase2(model, ref, sea, device, refine=True,
                            band=SHIPPED_BAND, verification=SHIPPED_VERIFICATION,
                            subpixel_rows=SHIPPED_SUBPIXEL_ROWS)
        dt = time.perf_counter() - t0
        row = dict(
            pair_id=r.get("pair_id", r.get("id", len(rows))),
            # Carried through when the split has them; absent on our own
            # generator's splits, where score() then pools the strata.
            severity=r.get("severity_level", -1),
            severity_continuous=r.get("severity_continuous", np.nan),
            architecture=r.get("architecture", ""),
            gt_found=int(r.found),
            gt_x=r.gt_x_corr, gt_y=r.gt_y_corr,
            gt_scale=r.magnification, gt_rot=r.rotation_deg,
            x=res["x"], y=res["y"],
            scale=res.get("scale", np.nan), theta=res.get("theta", np.nan),
            # The SHIPPED statistic, identical to register.py:523 and
            # eval_ext.py's _worker: `confidence` (legacy_min = min(network
            # score, native ZNCC)), NOT the raw network `score`. They are
            # different unit systems -- SHIPPED_THRESHOLD is calibrated against
            # confidence, so thresholding the network score here measured
            # something the graded system never computes.
            score=float(res.get("confidence", res.get("score", np.nan))),
            net_score=float(res.get("score", np.nan)),
            secs=dt)
        if "phase2_set" in d.columns:
            row["set"] = r.phase2_set
        rows.append(row)

    o = pd.DataFrame(rows)
    out = a.out or os.path.join(a.split, "phase2_eval.csv")
    o.to_csv(out, index=False)

    # One shared rubric implementation (driftsense.rubric.score): submission
    # masking, A/B strata when labelled, reject-positive F1 at the shipped
    # threshold, and both calibration variants while G5 is unresolved.
    score(o, SHIPPED_THRESHOLD, label=f"SPLIT {os.path.basename(a.split.rstrip('/'))}")
    print(f"{'RUNTIME':<28}{f'median {o.secs.median():.2f}s  p90 {o.secs.quantile(.9):.2f}s':>26}"
          f"{'':>10}\n{'':<28}{f'max {o.secs.max():.2f}s  [{a.threads} threads]':>26}")
    print(f"\nper-pair predictions: {out}")


if __name__ == "__main__":
    main()
