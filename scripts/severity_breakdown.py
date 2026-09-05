#!/usr/bin/env python3
"""Per-severity component breakdown for a Phase 2 per-pair results CSV.

G2 asks what a severity-3/4-weighted Set B costs. `grade_emulation.py
--severity-sweep` answers that at the level of the *grade* (F1, bonus gate,
subtotal under stratified A70/B70/C40 draws). This script answers the
component question underneath it: which parts of the rubric degrade with
severity, and by how much, per set and per level.

Two rules it exists to enforce:

* **Credit is read from the shared scorer**, `driftsense.rubric.score`, so the
  localisation and pose credit here are the submission-masked values -- a
  present pair the system declined contributes zero, exactly as on the graded
  CSV. Grouping raw predictions by severity (the obvious thing to do) reports
  accuracy conditional on the pairs we accepted, which improves as the system
  declines more, and is therefore backwards as a severity measurement.
* **Severity is verified as REALISED, not trusted as a label.** With
  `--manifest`, the per-level `severity_continuous` and a physical proxy
  (`drift_jitter_px`) are printed alongside the counts. A label whose physical
  parameters do not move across levels is a severity that was asked for and
  never rendered -- the defect that retracted the 81.45/81.93 figures, and the
  one `tests/test_severity_pin_guard.py` now guards at the CLI.

    python scripts/severity_breakdown.py --csv .agents/run.csv \\
        --manifest 'data/ext_p2/*_manifest.csv'
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from driftsense.config import SHIPPED_THRESHOLD
from driftsense.rubric import ROT_TIERS, SCALE_TIERS, score, tier

LEVELS = (1, 2, 3, 4)
# Physical parameters the generator's severity ladder drives. Used only to
# check that a severity LABEL corresponds to rendered severity.
PHYSICAL = ("drift_jitter_px", "speckle_sigma", "detector_noise_sigma_search")


def realised_severity_audit(manifest_glob):
    """Is the severity label backed by rendered severity, per set?"""
    files = sorted(glob.glob(manifest_glob))
    if not files:
        print(f"no manifests matched {manifest_glob!r} -- realised-severity "
              f"audit SKIPPED (the label is unverified)")
        return None
    man = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"\n{'='*78}\nREALISED SEVERITY AUDIT — {len(man)} pairs over "
          f"{len(files)} manifest(s)\n{'='*78}")
    print(f"{'set':>4}{'level':>7}{'n':>6}{'sev_cont mean':>15}{'[min, max]':>18}"
          f"{'drift_jitter':>14}{'speckle':>10}")
    print("-" * 78)
    verdicts = {}
    for s in sorted(man.phase2_set.dropna().unique()):
        sub = man[man.phase2_set == s]
        spread = {}
        for lv in LEVELS:
            r = sub[sub.severity_level == lv]
            if not len(r):
                continue
            sc = r.severity_continuous.dropna()
            rng = (f"[{sc.min():.3f}, {sc.max():.3f}]" if len(sc) else "[-, -]")
            print(f"{s:>4}{lv:>7}{len(r):>6}"
                  f"{(f'{sc.mean():.4f}' if len(sc) else 'NOT RECORDED'):>15}{rng:>18}"
                  f"{r.drift_jitter_px.mean():>14.3f}{r.speckle_sigma.mean():>10.4f}")
            spread[lv] = {c: r[c].mean() for c in PHYSICAL if c in r.columns}
        # A realised ladder moves its physical parameters monotonically across
        # levels. A flat one means the label is decorative.
        moved = all(
            spread[LEVELS[-1]].get(c, 0) > 1.15 * spread[LEVELS[0]].get(c, 0)
            for c in PHYSICAL if c in spread.get(LEVELS[0], {}))
        verdicts[s] = moved
        print(f"{'':>4}{'':>7}{'-> severity is ' + ('REALISED' if moved else 'LABEL ONLY (physical parameters flat across levels)'):<60}")
    print("-" * 78)
    flat = [s for s, ok in verdicts.items() if not ok]
    if flat:
        print(f"WARNING: sets {flat} carry severity LABELS whose physical "
              f"parameters do not vary across the four levels.\n"
              f"         Any severity result stratified over those sets is "
              f"measuring nothing. Restrict severity claims to "
              f"{[s for s, ok in verdicts.items() if ok]}.")
    return verdicts


def breakdown(df, threshold):
    """Component credit per (set, severity), read off the shared scorer."""
    _, scored = score(df, threshold, quiet=True)
    scored["s_credit"] = scored.s_err.map(lambda v: tier(v, SCALE_TIERS))
    scored["r_credit"] = scored.r_err.map(lambda v: tier(v, ROT_TIERS))

    print(f"\n{'='*78}\nPRESENT-PAIR COMPONENTS BY SEVERITY (submission-masked, "
          f"t={threshold})\n{'='*78}")
    print(f"{'set':>4}{'sev':>5}{'n':>6}{'loc':>8}{'<=1px':>8}{'<=5px':>8}"
          f"{'med px':>9}{'scale':>8}{'rot':>8}{'med rot':>9}{'declined':>10}")
    print("-" * 78)
    for s in sorted(scored[scored.gt_found == 1]["set"].dropna().unique()):
        rows = scored[(scored["set"] == s) & (scored.gt_found == 1)]
        for lv in list(LEVELS) + ["all"]:
            r = rows if lv == "all" else rows[rows.severity == lv]
            if not len(r):
                continue
            # Pose is scored only where localisation earned credit -- the
            # rubric's own conditioning, and the mask already applied.
            ok = r[r.loc_credit > 0]
            declined = int((r.pred_found == 0).sum())
            print(f"{s:>4}{str(lv):>5}{len(r):>6}{r.loc_credit.mean():>8.4f}"
                  f"{100*(r.err<=1).mean():>7.1f}%{100*(r.err<=5).mean():>7.1f}%"
                  f"{r.err.median():>9.2f}"
                  f"{(ok.s_credit.mean() if len(ok) else np.nan):>8.4f}"
                  f"{(ok.r_credit.mean() if len(ok) else np.nan):>8.4f}"
                  f"{(ok.r_err.median() if len(ok) else np.nan):>9.3f}"
                  f"{declined:>10}")
        print("-" * 78)

    # ---- Rejection confusion by severity ---------------------------------
    # The two error kinds live in different sets: declining a real pair is a
    # Set A/B error, accepting an absent one is a Set C error. Reporting them
    # pooled hides which set a severity shift is actually hurting.
    print(f"\n{'='*78}\nREJECTION CONFUSION BY SEVERITY (t={threshold}, "
          f"reject-positive)\n{'='*78}")
    gray = scored[scored["set"].isin(["A", "B", "C"])]
    print(f"{'set':>4}{'sev':>5}{'n':>6}{'lost-real':>11}{'missed-abs':>12}"
          f"{'correct-rej':>13}{'rate':>9}")
    print("-" * 78)
    for s in sorted(gray["set"].dropna().unique()):
        rows = gray[gray["set"] == s]
        for lv in list(LEVELS) + ["all"]:
            r = rows if lv == "all" else rows[rows.severity == lv]
            if not len(r):
                continue
            lost = int(((r.pred_found == 0) & (r.gt_found == 1)).sum())
            missed = int(((r.pred_found == 1) & (r.gt_found == 0)).sum())
            crej = int(((r.pred_found == 0) & (r.gt_found == 0)).sum())
            n_pres, n_abs = int((r.gt_found == 1).sum()), int((r.gt_found == 0).sum())
            rate = (lost / n_pres if n_pres else missed / n_abs if n_abs else np.nan)
            print(f"{s:>4}{str(lv):>5}{len(r):>6}{lost:>11}{missed:>12}{crej:>13}"
                  f"{rate:>9.3f}")
        print("-" * 78)

    # Full-frame F1 at the shipped threshold, for reference against the
    # stratified-draw distribution grade_emulation.py reports.
    pf, gt = (gray.score >= threshold).values, gray.gt_found.values
    tp = int(((~pf) & (gt == 0)).sum())
    fp = int(((~pf) & (gt == 1)).sum())
    fn = int((pf & (gt == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    print(f"\nfull-frame F1(reject) @ {threshold} = {f1:.4f}  "
          f"(correct-rej {tp}  lost-real {fp}  missed-abs {fn})  "
          f"over {len(gray)} grayscale pairs")
    print("This is the POINT estimate on the whole pool. The graded set is 180 "
          "pairs;\nfor the sampling distribution and the bonus gate use "
          "scripts/grade_emulation.py.")
    return scored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="per-pair results CSV")
    ap.add_argument("--manifest", default=None,
                    help="glob of dataset manifests, to verify that the "
                         "severity labels correspond to rendered severity")
    ap.add_argument("--threshold", type=float, default=SHIPPED_THRESHOLD)
    a = ap.parse_args(argv)

    df = pd.read_csv(a.csv)
    print(f"CSV: {a.csv}  ({len(df)} pairs)")
    if "severity" not in df.columns:
        ap.error("CSV has no `severity` column -- nothing to stratify by")
    if a.manifest:
        realised_severity_audit(a.manifest)
    breakdown(df, a.threshold)


if __name__ == "__main__":
    main()
