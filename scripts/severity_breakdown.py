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
# Each rung must exceed the one below it by this factor, and adjacent
# severity_continuous bands may overlap by at most this share of a band.
STEP_RATIO = 1.05
BAND_OVERLAP = 0.25


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
        bands = {}
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
            if len(sc):
                bands[lv] = (float(sc.min()), float(sc.max()))
        # A realised ladder moves its physical parameters monotonically across
        # ALL FOUR levels, and carries four of them. Checking only L4 against
        # L1 passes a ladder whose middle rungs are flat -- or one missing L2
        # entirely -- and a severity claim stratified over that is measuring
        # nothing. Fail closed: absent level, non-monotone step, or materially
        # overlapping severity_continuous bands all mark the set unrealised.
        reasons = []
        missing = [lv for lv in LEVELS if lv not in spread]
        if missing:
            reasons.append("level(s) " + ", ".join("L%d" % lv for lv in missing)
                           + " absent")
        else:
            for c in PHYSICAL:
                if c not in spread[LEVELS[0]]:
                    continue
                vals = [float(spread[lv].get(c, 0.0)) for lv in LEVELS]
                bad = [i for i, (a, b) in enumerate(zip(vals, vals[1:]))
                       if not b > STEP_RATIO * a]
                if bad:
                    steps = ", ".join(
                        "L%d->L%d (%.4g -> %.4g)" % (LEVELS[i], LEVELS[i + 1],
                                                     vals[i], vals[i + 1])
                        for i in bad)
                    reasons.append(c + " not monotone: " + steps)
            # Bands that overlap materially mean a "level" is not a distinct
            # rung, even when the means happen to progress.
            for i in range(len(LEVELS) - 1):
                lo, hi = bands.get(LEVELS[i]), bands.get(LEVELS[i + 1])
                if not lo or not hi:
                    continue
                overlap = lo[1] - hi[0]
                width = max(lo[1] - lo[0], hi[1] - hi[0], 1e-9)
                if overlap > BAND_OVERLAP * width:
                    reasons.append(
                        "severity_continuous bands L%d/L%d overlap by %.0f%% "
                        "of a band width" % (LEVELS[i], LEVELS[i + 1],
                                             100.0 * overlap / width))
        moved = not reasons
        verdicts[s] = moved
        if moved:
            print(f"{'':>4}{'':>7}-> severity is REALISED across L1->L2->L3->L4")
        else:
            print(f"{'':>4}{'':>7}-> severity is LABEL ONLY:")
            for why in reasons:
                print(f"{'':>11}   - {why}")
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
            # These three read the geometry, which exists whether or not the
            # pair was submitted. Under a table labelled submission-masked a
            # declined present pair is a MISS, not a geometric success, so the
            # tiers count it as failed and the median is taken over accepted
            # pairs only (a median over rows the submission never claimed is
            # not a quantity the rubric pays). The raw-geometry view of the
            # same rows is printed separately below.
            acc = r[r.pred_found == 1]
            t1 = 100.0 * ((r.err <= 1) & (r.pred_found == 1)).mean()
            t5 = 100.0 * ((r.err <= 5) & (r.pred_found == 1)).mean()
            med = acc.err.median() if len(acc) else np.nan
            print(f"{s:>4}{str(lv):>5}{len(r):>6}{r.loc_credit.mean():>8.4f}"
                  f"{t1:>7.1f}%{t5:>7.1f}%"
                  f"{med:>9.2f}"
                  f"{(ok.s_credit.mean() if len(ok) else np.nan):>8.4f}"
                  f"{(ok.r_credit.mean() if len(ok) else np.nan):>8.4f}"
                  f"{(ok.r_err.median() if len(ok) else np.nan):>9.3f}"
                  f"{declined:>10}")
        print("-" * 78)
    print("<=1px / <=5px count a declined present pair as a miss; med px is "
          "over accepted pairs only.")

    # The same rows read as RAW GEOMETRY -- no submission mask. Useful for one
    # question the masked table cannot answer: when a present pair is declined,
    # was the geometry wrong too, or only the confidence? Explicitly labelled,
    # because these numbers are NOT what the rubric pays.
    print(f"\n{'='*78}\nPRESENT-PAIR GEOMETRY BY SEVERITY (RAW -- NOT "
          f"submission-masked, not a score)\n{'='*78}")
    print(f"{'set':>4}{'sev':>5}{'n':>6}{'<=1px':>8}{'<=5px':>8}{'med px':>9}"
          f"{'declined':>10}{'of which <=5px':>16}")
    print("-" * 78)
    for s in sorted(scored[scored.gt_found == 1]["set"].dropna().unique()):
        rows = scored[(scored["set"] == s) & (scored.gt_found == 1)]
        for lv in list(LEVELS) + ["all"]:
            r = rows if lv == "all" else rows[rows.severity == lv]
            if not len(r):
                continue
            dec = r[r.pred_found == 0]
            dec_ok = (100.0 * (dec.err <= 5).mean()) if len(dec) else np.nan
            print(f"{s:>4}{str(lv):>5}{len(r):>6}"
                  f"{100*(r.err<=1).mean():>7.1f}%{100*(r.err<=5).mean():>7.1f}%"
                  f"{r.err.median():>9.2f}{len(dec):>10}"
                  f"{dec_ok:>15.1f}%")
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
