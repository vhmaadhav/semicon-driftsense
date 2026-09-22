"""The Phase 2 rubric scorer -- ONE definition, shared by every evaluator.

`scripts/eval_ext.py` (external shards, A/B/C/D labelled) and
`scripts/eval_phase2.py` (splits from our own generator, unlabelled) previously
carried two independent implementations of the same rubric, and they disagreed
on two things that both inflate a score:

* **Submission masking.** `register.py` zero-fills the pose/location columns of
  a declined answer, so a wrongly declined PRESENT pair earns zero localisation
  (and therefore zero pose) on the graded CSV. `eval_ext` masked by
  `pred_found`; `eval_phase2` scored the raw predictions, crediting pairs the
  submitted output never claimed.
* **A/B strata.** The rubric weights localisation 0.45 A + 0.55 B.
  `eval_phase2` pooled the two, which silently reweights the degraded set
  towards whatever mixture the split happened to contain.

Both now call `score()` here. The A/B path is byte-identical to the eval_ext
implementation it was lifted from (pinned by tests/test_eval_found_masking.py
and tests/test_score_semantics.py). A frame with no `set` column -- which is
every split our own generator writes, since it emits no `phase2_set` -- is
scored as a single pooled present-pair stratum and the header says so, because
applying 0.45/0.55 to unlabelled data would invent a composition the split does
not have.
"""
from __future__ import annotations

import numpy as np

# Published Phase 2 credit tiers.
LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))
SCALE_TIERS = ((0.01, 1.00), (0.02, 0.60), (0.05, 0.30))
ROT_TIERS = ((0.25, 1.00), (0.50, 0.60), (1.00, 0.30))
# Localisation weighting across the two present-pair sets.
W_A, W_B = 0.45, 0.55


def tier(value: float, tiers) -> float:
    for bound, credit in tiers:
        if value <= bound:
            return credit
    return 0.0


def score(df, threshold, quiet=False, label="EXTERNAL TEST SET"):
    df = df.copy()
    # Splits from our own generator carry no phase2_set column, so there are
    # no A/B strata to weight and no C label to select the grayscale subset
    # by. Score them pooled rather than pretending to a composition they do
    # not have -- and record the fact so the header can say it out loud.
    pooled = "set" not in df.columns
    if pooled:
        df["set"] = np.where(df.gt_found.values == 1, "A", "C")
    df["err"] = np.where(df.gt_found == 1, np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    df["pred_found"] = (df.score >= threshold).astype(int)
    df["loc_credit"] = df.err.map(lambda e: tier(e, LOC_TIERS) if np.isfinite(e) else np.nan)
    df["s_err"] = np.abs(df.scale - df.gt_scale) / df.gt_scale
    df["r_err"] = np.abs(df.theta - df.gt_rot)

    # ---- Submission masking, applied to the FULL frame ----------------------
    # register.py writes zero pose/location fields for a declined answer, so
    # the grader credits a wrongly declined PRESENT pair with zero
    # localisation (and therefore zero pose). Mask by pred_found -- parity
    # with optimize_threshold.points() (issue #22 P0). The mask is applied to
    # every row (A/B/C *and* D) BEFORE any subset is taken, so the Set D bonus
    # path cannot read credit a declined answer could never have earned
    # (PR #24/#25 review): for a declined pair register.py submits x=y=0
    # against a real target, so its geometric credit is not submittable.
    df["loc_credit"] = np.where(
        df.pred_found.values == 1, df.loc_credit.fillna(0.0).values, 0.0)

    gray = df[df["set"].isin(["A", "B", "C"])]

    # ---- Localisation (40 pts), weighted 0.45 A + 0.55 B -------------------
    # Masking note: pred_found masking of loc_credit is applied to the FULL df
    # above (all sets, before this split) -- supersedes the gray-only mask from
    # the #18 lineage and also covers the Set D bonus path (PR #25 review).
    present = gray[gray.gt_found == 1]

    parts, res = {}, {}
    for s in ("A", "B"):
        p = present[present["set"] == s]
        parts[s] = p.loc_credit.mean() if len(p) else np.nan
    if pooled:
        # One unlabelled present-pair stratum: weight 1.0. Using W_A/W_B here
        # would multiply the only measured mean by 0.45 and add NaN.
        loc = parts["A"]
    else:
        loc = W_A * parts["A"] + W_B * parts["B"]
    res["localisation"] = (loc, 40 * loc)

    # ---- Pose (20 pts), only where localisation earned credit --------------
    ok = present[present.loc_credit > 0]
    sc = ok.s_err.map(lambda v: tier(v, SCALE_TIERS)).mean()
    rc = ok.r_err.map(lambda v: tier(v, ROT_TIERS)).mean()
    res["scale"] = (sc, 10 * sc)
    res["rotation"] = (rc, 10 * rc)

    # ---- Rejection (15 pts): F1 on `found` over all grayscale pairs --------
    # Which class is "positive" is NOT resolved by the source material, and the
    # two readings differ by ~1.7 points, so both are reported.
    #
    #   * The scoring slide's line "never rejecting scores zero here" can only
    #     be true under *reject*-as-positive: on 140 present / 40 absent an
    #     always-found system scores exactly 0.000 that way.
    #   * The briefing call instead said "F1 on the found flag", and a second
    #     slide softened it to "a system that never rejects cannot score well",
    #     both of which read naturally as *present*-as-positive, where the same
    #     always-found system scores 0.875.
    #
    # Reject-positive is therefore treated as the conservative planning number
    # -- it is the one that would hurt if we guessed wrong -- and the lenient
    # figure is printed alongside because it is what the earlier self-reported
    # 0.978 was measuring. Do not quote them as if they were the same metric.
    # The operating point is chosen by scripts/optimize_threshold.py against
    # the *total* rubric, which is what makes this ambiguity survivable: the
    # chosen threshold is near-optimal under either reading.
    def f1_at(t, positive="reject"):
        pf = (gray.score >= t).astype(int)              # 1 = we say "found"
        if positive == "reject":
            tp = int(((pf == 0) & (gray.gt_found == 0)).sum())   # correct reject
            fp = int(((pf == 0) & (gray.gt_found == 1)).sum())   # rejected a real one
            fn = int(((pf == 1) & (gray.gt_found == 0)).sum())   # missed an absent
        else:
            tp = int(((pf == 1) & (gray.gt_found == 1)).sum())
            fp = int(((pf == 1) & (gray.gt_found == 0)).sum())
            fn = int(((pf == 0) & (gray.gt_found == 1)).sum())
        return (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0), tp, fp, fn

    f1, tp, fp, fn = f1_at(threshold)
    f1_lenient = f1_at(threshold, "present")[0]
    res["rejection"] = (f1, 15 * f1)
    best = max(((f1_at(t)[0], t) for t in np.unique(gray.score.values)), default=(0, 0))

    # ---- Calibration (10 pts): AUC of score vs per-pair correctness --------
    # The task material says "AUC of your score column against per-pair
    # correctness" without defining correctness for ABSENT pairs (slide 6;
    # see ORGANIZER_PHASE2_GROUND_TRUTH.md G5). Two defensible readings:
    #   present-only: correct = present pair localised within 5 px (the
    #     reading this evaluator scored with historically);
    #   submitted: the submitted output itself is correct -- for a present
    #     pair that means we said found AND localised within 5 px, for an
    #     absent pair that means we said NOT found. A declined present pair
    #     forfeits its measurement and is NOT correct (issue #27;
    #     ORGANIZER_PHASE2_GROUND_TRUTH.md G5).
    # The primary figure stays present-only (comparable with every number
    # quoted before this change); the submitted-output variant is printed
    # alongside. Note the old `correct_all` variant (issue #27) labelled
    # every absent pair correct regardless of the submitted decision --
    # it was NOT a correct-rejection AUC and is not preserved.
    correct = np.where(gray.gt_found == 1, (gray.err <= 5).fillna(False), False)
    a, b = gray.score.values[correct], gray.score.values[~correct]
    auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()) \
        if len(a) and len(b) else float("nan")
    res["calibration"] = (auc, 10 * auc)
    # Submitted-output correctness (issue #27, G5): judge the decision
    # register.py would actually emit. pred_found is the same threshold
    # register.py applies, so a present pair whose score falls below it is
    # submitted as found=0 (declined) and is not correct here.
    pred_found = gray.score.values >= threshold
    correct_submitted = (
        ((gray.gt_found.values == 1) & pred_found & (gray.err.fillna(np.inf).values <= 5))
        | ((gray.gt_found.values == 0) & ~pred_found)
    )
    a3, b3 = gray.score.values[correct_submitted], gray.score.values[~correct_submitted]
    auc_submitted = float((a3[:, None] > b3[None, :]).mean() + 0.5 * (a3[:, None] == b3[None, :]).mean()) \
        if len(a3) and len(b3) else float("nan")
    res["calibration_submitted"] = (auc_submitted, 10 * auc_submitted)

    if quiet:
        return res, df

    comp = (f"{len(present)} present, {int((df.gt_found==0).sum())} absent "
            f"[unlabelled split: pooled, no A/B weighting]" if pooled else
            f"(A {int((df['set']=='A').sum())}, B {int((df['set']=='B').sum())}, "
            f"C {int((df['set']=='C').sum())}, D {int((df['set']=='D').sum())})")
    print(f"\n{'='*74}\n{label} — {len(df)} pairs {comp}\n{'='*74}")
    print(f"{'component':<28}{'metric':>26}{'points':>10}")
    print("-" * 74)
    print(f"{'Localisation (40)':<28}{f'credit {loc:.4f}':>26}{40*loc:>10.2f}")
    for s in (("A",) if pooled else ("A", "B")):
        p = present[present["set"] == s]
        print(f"{('   present' if pooled else '   set '+s)+f' ({len(p)} pairs)':<28}"
              f"{f'{parts[s]:.4f}  <=5px {100*(p.err<=5).mean():.1f}%  <=1px {100*(p.err<=1).mean():.1f}%':>26}")
        print(f"{'':<28}{f'median {p.err.median():.2f}px':>26}")
    print(f"{'Pose — scale (10)':<28}{f'credit {sc:.4f}  med {100*ok.s_err.median():.2f}%':>26}{10*sc:>10.2f}")
    print(f"{'Pose — rotation (10)':<28}{f'credit {rc:.4f}  med {ok.r_err.median():.3f}deg':>26}{10*rc:>10.2f}")
    print(f"{'Rejection (15)':<28}{f'F1(reject) {f1:.4f} @fix {threshold:.3f}':>26}{15*f1:>10.2f}")
    print(f"{'':<28}{f'correct-rej {tp}  lost-real {fp}  missed-abs {fn}':>26}")
    print(f"{'':<28}{f'[lenient F1(present) {f1_lenient:.4f}]':>26}")
    print(f"{'':<28}{f'[best-possible F1(reject) {best[0]:.4f} @ {best[1]:.4f}]':>26}")
    print(f"{'Calibration (10)':<28}{f'AUC {auc:.4f}':>26}{10*auc:>10.2f}")
    print(f"{'   [submitted AUC (issue #27)':<28}{f'{auc_submitted:.4f} -> {10*auc_submitted:.2f}]':>26}")
    print("-" * 74)
    # The submitted AUC is a printed ALTERNATIVE to the primary calibration
    # number, not an additive component -- summing both res entries would
    # double-count the 10 calibration points (issue #27 side-fix; the same
    # bug shipped with PR #25's calibration_all_pairs).
    sub = sum(v[1] for k, v in res.items() if k != "calibration_submitted")
    print(f"{'SUBTOTAL (85 measurable)':<28}{'':>26}{sub:>10.2f}")
    print(f"{'  + efficiency (5)':<28}{'not measurable in parallel':>26}")
    print(f"{'  + generator/report (10)':<28}{'judged, not self-assessable':>26}")

    if (df["set"] == "D").any():
        # Set D reads the SAME submission-masked loc_credit as A/B/C: the
        # mask was applied to the full frame above, so a declined present-D
        # pair scores 0 credit here exactly as it would on the submitted CSV.
        d = df[df["set"] == "D"]
        dp = d[d.gt_found == 1]
        dc = dp.loc_credit.mean()
        print(f"\n{'BONUS set D (optical)':<28}{f'credit {dc:.4f}  <=5px {100*(dp.err<=5).mean():.1f}%':>26}")
        print(f"{'':<28}{'+6 needs D>=0.40 and A-C>=0.50':>26}")
    return res, df

