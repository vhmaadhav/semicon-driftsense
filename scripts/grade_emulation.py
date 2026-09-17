#!/usr/bin/env python3
"""Stratified blind-grade emulation for the Phase 2 bonus probability.

PR #24 review blocker 4: eval_ext.py --sample 200 draws an *unstratified*
sample, but the disclosed blind-grade composition is stratified --
A=70, B=70, C=40 (set D=20 is excluded from the grayscale grade; the
rejection F1 runs over exactly the 180 A/B/C pairs). This module reproduces
that composition exactly and runs a stratified bootstrap over a per-pair
results CSV to answer one planning question: how likely is the +4 bonus
(F1_reject >= 0.90 on the 180-pair grade, reject-positive) at a
given found threshold?

Rubric (corrected semantics, identical credit tiers to scripts/eval_ext.py
-> score()):

* localisation: 1 / 0.8 / 0.6 / 0.4 credit at <=1 / 2 / 3 / 5 px, weighted
  0.45 * set A + 0.55 * set B, ZERO for a present pair the system declined
  (register.py writes no pose/location fields for a declined answer);
* pose (scale <=1/2/5% and rotation <=0.25/0.5/1.0 deg, credits 1/0.6/0.3),
  scored only where localisation earned credit;
* rejection F1 (REJECT-POSITIVE: a correctly declined absent pair is the
  true positive) at found = score >= t over the 180 grayscale pairs. This
  is the Phase 2 scoring convention -- the official slide statement that
  "a system that never rejects scores zero" only holds under
  reject-positive, and eval_ext.py / the promoted 0.9078 are
  reject-positive;
* calibration AUC, present-only (correct = present pair localised <=5 px).

Bootstrap: N stratified draws (draw i uses independent per-set streams at seed + i so
draw 0 equals a standalone stratified_draw call with the same seed), the
rubric scored on each draw, and

    P(bonus) = P(F1_reject >= 0.90)   over draws (reject-positive F1)
    E[total + 4*P(bonus)] = E[total] + 4 * P(bonus)

where total is the 85-point measurable subtotal (loc 40 + pose 20 +
rejection 15 + calibration 10).

Dependencies: stdlib + numpy + pandas only (torch-free on purpose -- this
runs offline on per-pair CSVs, never on images).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Published Phase 2 credit tiers (kept in lockstep with scripts/eval_ext.py).
LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))
SCALE_TIERS = ((0.01, 1.00), (0.02, 0.60), (0.05, 0.30))
ROT_TIERS = ((0.25, 1.00), (0.50, 0.60), (1.00, 0.30))
W_A, W_B = 0.45, 0.55

# Disclosed blind-grade composition. Set D (20 pairs, optical) is excluded
# from the grayscale grade; the rejection F1 is over exactly these 180 pairs.
BLIND_COMPOSITION = {"A": 70, "B": 70, "C": 40}
BONUS_F1 = 0.90          # the +4 bonus gate used for planning:
                         # F1_reject >= 0.90 (reject-positive F1)
BONUS_WEIGHT = 4.0       # F1 bonus is four points; Set D bonus is separate
# The four undisclosed Set B severity levels (slide 4). G2 warns the real
# blind set shifts B toward 3-4 while our pool is uniform over the four, so a
# draw can be constrained to a severity MIXTURE instead of drawing B
# uniformly -- see severity_quotas() and --b-mix.
SEVERITY_LEVELS = (1, 2, 3, 4)
DEFAULT_THRESHOLDS = (0.1587, 0.18, 0.25)

REQUIRED_COLUMNS = (
    "pair_id", "set", "gt_found", "score",
    "x", "y", "gt_x", "gt_y",
    "scale", "theta", "gt_scale", "gt_rot",
)


def tier(value, tiers):
    """Credit for value under tiers ((bound, credit), ...): the credit of
    the first bound the value does not exceed; 0.0 past the last bound."""
    for bound, credit in tiers:
        if value <= bound:
            return credit
    return 0.0


# ---------------------------------------------------------------------------
# Stratified draw
# ---------------------------------------------------------------------------

def _stratum_rng(seed, set_name):
    """Stable independent stream per named set; mixtures cannot perturb A/C.

    Reusing RandomState(seed) for every set selects identical positions in
    equal-sized pools, coupling the simulated errors across A and B.
    """
    sequence = np.random.SeedSequence([seed, *set_name.encode("utf-8")])
    return np.random.RandomState(sequence.generate_state(1)[0])


def stratified_draw(df, quotas=None, seed=0):
    """Exact stratified sample: independent per-set seeded permutations,
    take exactly the quota, union. Order-stable (rows keep the
    original frame order).

    Raises ValueError (clearly) if a set is smaller than its quota.
    """
    quotas = dict(BLIND_COMPOSITION if quotas is None else quotas)
    parts = []
    for set_name in sorted(quotas):
        quota = quotas[set_name]
        sub = df[df["set"] == set_name]
        if len(sub) < quota:
            raise ValueError(
                f"set {set_name!r} has {len(sub)} rows but the blind-grade "
                f"quota needs {quota}; cannot draw an exact stratified "
                f"sample from this frame"
            )
        rng = _stratum_rng(seed, set_name)
        idx = rng.permutation(len(sub))[:quota]
        parts.append(sub.iloc[np.sort(idx)])
    out = pd.concat(parts)
    # Order-stable union: restore the original frame order.
    return out.iloc[np.argsort(out.index.values, kind="stable")]


def severity_quotas(spec, quota, levels=SEVERITY_LEVELS):
    """Turn a mixture spec into exact per-level counts summing to `quota`.

    `spec` is one number per severity level. If the numbers already sum to
    `quota` they are used as counts; otherwise they are read as relative
    weights and apportioned by largest remainder, which is the only way to
    hit an exact integer quota without silently rounding the mixture away.

        severity_quotas("10,20,35,35", 70) -> {1: 7, 2: 14, 3: 24, 4: 25}

    The apportioned counts are what callers must REPORT: a mixture that was
    asked for is not evidence of a mixture that was drawn (the same class of
    error as the --severity-range pin trap, see
    tests/test_severity_pin_guard.py).
    """
    if isinstance(spec, dict):
        w = [float(spec.get(l, 0.0)) for l in levels]
    else:
        parts = [x for x in str(spec).replace(":", ",").split(",") if x.strip()]
        if len(parts) != len(levels):
            raise ValueError(f"severity mix needs {len(levels)} numbers "
                             f"(levels {levels}), got {len(parts)}: {spec!r}")
        w = [float(x) for x in parts]
    if any(v < 0 for v in w):
        raise ValueError(f"severity mix has a negative weight: {spec!r}")
    total = sum(w)
    if total <= 0:
        raise ValueError(f"severity mix sums to zero: {spec!r}")
    if all(float(v).is_integer() for v in w) and int(total) == int(quota):
        return {l: int(v) for l, v in zip(levels, w)}
    exact = [quota * v / total for v in w]
    counts = [int(np.floor(e)) for e in exact]
    # Largest remainder, ties to the HIGHER severity level. With four equal
    # shares of 70 every remainder is 0.5, and handing those seats to the
    # lower levels (the natural stable-sort order) makes an evenly-split
    # mixture draw 18,18,17,17 -- 48.6% at severity 3-4 when 50% was asked
    # for, biased toward the easier half on every tie. Callers report the
    # realised fraction regardless; this just stops the bias being built in.
    for i in sorted(range(len(exact)),
                    key=lambda i: (exact[i] - counts[i], i),
                    reverse=True)[:quota - sum(counts)]:
        counts[i] += 1
    return {l: c for l, c in zip(levels, counts)}


def _blocks_for(sets, severity, set_name, quota, mix):
    """Row-index blocks and per-block quotas for one set.

    Without a mixture this is a single block (the whole set) -- the historical
    behaviour, preserved exactly. With one it is one block per severity level.
    """
    if mix is None:
        block = np.flatnonzero(sets == set_name)
        if len(block) < quota:
            raise ValueError(f"set {set_name!r} has {len(block)} rows but the "
                             f"quota needs {quota}; cannot draw this frame")
        return [(block, quota)]
    if severity is None:
        raise ValueError("a severity mixture was requested but the CSV has no "
                         "`severity` column to draw it from")
    out = []
    for level, want in severity_quotas(mix, quota).items():
        block = np.flatnonzero((sets == set_name) & (severity == level))
        if len(block) < want:
            raise ValueError(
                f"set {set_name} severity {level} has {len(block)} rows but the "
                f"mixture needs {want}; cannot draw this mixture from this frame")
        out.append((block, want))
    return out


# ---------------------------------------------------------------------------
# Rubric (corrected semantics) on a grayscale A/B/C frame
# ---------------------------------------------------------------------------

def _prepare(df):
    """Precompute per-pair arrays the rubric needs (NaN-safe)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    gt_found = df["gt_found"].to_numpy().astype(int)
    score = df["score"].to_numpy().astype(float)
    err = np.where(
        gt_found == 1,
        np.hypot(df["x"].to_numpy() - df["gt_x"].to_numpy(),
                 df["y"].to_numpy() - df["gt_y"].to_numpy()),
        np.nan,
    )
    raw_loc = np.where(
        np.isfinite(err),
        np.array([tier(e, LOC_TIERS) for e in err], dtype=float),
        0.0,
    )
    s_err = np.abs(df["scale"].to_numpy() - df["gt_scale"].to_numpy()) / df["gt_scale"].to_numpy()
    r_err = np.abs(df["theta"].to_numpy() - df["gt_rot"].to_numpy())
    raw_sc = np.array([tier(v, SCALE_TIERS) for v in s_err], dtype=float)
    raw_rc = np.array([tier(v, ROT_TIERS) for v in r_err], dtype=float)
    sets = df["set"].to_numpy()
    return {
        "gt_found": gt_found, "score": score, "err": err,
        "raw_loc": raw_loc, "raw_sc": raw_sc, "raw_rc": raw_rc,
        "isA": sets == "A", "isB": sets == "B",
    }


def _f1(score, gt, t, positive):
    """F1 at found = score >= t; positive='present' or 'reject'."""
    pf = score >= t
    if positive == "reject":
        tp = int(((pf == 0) & (gt == 0)).sum())   # correct reject
        fp = int(((pf == 0) & (gt == 1)).sum())   # rejected a real one
        fn = int(((pf == 1) & (gt == 0)).sum())   # missed an absent
    else:
        tp = int(((pf == 1) & (gt == 1)).sum())
        fp = int(((pf == 1) & (gt == 0)).sum())
        fn = int(((pf == 0) & (gt == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def f1_found(gray, t):
    """F1 on the found flag (present-as-positive) at found = score >= t."""
    return _f1(gray["score"].to_numpy().astype(float),
               gray["gt_found"].to_numpy().astype(int), t, "present")


def f1_reject(gray, t):
    """Reject-as-positive F1 -- the Phase 2 scoring convention (the official
    slide statement that a system that never rejects scores zero only holds
    under reject-positive; eval_ext.py and the promoted 0.9078 are
    reject-positive). f1_found (present-positive) is reported alongside as a
    labelled diagnostic only; the two are NOT the same metric and must not
    be quoted as one."""
    return _f1(gray["score"].to_numpy().astype(float),
               gray["gt_found"].to_numpy().astype(int), t, "reject")


def rubric(gray, t):
    """Corrected-semantics rubric on a grayscale (A/B/C) frame at threshold t."""
    p = _prepare(gray)
    gt, score, err = p["gt_found"], p["score"], p["err"]
    said = score >= t
    present = gt == 1
    loc_credit = np.where(present & said, p["raw_loc"], 0.0)

    res = {}
    parts = {}
    for s, mask in (("A", p["isA"]), ("B", p["isB"])):
        sel = mask & present
        parts[s] = float(loc_credit[sel].mean()) if sel.any() else float("nan")
    loc = W_A * parts["A"] + W_B * parts["B"]
    res["loc_A"], res["loc_B"], res["loc"] = parts["A"], parts["B"], loc

    ok = present & (loc_credit > 0)
    res["scale"] = float(p["raw_sc"][ok].mean()) if ok.any() else float("nan")
    res["rot"] = float(p["raw_rc"][ok].mean()) if ok.any() else float("nan")

    res["f1_found"] = _f1(score, gt, t, "present")
    res["f1_reject"] = _f1(score, gt, t, "reject")

    correct = present & (err <= 5)
    a, b = score[correct], score[~correct]
    res["auc"] = (float((a[:, None] > b[None, :]).mean()
                        + 0.5 * (a[:, None] == b[None, :]).mean())
                  if len(a) and len(b) else float("nan"))

    total = (40 * loc + 10 * res["scale"] + 10 * res["rot"]
             + 15 * res["f1_reject"] + 10 * res["auc"])
    res["total"] = float(total) if np.isfinite(total) else float("nan")
    res["n"] = len(gray)
    res["n_present"] = int(present.sum())
    return res


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(df, thresholds=None, draws=10000, seed=0, mixes=None):
    """Stratified bootstrap: draws exact-composition blind grades; per
    threshold, P(F1_reject >= 0.90) over draws (reject-positive F1 -- the
    Phase 2 scoring convention) and E[total] + 4*P(bonus).

    Draw i uses independent, named per-set random streams at seed + i,
    so the draw sequence is deterministic in (seed, draws, frame order) and
    shared across thresholds (which draw is taken does not depend on t).

    `mixes` optionally constrains a set's draw to a severity mixture, e.g.
    {"B": "10,20,35,35"} to emulate the severity-3/4-weighted Set B the jury
    says the real blind set uses (G2). A set with no entry is drawn
    uniformly. Named streams keep A/C unchanged when only B's mixture varies.
    Draws intentionally differ from the former coupled-stream implementation.
    """
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)
    gray = df[df["set"].isin(BLIND_COMPOSITION)]
    q = BLIND_COMPOSITION
    p = _prepare(gray)
    sets = gray["set"].to_numpy()
    mixes = dict(mixes or {})
    severity = (gray["severity"].to_numpy()
                if "severity" in gray.columns else None)
    # Resolved once: a mixture that cannot be drawn from this pool must fail
    # here, before 10,000 draws, not silently short a level.
    blocks = {s: _blocks_for(sets, severity, s, q[s], mixes.get(s))
              for s in sorted(q)}

    totals = {t: [] for t in thresholds}
    f1s = {t: [] for t in thresholds}

    score_all, gt_all = p["score"], p["gt_found"]
    for i in range(draws):
        idx_parts = []
        for s in sorted(q):
            rng = _stratum_rng(seed + i, s)
            for block, want in blocks[s]:
                idx_parts.append(block[np.sort(rng.permutation(len(block))[:want])])
        idx = np.concatenate(idx_parts)
        sc, gt = score_all[idx], gt_all[idx]
        present = gt == 1
        err_d = p["err"][idx]
        # Positional credit for a present pair; the threshold mask (did the
        # system decline it?) is applied per threshold below.
        loc_credit_all = np.where(present & np.isfinite(err_d),
                                  p["raw_loc"][idx], 0.0)
        isA, isB = p["isA"][idx], p["isB"][idx]
        correct = present & (err_d <= 5)
        for t in thresholds:
            said = sc >= t
            loc_credit = np.where(said, loc_credit_all, 0.0)
            selA, selB = isA & present, isB & present
            locA = float(loc_credit[selA].mean()) if selA.any() else float("nan")
            locB = float(loc_credit[selB].mean()) if selB.any() else float("nan")
            loc = W_A * locA + W_B * locB
            okm = present & (loc_credit > 0)
            sc_credit = float(p["raw_sc"][idx][okm].mean()) if okm.any() else float("nan")
            rc_credit = float(p["raw_rc"][idx][okm].mean()) if okm.any() else float("nan")
            f1 = _f1(sc, gt, t, "reject")   # bonus gate: reject-positive F1
            a, b = sc[correct], sc[~correct]
            auc = (float((a[:, None] > b[None, :]).mean()
                         + 0.5 * (a[:, None] == b[None, :]).mean())
                   if len(a) and len(b) else float("nan"))
            total = (40 * loc + 10 * sc_credit + 10 * rc_credit
                     + 15 * f1 + 10 * auc)
            totals[t].append(total)
            f1s[t].append(f1)

    out = []
    for t in thresholds:
        tot = np.asarray(totals[t], dtype=float)
        f1 = np.asarray(f1s[t], dtype=float)
        p_bonus = float((f1 >= BONUS_F1).mean())
        e_total = float(np.nanmean(tot)) if np.isfinite(tot).any() else float("nan")
        # Monte-Carlo standard error on the gate rate: a P(bonus) quoted
        # without it invites reading a 2-point difference between two mixtures
        # as a result when N draws cannot resolve it.
        out.append({
            "threshold": t,
            "p_f1_ge_bonus": p_bonus,
            "p_bonus_mcse": float(np.sqrt(p_bonus * (1 - p_bonus) / draws)),
            "e_total": e_total,
            "e_total_plus_bonus": (e_total + BONUS_WEIGHT * p_bonus
                                   if np.isfinite(e_total) else float("nan")),
            "mean_f1": float(np.mean(f1)),
            "sd_f1": float(np.std(f1, ddof=1)) if draws > 1 else float("nan"),
            "f1_p2.5": float(np.percentile(f1, 2.5)),
            "f1_p50": float(np.percentile(f1, 50)),
            "f1_p97.5": float(np.percentile(f1, 97.5)),
            "total_p2.5": float(np.nanpercentile(tot, 2.5)),
            "total_p97.5": float(np.nanpercentile(tot, 97.5)),
            "draws": draws,
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def sweep_mix(frac_34, quota=None):
    """Exact counts putting `frac_34` of Set B's draw at severity 3-4, split
    as evenly as an integer quota allows inside each half.

    Counts rather than weights on purpose: four equal weights over 70 seats
    leave every largest-remainder tie at 0.5, and whichever way those ties
    break, the realised severity-3/4 fraction misses the requested one by a
    pair. Splitting the halves first hits the fraction exactly, which is the
    only thing this sweep varies. frac_34 = 0.50 reproduces the uniform pool.
    """
    quota = BLIND_COMPOSITION["B"] if quota is None else quota
    n34 = int(round(frac_34 * quota))
    n12 = quota - n34
    return f"{n12 // 2},{n12 - n12 // 2},{n34 // 2},{n34 - n34 // 2}"


SWEEP_FRACTIONS = (0.50, 0.65, 0.75, 0.85, 1.00)


def _severity_sweep(df, gray, threshold, draws, seed, c_mix=None):
    """Tabulate the bonus gate against Set B's severity-3/4 fraction.

    The point of a sweep rather than one guessed mixture: the jury materials
    say the real set shifts B toward severity 3-4 but never say by how much,
    so the honest answer is the sensitivity curve, not a single number
    conditioned on a proportion nobody disclosed.
    """
    print(f"Severity sweep at t = {threshold:.4f}"
          + (f", Set C mixture {c_mix}" if c_mix else ", Set C drawn uniformly"))
    print()
    print(f"{'B sev 3-4':>10}{'mix (L1,L2,L3,L4)':>22}{'mean F1':>10}{'sd':>8}"
          f"{'95% draw interval':>18}{'P(F1>=.90)':>12}{'+-mcse':>8}{'E[total]':>10}")
    print("-" * 98)
    rows = []
    for frac in SWEEP_FRACTIONS:
        mixes = {"B": sweep_mix(frac)}
        if c_mix:
            mixes["C"] = c_mix
        counts = severity_quotas(mixes["B"], BLIND_COMPOSITION["B"])
        bs = bootstrap(df, thresholds=[threshold], draws=draws, seed=seed,
                       mixes=mixes)[0]
        rows.append((frac, counts, bs))
        lo, hi = bs["f1_p2.5"], bs["f1_p97.5"]
        mix_str = ",".join(str(counts[l]) for l in SEVERITY_LEVELS)
        ci = f"[{lo:.3f}, {hi:.3f}]"
        # The REALISED fraction the draw uses, not the one requested: an
        # integer quota cannot always hit the requested proportion, and the
        # requested one is the number nobody should be quoting.
        realised = (counts[3] + counts[4]) / BLIND_COMPOSITION["B"]
        print(f"{100*realised:>9.1f}%{mix_str:>22}"
              f"{bs['mean_f1']:>10.4f}{bs['sd_f1']:>8.4f}{ci:>18}"
              f"{bs['p_f1_ge_bonus']:>12.3f}{bs['p_bonus_mcse']:>8.3f}"
              f"{bs['e_total']:>10.2f}")
    print("-" * 98)
    base, worst = rows[0][2], rows[-1][2]
    print(f"gate rate {base['p_f1_ge_bonus']:.3f} (pool as-is) -> "
          f"{worst['p_f1_ge_bonus']:.3f} (all B at severity 3-4); "
          f"mean F1 {base['mean_f1']:.4f} -> {worst['mean_f1']:.4f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stratified blind-grade emulation (A=70 B=70 C=40; "
                    "F1 over the 180 grayscale pairs) and bonus-probability "
                    "bootstrap over a per-pair results CSV.")
    ap.add_argument("--csv", required=True, help="per-pair results CSV "
                    "(columns: pair_id, set, gt_found, score, x, y, gt_x, "
                    "gt_y, scale, theta, gt_scale, gt_rot)")
    ap.add_argument("--threshold",
                    default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                    help="comma-separated found thresholds "
                         "(default 0.1587,0.18,0.25)")
    ap.add_argument("--draws", type=int, default=10000,
                    help="bootstrap draws (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b-mix", default=None,
                    help="severity mixture for Set B's 70 drawn pairs, four "
                         "numbers over levels 1,2,3,4 (counts if they sum to "
                         "70, otherwise relative weights). G2: the jury says "
                         "the real blind set weights B toward 3-4, while our "
                         "pool is uniform. Example: --b-mix 10,15,35,40")
    ap.add_argument("--c-mix", default=None,
                    help="severity mixture for Set C's 40 drawn pairs. Set C "
                         "carries realised severity too, and the rejection F1 "
                         "is half decided by it, so B-only shifts are a "
                         "lower bound on the severity cost")
    ap.add_argument("--severity-sweep", action="store_true",
                    help="sweep Set B's severity-3/4 fraction (0.50 = the "
                         "pool as-is, through 1.00) at the first --threshold "
                         "and tabulate the bonus gate against it, instead of "
                         "reporting one mixture")
    a = ap.parse_args(argv)

    thresholds = [float(x) for x in str(a.threshold).split(",") if x.strip()]
    df = pd.read_csv(a.csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        ap.error(f"CSV is missing required columns: {missing}")

    gray = df[df["set"].isin(BLIND_COMPOSITION)]
    print(f"CSV: {a.csv}  ({len(df)} pairs; grayscale A/B/C frame = {len(gray)}: "
          + ", ".join(f"{s}={int((gray['set'] == s).sum())}"
                      for s in sorted(BLIND_COMPOSITION)) + ")")
    print("Blind composition: " + ", ".join(f"{s}={q}"
          for s, q in sorted(BLIND_COMPOSITION.items()))
          + f"  (D excluded; reject-positive F1 over "
          f"{sum(BLIND_COMPOSITION.values())} pairs)")
    print(f"Bootstrap: {a.draws} stratified draws, seed {a.seed}, "
          f"bonus gate F1_reject >= {BONUS_F1}")

    if a.severity_sweep:
        return _severity_sweep(df, gray, thresholds[0], a.draws, a.seed,
                               c_mix=a.c_mix)

    mixes = {s: m for s, m in (("B", a.b_mix), ("C", a.c_mix)) if m}
    if mixes:
        # Report the REALISED per-level counts, not the mixture as typed: an
        # apportioned mixture is not the mixture asked for, and the pool's own
        # severity has to be checked rather than assumed (G2 pin-trap lesson).
        for set_name, mix in sorted(mixes.items()):
            counts = severity_quotas(mix, BLIND_COMPOSITION[set_name])
            pool = {l: int(((gray["set"] == set_name)
                            & (gray["severity"] == l)).sum())
                    for l in SEVERITY_LEVELS}
            print(f"Set {set_name} severity mixture: "
                  + " ".join(f"L{l}={counts[l]}" for l in SEVERITY_LEVELS)
                  + f"  ({100*sum(counts[l] for l in (3, 4))/BLIND_COMPOSITION[set_name]:.0f}%"
                  f" at severity 3-4; pool has "
                  + " ".join(f"L{l}={pool[l]}" for l in SEVERITY_LEVELS) + ")")
    else:
        print("Set B severity mixture: none (drawn uniformly from the pool)")
    print()

    # F1_found / F1_rej are FULL-FRAME point estimates over every grayscale
    # pair in the CSV: they do not respond to --b-mix/--c-mix, because the
    # mixture describes what gets DRAWN, not what the pool contains. The
    # mixture-dependent figures are the draw mean and everything right of it.
    # Printing them side by side without saying so reads as "the mixture did
    # not change F1", which is the opposite of what the draws show.
    header = f"{'t':>8}{'F1_found*':>10}{'F1_rej*':>10}{'mean F1':>9}" \
             f"{'P(F1_rej>=.90)':>15}{'E[total]':>10}{'E[tot]+4P':>11}"
    print(header)
    print(f"{'':>8}{'(* full-frame point estimates, mixture-independent)':<48}")
    print("-" * 73)
    # One bootstrap pass for ALL thresholds: the stratified draws do not
    # depend on t, so the draw sequence is shared (and computed once).
    bs_all = bootstrap(df, thresholds=thresholds, draws=a.draws, seed=a.seed,
                       mixes=mixes)
    results = []
    for t, bs in zip(thresholds, bs_all):
        f1f, f1r = f1_found(gray, t), f1_reject(gray, t)
        results.append((t, f1f, f1r, bs))
        print(f"{t:>8.4f}{f1f:>10.4f}{f1r:>10.4f}{bs['mean_f1']:>9.4f}"
              f"{bs['p_f1_ge_bonus']:>15.4f}{bs['e_total']:>10.2f}"
              f"{bs['e_total_plus_bonus']:>11.2f}")
    print("-" * 73)
    best = max(results, key=lambda r: r[3]["e_total_plus_bonus"])
    print(f"argmax E[total + 4*P(bonus)]: t = {best[0]:.4f} "
          f"(E = {best[3]['e_total_plus_bonus']:.2f})")
    print()
    print("Caveat: draws sample without replacement from this fixed CSV pool. "
          "Intervals describe simulated 180-pair grades, not confidence "
          "intervals for the population mean or guarantees on a new generator. "
          "The +4 gate probability is conditional on this pool and mixture; "
          "the separate Set D bonus is not included.")
    return results


if __name__ == "__main__":
    main()
