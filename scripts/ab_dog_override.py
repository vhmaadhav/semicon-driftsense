#!/usr/bin/env python3
"""Sweep the `dog-override` epsilon in ONE decode pass.

`verification="dog-override"` keeps the shipped native-ZNCC winner unless two
things are true at once: the DoG score prefers a *different* hypothesis, and
the ZNCC margin between those two candidates is below epsilon -- i.e. ZNCC
itself is not confident. Epsilon is the only free parameter, and the honest way
to choose it is to measure several values.

**The point of this script is that the network runs once, not once per
epsilon.** `locate_phase2(..., return_hypotheses=True)` attaches every
candidate's `zncc`, `dog`, `rank` and `band` to the result, and the override
rule is a two-line comparison over those numbers. So the decode -- the
expensive part, minutes per hundred pairs -- happens once, and an N-value
sweep is replayed offline in pure Python. A five-epsilon sweep run the naive
way would decode the set five times for five answers that share every input.

    # stride-5 pre-read on Set B (the only set expected to move)
    python scripts/ab_dog_override.py <B shards> --stride 5 --out .agents/dogov.csv

    # re-sweep different epsilons with no GPU/CPU cost at all
    python scripts/ab_dog_override.py --replay .agents/dogov.csv \\
        --margins 0.01 0.03 0.05 0.08

What is reported, per epsilon and broken out per phase2 set: how many pairs the
decision CHANGED, how many were RECOVERED (error crossed >5 px -> <=5 px), how
many were BROKEN (<=5 px -> >5 px), the net, and the change in mean
localisation error. Set A already localises at ~0.95 credit and Set C is
absent-heavy, so a healthy result is "Set B moves, the others do not"; an
epsilon that churns Set A is buying Set B points with Set A points.

Two measurement rules this script enforces rather than trusts:

* **Stride must be odd.** The manifests order pairs by a repeating severity
  cycle [1,2,3,4,...], so an even stride samples a different *population*, not
  a subsample -- `--stride 4` reads Set B at 95.4% within 5 px where the full
  set reads 81.4%. Two experiments were invalidated this way before it was
  spotted (see `.agents/PHASE2_STATE.md`). An even `--stride` aborts here.
* **Every candidate's position is measured on the same footing.** The decode
  mutates the winning candidate dict in place after selection, so with the
  shipped `subpixel_rows=True` the winner's recorded `x` is post-refinement
  while every loser's is not -- which would quietly hand the incumbent a head
  start in exactly the comparison this script exists to make. The decode here
  therefore runs `subpixel_rows=False`, `refit_xy=False`, and the surviving
  polish stage touches only scale/theta. `--subpixel-rows` opts back in and
  says so loudly.

Because of that second rule, and because polish/sub-pixel refinement run
*after* selection in the real decode, the +-5 px verdicts here are a
**selection proxy**, on the same footing as `scripts/verify_scores.py`. Use it
to rank epsilons cheaply; promote nothing on it. The winning epsilon still owes
a full-set `scripts/eval_ext.py --verification dog-override
--dog-override-margin <eps>` A/B against the incumbent, and the standing
promotion gate is a paired delta of >= +0.35 rubric points (~2 sigma).

Per-pair timings collected here are under process parallelism and are NOT
valid for the efficiency component; measure that with scripts/profile_pair.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.config import SHIPPED_BAND, DOG_OVERRIDE_MARGIN  # noqa: E402

# The rubric's full-credit localisation boundary and the line every "recovered"
# / "broken" count in this file is drawn at.
ERR_TOL_PX = 5.0

DEFAULT_MARGINS = (0.02, 0.05, 0.10, 0.15, 0.20)

# Columns written per CANDIDATE. One row per (pair, hypothesis).
CAND_FIELDS = [
    "pair_id", "set", "severity", "architecture", "gt_found",
    "cand", "n_cands", "x", "y", "err", "coarse_x_native", "coarse_y_native",
    "zncc", "dog", "rank", "band", "network_score", "pose_peak", "peak_ratio",
    "scale", "theta", "winner_zncc", "secs",
]


def _worker(job):
    """Decode one pair and return its candidate list. Imports are inside so
    each process caps its own threads before torch/cv2 build their pools."""
    (shard_dir, row, weights, threads, hypotheses, coarse, band,
     subpixel_rows) = job
    import torch
    torch.set_num_threads(threads)
    import cv2
    # OpenCV otherwise spawns a pool per *process*, so N workers oversubscribe
    # the machine N-fold and every per-pair timing becomes meaningless.
    cv2.setNumThreads(threads)
    import infer as I
    from driftsense.matching import locate_phase2

    global _MODEL
    try:
        model, device = _MODEL
    except NameError:
        _MODEL = I.load_model(weights)
        model, device = _MODEL

    ref = I.read_gray(os.path.join(shard_dir, row["reference_path"]))
    sea = I.read_gray(os.path.join(shard_dir, row["search_path"]))
    t0 = time.perf_counter()
    # verification="zncc" on purpose: the INCUMBENT decode is the reference
    # frame for every number below, and the override is replayed offline.
    # return_hypotheses=True is what makes `dog` exist on each candidate.
    res = locate_phase2(model, ref, sea, device, refine=True,
                        hypotheses=hypotheses, coarse_scales=coarse,
                        band=band, verification="zncc",
                        subpixel_rows=subpixel_rows,
                        refit_xy=False,
                        return_hypotheses=True)
    dt = time.perf_counter() - t0

    gx, gy = float(row["gt_x_corr"]), float(row["gt_y_corr"])
    hyps = res.get("hypotheses", [])
    out = []
    for j, c in enumerate(hyps):
        out.append({
            "pair_id": row["pair_id"], "set": row["phase2_set"],
            "severity": row.get("severity_level", -1),
            "architecture": row.get("architecture", ""),
            "gt_found": int(row["found"]),
            "cand": j, "n_cands": len(hyps),
            "x": c["x"], "y": c["y"],
            "err": float(np.hypot(c["x"] - gx, c["y"] - gy)),
            "coarse_x_native": c.get("coarse_x_native", np.nan),
            "coarse_y_native": c.get("coarse_y_native", np.nan),
            "zncc": c.get("zncc", np.nan), "dog": c.get("dog", np.nan),
            "rank": c.get("rank", np.nan), "band": c.get("band", np.nan),
            "network_score": c.get("network_score", np.nan),
            "pose_peak": c.get("pose_peak", np.nan),
            "peak_ratio": c.get("peak_ratio", np.nan),
            "scale": c.get("scale", np.nan), "theta": c.get("theta", np.nan),
            # The zncc of the hypothesis the DECODE actually selected. The
            # offline replay of the incumbent rule must reproduce it; see
            # _check_replay_matches_decode.
            "winner_zncc": float(res.get("zncc", np.nan)),
            "secs": dt,
        })
    return out


# ---------------------------------------------------------------------------
# offline replay -- no images, no network, pure arithmetic over the CSV
# ---------------------------------------------------------------------------

def _zncc_of(c: dict) -> float:
    """The selection statistic, exactly as locate_phase2's choose() reads it:
    native zncc, falling back to the network score when zncc is absent (the
    refine=False path), and to -inf when neither is finite."""
    for k in ("zncc", "network_score"):
        v = c.get(k, np.nan)
        if v is not None and np.isfinite(v):
            return float(v)
    return -np.inf


def _dog_of(c: dict) -> float:
    v = c.get("dog", np.nan)
    return float(v) if v is not None and np.isfinite(v) else -np.inf


def _argmax(cands: list[dict], key) -> int:
    """First index attaining the max -- the same tie-break as choose()'s
    `max(range(n), key=...)` and as np.argmax."""
    return max(range(len(cands)), key=lambda i: key(cands[i]))


def decide(cands: list[dict], eps: float) -> int:
    """The dog-override rule. Native ZNCC picks the winner; the DoG pick
    overrides it only when the two disagree AND the ZNCC margin between those
    two candidates is below eps (ZNCC is not confident).

    eps <= 0 can never fire -- the margin is non-negative by construction --
    so the selector degrades exactly to `zncc`, which is what makes eps=0 a
    valid no-op control rather than a separate code path."""
    zncc_i = _argmax(cands, _zncc_of)
    if len(cands) == 1:
        return zncc_i
    dog_i = _argmax(cands, _dog_of)
    if dog_i == zncc_i:
        return zncc_i
    margin = _zncc_of(cands[zncc_i]) - _zncc_of(cands[dog_i])
    return dog_i if margin < eps else zncc_i


def group_pairs(df: pd.DataFrame) -> list[dict]:
    """CSV rows -> one record per pair, candidates ordered by `cand`."""
    pairs = []
    for pid, g in df.groupby("pair_id", sort=True):
        g = g.sort_values("cand")
        pairs.append({
            "pair_id": pid,
            "set": g["set"].iloc[0],
            "gt_found": int(g["gt_found"].iloc[0]),
            "winner_zncc": float(g["winner_zncc"].iloc[0]),
            "cands": g.to_dict("records"),
        })
    return pairs


def _check_replay_matches_decode(pairs: list[dict]) -> int:
    """Does the offline incumbent rule pick what the decode picked?

    If this disagrees the replay is measuring a rule the pipeline does not
    run, and every number below is void -- so it is checked rather than
    assumed. Compared on the winner's zncc value (the decode returns the
    winning candidate's dict, and zncc is not mutated after selection)."""
    bad = 0
    for p in pairs:
        if not np.isfinite(p["winner_zncc"]):
            continue
        mine = _zncc_of(p["cands"][_argmax(p["cands"], _zncc_of)])
        if not np.isclose(mine, p["winner_zncc"], rtol=0, atol=1e-9):
            bad += 1
    return bad


def report(df: pd.DataFrame, margins, err_tol: float = ERR_TOL_PX) -> None:
    pairs = group_pairs(df)
    # Localisation is only scored on present pairs, and an absent pair has no
    # ground-truth position to measure an error against.
    pairs = [p for p in pairs if p["gt_found"] == 1]
    if not pairs:
        print("no present pairs in the input -- nothing to score")
        return

    bad = _check_replay_matches_decode(pairs)
    if bad:
        print(f"\n!! WARNING: the offline incumbent rule disagrees with the "
              f"decode's own winner on {bad}/{len(pairs)} pairs. The replay "
              f"is not modelling the shipped selector; treat every number "
              f"below as void until that is explained.\n")

    sets = sorted({p["set"] for p in pairs})
    base_pick = {p["pair_id"]: _argmax(p["cands"], _zncc_of) for p in pairs}
    base_err = {p["pair_id"]: p["cands"][base_pick[p["pair_id"]]]["err"]
                for p in pairs}

    # How much room there is at all: pairs the incumbent gets wrong where some
    # OTHER hypothesis was within tolerance. Nothing outside this set can be
    # recovered by any selector, at any epsilon.
    def _recoverable(subset):
        return sum(1 for p in subset
                   if base_err[p["pair_id"]] > err_tol
                   and any(c["err"] <= err_tol for c in p["cands"]))

    print(f"\nincumbent (verification=zncc), {len(pairs)} present pairs")
    print(f"{'set':<6}{'pairs':>7}{'>5px':>7}{'recoverable':>13}"
          f"{'mean err px':>13}")
    print("-" * 46)
    for s in sets + ["ALL"]:
        sub = pairs if s == "ALL" else [p for p in pairs if p["set"] == s]
        if not sub:
            continue
        errs = np.array([base_err[p["pair_id"]] for p in sub])
        print(f"{s:<6}{len(sub):>7}{int((errs > err_tol).sum()):>7}"
              f"{_recoverable(sub):>13}{errs.mean():>13.3f}")

    for eps in margins:
        new_pick = {p["pair_id"]: decide(p["cands"], eps) for p in pairs}
        print(f"\nepsilon = {eps:g}")
        print(f"{'set':<6}{'pairs':>7}{'changed':>9}{'recovered':>11}"
              f"{'broken':>8}{'net':>7}{'mean err px':>13}{'delta':>9}")
        print("-" * 70)
        for s in sets + ["ALL"]:
            sub = pairs if s == "ALL" else [p for p in pairs if p["set"] == s]
            if not sub:
                continue
            changed = rec = brk = 0
            new_errs, old_errs = [], []
            for p in sub:
                pid = p["pair_id"]
                oe = base_err[pid]
                ne = p["cands"][new_pick[pid]]["err"]
                old_errs.append(oe)
                new_errs.append(ne)
                if new_pick[pid] != base_pick[pid]:
                    changed += 1
                    if oe > err_tol >= ne:
                        rec += 1
                    elif ne > err_tol >= oe:
                        brk += 1
            nm, om = float(np.mean(new_errs)), float(np.mean(old_errs))
            print(f"{s:<6}{len(sub):>7}{changed:>9}{rec:>11}{brk:>8}"
                  f"{rec - brk:>+7}{nm:>13.3f}{nm - om:>+9.3f}")

    print(f"\n'recovered' counts only pairs where a hypothesis within "
          f"{err_tol:g} px existed and the override reached it -- the rest "
          f"need\na better pose search or a better network, not a better "
          f"selector. 'broken' is the same trade in reverse.")
    print(f"The incumbent IS the reference frame, so its net is 0 and its "
          f"mean-error delta is 0.000 by construction:")
    print(f"an epsilon is worth carrying forward only if net > 0 AND the "
          f"mean-error delta is negative, on Set B without\nchurning Set A. "
          f"That is a cheap ranking, not a promotion -- confirm the survivor "
          f"with a full-set eval_ext.py\nA/B (gate: paired delta >= +0.35 "
          f"rubric points).")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="*",
                    help="shard directories, each containing a manifest.csv")
    ap.add_argument("--weights",
                    default=os.path.join(HERE, "weights", "driftsense.pt"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="write the per-CANDIDATE records here, so the sweep "
                         "can be re-run with --replay at zero decode cost")
    ap.add_argument("--replay", default=None,
                    help="skip every decode and sweep this CSV (written by an "
                         "earlier --out). Mirrors eval_ext.py's --rescore")
    ap.add_argument("--stride", type=int, default=5,
                    help="take every Nth manifest row. 5 (default) is the "
                         "repo's pre-read convention; it reads ~1.5 rubric "
                         "points optimistic in absolute terms but ranks "
                         "variants correctly. MUST be odd -- see the module "
                         "docstring")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap pairs per shard AFTER striding (0 = no cap)")
    ap.add_argument("--margins", type=float, nargs="+",
                    default=list(DEFAULT_MARGINS),
                    help=f"epsilon values to sweep. Default: "
                         f"{' '.join(str(m) for m in DEFAULT_MARGINS)}. The "
                         f"shipped constant is "
                         f"{DOG_OVERRIDE_MARGIN} (driftsense.config."
                         f"DOG_OVERRIDE_MARGIN)")
    ap.add_argument("--hypotheses", type=int, default=3)
    ap.add_argument("--coarse-scales", type=int, default=17)
    ap.add_argument("--band", action="store_true",
                    help="opt in to the DoG band pre-filter on the COARSE "
                         "sweep (a different thing from the dog verification "
                         "score); off by default, which is SHIPPED_BAND")
    ap.add_argument("--subpixel-rows", action="store_true",
                    help="run the shipped sub-pixel row refinement. OFF by "
                         "default here on purpose: it mutates the winning "
                         "candidate in place AFTER selection, so it would "
                         "measure the incumbent's x post-refinement against "
                         "every challenger's pre-refinement x")
    ap.add_argument("--err-tol", type=float, default=ERR_TOL_PX,
                    help="the px boundary 'recovered'/'broken' are drawn at")
    a = ap.parse_args()

    if a.replay:
        df = pd.read_csv(a.replay)
        print(f"replaying {a.replay}: {len(df)} candidate rows, "
              f"{df.pair_id.nunique()} pairs (no decode)")
        report(df, a.margins, a.err_tol)
        return

    if not a.shards:
        ap.error("give at least one shard directory, or --replay a CSV")
    if a.stride % 2 == 0:
        ap.error(
            f"--stride {a.stride} is even. The manifests cycle severity "
            f"[1,2,3,4,...], so an even stride samples a severity-biased "
            f"SUBPOPULATION, not a subsample (stride 4 reads Set B at 95.4% "
            f"within 5 px where the full set reads 81.4%). Use 1, 3 or 5.")
    if a.subpixel_rows:
        print("!! --subpixel-rows: the winner's x is refined after selection "
              "and the losers' are not.\n!! The incumbent is favoured; this "
              "arm is for cross-checking absolute error, not for ranking.",
              flush=True)

    import multiprocessing as mp
    band = True if a.band else SHIPPED_BAND
    tasks = []
    for d in a.shards:
        man = pd.read_csv(os.path.join(d, "manifest.csv"))
        if a.stride > 1:
            man = man.iloc[::a.stride]
        if a.limit:
            man = man.head(a.limit)
        for _, r in man.iterrows():
            tasks.append((d, r.to_dict(), a.weights, a.threads, a.hypotheses,
                          a.coarse_scales, band, a.subpixel_rows))
    print(f"{len(tasks)} pairs over {len(a.shards)} shard(s), {a.jobs} "
          f"workers, stride {a.stride}, band={band}", flush=True)
    print(f"one decode pass; {len(a.margins)} epsilons replayed offline from "
          f"it", flush=True)

    rows, t0 = [], time.perf_counter()
    with mp.Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
            rows.extend(r)
            if i % 25 == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)}  {el/i:.2f}s/pair wall  "
                      f"eta {(len(tasks)-i)*el/i/60:.1f} min", flush=True)

    df = pd.DataFrame(rows, columns=CAND_FIELDS)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}  ({len(df)} candidate rows, "
              f"{df.pair_id.nunique()} pairs)")
    else:
        print("no --out given: these records are NOT saved, so re-sweeping a "
              "different epsilon list will cost another full decode")
    report(df, a.margins, a.err_tol)


if __name__ == "__main__":
    main()
