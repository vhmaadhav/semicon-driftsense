#!/usr/bin/env python3
"""How often does the uncontested-hypothesis early exit actually fire?

`driftsense.config.EARLY_EXIT_GATES` was validated for *correctness* in
`.agents/PR51_CAMPAIGN.md` -- bit-identical answers against a full
no-early-exit decode -- but its **fire rate** has never been recorded. Without
it the 1.18x headline cannot be attributed to the gate, and there is no
evidence either way on whether loosening the gates would buy more.

This measures three things on the shipped decode (register.py's exact
`locate_phase2` arguments, so the numbers describe the graded path):

1. **Fire rate** -- what fraction of pairs skip the remaining hypotheses,
   split by which gate fired.
2. **Work saved** -- network forward passes actually skipped, as a fraction of
   the passes a no-early-exit decode would pay.
3. **Near misses** -- for every pair that did *not* fire, which gate term was
   binding. That is the input to any proposal to retune the gates.

With `--ab` each pair is additionally decoded with the gates disabled
(`EARLY_EXIT_GATES = ()`), giving a paired wall-clock delta and an
answer-identity check on this shard. Arm order alternates by pair index so
neither arm systematically gets the warm cache.

    python scripts/early_exit_rate.py data/holdout_p2 --out results/early_exit.csv
    python scripts/early_exit_rate.py data/holdout_p2 --n 150 --ab
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# Terms of a gate, in the order they appear in EARLY_EXIT_GATES tuples. Used
# only to name the binding constraint in the near-miss table.
TERMS = ("score", "zncc", "ratio", "gap")


def gate_terms(gate, stats):
    """Per-term pass/fail for one gate against one first-hypothesis result."""
    min_score, min_zncc, max_ratio, min_gap = gate
    return {
        "score": stats["score"] >= min_score,
        "zncc": stats["zncc"] >= min_zncc,
        "ratio": stats["ratio"] <= max_ratio,
        "gap": True if min_gap is None else stats["gap"] >= min_gap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--n", type=int, default=0, help="0 = every pair in the shard")
    ap.add_argument("--threads", type=int, default=4, help="reference machine has 4 cores")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "driftsense.pt"))
    ap.add_argument("--ab", action="store_true",
                    help="also decode each pair with the gates disabled")
    ap.add_argument("--out", default="", help="write the per-pair CSV here")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    import infer as I
    import driftsense.matching as M
    from driftsense.config import (EARLY_EXIT_GATES, SHIPPED_BAND,
                                   SHIPPED_SUBPIXEL_ROWS)

    # --- instrumentation -------------------------------------------------
    # Every hook wraps a module global that locate_phase2 looks up at call
    # time, so nothing inside the decode itself changes.
    live = {}

    real_fires = M._early_exit_fires
    real_cands = M.pose_candidates
    real_locate = M.locate

    def hooked_fires(result, coarse_gap):
        fired = real_fires(result, coarse_gap)
        stats = {
            "score": float(result.get("score", 0.0)),
            "zncc": float(result.get("zncc", -np.inf)),
            "ratio": float(result.get("peak_ratio", 1.0)),
            "gap": float(coarse_gap),
        }
        live.update({"reached": True, "fired": bool(fired), **stats})
        if fired:
            live["gate"] = next(i for i, g in enumerate(M.EARLY_EXIT_GATES)
                                if all(gate_terms(g, stats).values()))
        return fired

    def hooked_cands(*args, **kw):
        out = real_cands(*args, **kw)
        live["n_cands"] = len(out)
        return out

    def hooked_locate(*args, **kw):
        live["n_locate"] = live.get("n_locate", 0) + 1
        return real_locate(*args, **kw)

    M._early_exit_fires = hooked_fires
    M.pose_candidates = hooked_cands
    M.locate = hooked_locate

    model, device = I.load_model(a.weights)
    man = pd.read_csv(os.path.join(a.shard, "manifest.csv"))
    if a.n:
        man = man.head(a.n)

    def decode(ref, sea):
        """One shipped decode; returns (result, seconds, instrumentation)."""
        live.clear()
        live["n_locate"] = 0
        t0 = time.perf_counter()
        res = M.locate_phase2(model, ref, sea, device, refine=True,
                              band=SHIPPED_BAND,
                              subpixel_rows=SHIPPED_SUBPIXEL_ROWS)
        return res, time.perf_counter() - t0, dict(live)

    rows = []
    for i, (_, r) in enumerate(man.iterrows()):
        ref = I.read_gray(os.path.join(a.shard, r.reference_path))
        sea = I.read_gray(os.path.join(a.shard, r.search_path))

        if a.ab and i % 2:
            # Odd pairs run the gates-off arm first, so the warm-cache
            # advantage of going second is split evenly between the arms.
            M.EARLY_EXIT_GATES = ()
            off_res, off_secs, _ = decode(ref, sea)
            M.EARLY_EXIT_GATES = EARLY_EXIT_GATES
            on_res, on_secs, inst = decode(ref, sea)
        else:
            on_res, on_secs, inst = decode(ref, sea)
            if a.ab:
                M.EARLY_EXIT_GATES = ()
                off_res, off_secs, _ = decode(ref, sea)
                M.EARLY_EXIT_GATES = EARLY_EXIT_GATES

        # gen_data.py splits key on `id`/`found`; the generator audit package
        # and the graded manifests use `pair_id`/`present`. Accept either.
        row = {
            "pair_id": getattr(r, "pair_id", getattr(r, "id", i)),
            "set_name": getattr(r, "set_name", ""),
            "present": int(getattr(r, "present", getattr(r, "found", 1))),
            "n_cands": inst.get("n_cands", 0),
            "n_locate": inst.get("n_locate", 0),
            "reached": bool(inst.get("reached", False)),
            "fired": bool(inst.get("fired", False)),
            "gate": inst.get("gate", -1),
            "score": inst.get("score", float("nan")),
            "zncc": inst.get("zncc", float("nan")),
            "ratio": inst.get("ratio", float("nan")),
            "gap": inst.get("gap", float("nan")),
            "secs_on": on_secs,
        }
        if a.ab:
            row["secs_off"] = off_secs
            row["identical"] = all(
                float(on_res.get(k, np.nan)) == float(off_res.get(k, np.nan))
                or (np.isnan(float(on_res.get(k, np.nan)))
                    and np.isnan(float(off_res.get(k, np.nan))))
                for k in ("x", "y", "theta", "scale", "confidence"))
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(man)} pairs", flush=True)

    d = pd.DataFrame(rows)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        d.to_csv(a.out, index=False)

    # --- report ----------------------------------------------------------
    n = len(d)
    reached = d[d.reached]
    fired = d[d.fired]
    print(f"\n{a.shard}   n={n}  threads={a.threads}  hypotheses=3 (shipped default)")
    print("\nfire rate")
    print(f"  gate reachable (>=2 hypotheses)   {len(reached):5d}  {100 * len(reached) / n:5.1f}%")
    share = f", {100 * len(fired) / len(reached):5.1f}% of reachable" if len(reached) else ""
    print(f"  fired                             {len(fired):5d}  "
          f"{100 * len(fired) / n:5.1f}% of all pairs{share}")
    for g, gate in enumerate(EARLY_EXIT_GATES):
        k = int((d.gate == g).sum())
        print(f"    gate {g} {str(gate):<28} {k:5d}  {100 * k / n:5.1f}%")

    paid, full = int(d.n_locate.sum()), int(d.n_cands.sum())
    print("\nwork saved (network forward passes)")
    print(f"  paid {paid}  of  {full} a no-early-exit decode would pay"
          f"   -> {100 * (full - paid) / full:.1f}% skipped, {full / paid:.3f}x fewer")

    miss = d[~d.fired & d.reached]
    if len(miss):
        print(f"\nnear misses ({len(miss)} pairs reached the gate and did not fire)")
        for g, gate in enumerate(EARLY_EXIT_GATES):
            terms = [gate_terms(gate, s) for _, s in miss[list(TERMS)].iterrows()]
            fails = pd.DataFrame(terms).apply(lambda c: int((~c).sum()))
            only = sum(1 for t in terms if sum(not v for v in t.values()) == 1)
            print(f"  gate {g}: blocked by  "
                  + "  ".join(f"{t} {int(fails[t]):4d}" for t in TERMS)
                  + f"   |  {only} pairs blocked by exactly one term")
        print("  (a pair blocked by exactly one term is what a retune could win;"
              " correctness would still have to be re-validated)")

    print(f"\nwall clock, gates on: median {d.secs_on.median():.3f}s  "
          f"mean {d.secs_on.mean():.3f}s  p90 {d.secs_on.quantile(0.9):.3f}s")
    if a.ab:
        print(f"           gates off: median {d.secs_off.median():.3f}s  "
              f"mean {d.secs_off.mean():.3f}s  p90 {d.secs_off.quantile(0.9):.3f}s")
        delta = d.secs_off - d.secs_on
        print(f"  paired saving per pair: median {delta.median():+.3f}s  "
              f"mean {delta.mean():+.3f}s   -> {d.secs_off.sum() / d.secs_on.sum():.3f}x overall")
        if d.fired.any():
            print(f"  on the {int(d.fired.sum())} pairs that fired: "
                  f"median {delta[d.fired].median():+.3f}s")
        else:
            print("  no pair fired")
        bad = int((~d.identical).sum())
        print(f"  answers identical to the gates-off arm: {n - bad}/{n}"
              + ("" if not bad else f"   ** {bad} DIFFER **"))
    if a.out:
        print(f"\nper-pair rows -> {a.out}")


if __name__ == "__main__":
    main()
