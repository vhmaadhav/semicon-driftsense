#!/usr/bin/env python3
"""Compare checkpoints already evaluated by evaluate.py, with a paired bootstrap.

Every difference between checkpoints on this problem is two or three samples in
three hundred, which is inside the +/-1.3 point binomial noise of a 300-scene
split. Reporting the raw accuracies side by side invites reading a 2-sample lead
as a result. This does the comparison the way it has to be done to mean
anything: the checkpoints are evaluated on the *same* scenes, so the difference
can be bootstrapped **paired** -- resample scenes, recompute both accuracies on
the same resample, take the difference. The pairing cancels scene difficulty,
which is the dominant variance term, and gives a far tighter interval than
comparing two independent confidence intervals would.

It reads the results.json files evaluate.py already wrote, so it costs nothing
and never re-runs a model.

    python scripts/compare_checkpoints.py results/cmp_ship results/cmp_v3last ...
    python scripts/compare_checkpoints.py --baseline results/cmp_ship results/cmp_*
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

TOL = 5.0


def load(d: str) -> tuple[str, np.ndarray]:
    """Accept either an evaluate.py output directory or a stream_eval.py json."""
    if os.path.isdir(d):
        with open(os.path.join(d, "results.json")) as f:
            blob = json.load(f)
        # evaluate.py used to write a bare list of split blocks and now writes
        # {"provenance": ..., "splits": [...]}. The committed cmp_* artifacts
        # predate the change, so accept both.
        blocks = blob["splits"] if isinstance(blob, dict) else blob
        errs = np.concatenate([np.asarray(b["errors_siamese"], dtype=np.float64)
                               for b in blocks])
        return os.path.basename(d.rstrip("/")), errs
    with open(d) as f:
        blob = json.load(f)
    name = os.path.splitext(os.path.basename(d))[0]
    return name, np.asarray(blob["errors"], dtype=np.float64)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int, rng, tol=TOL):
    """CI for acc@tol(a) - acc@tol(b) over the scenes both were run on."""
    if a.size != b.size:
        raise ValueError(f"unpaired: {a.size} vs {b.size} scenes")
    ha, hb = (a <= tol).astype(np.float64), (b <= tol).astype(np.float64)
    idx = rng.integers(0, a.size, size=(n, a.size))
    diffs = ha[idx].mean(1) - hb[idx].mean(1)
    return (float(ha.mean() - hb.mean()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)),
            float((diffs <= 0).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="directories containing results.json")
    p.add_argument("--baseline", default="", help="compare every dir against this one")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    runs = [load(d) for d in args.dirs]

    print(f"{'checkpoint':<28}{'n':>5}{'median':>9}{'acc@1px':>10}"
          f"{'acc@2px':>10}{'acc@5px':>10}{'hits@5':>9}")
    print("-" * 81)
    for name, e in sorted(runs, key=lambda r: -(r[1] <= TOL).mean()):
        print(f"{name:<28}{e.size:>5}{np.median(e):>8.2f}p"
              f"{(e <= 1).mean():>10.3f}{(e <= 2).mean():>10.3f}"
              f"{(e <= TOL).mean():>10.3f}{int((e <= TOL).sum()):>9}")

    base = args.baseline or args.dirs[0]
    bname, berr = load(base)
    others = [(n, e) for n, e in runs if n != bname]
    if not others:
        return

    print(f"\npaired bootstrap vs {bname}  ({args.boot} resamples, "
          f"same scenes both sides)")
    print(f"{'checkpoint':<28}{'d acc@5px':>12}{'95% CI':>22}{'P(worse)':>11}")
    print("-" * 74)
    for name, e in sorted(others, key=lambda r: -(r[1] <= TOL).mean()):
        try:
            d, lo, hi, pw = paired_bootstrap(e, berr, args.boot, rng)
        except ValueError as err:
            print(f"{name:<28}  skipped ({err})")
            continue
        print(f"{name:<28}{d:>+11.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>23}{pw:>11.3f}")
    print("\nP(worse) is the bootstrap probability the checkpoint is no better than "
          f"{bname}.\nA CI spanning 0 means the split cannot tell them apart.")


if __name__ == "__main__":
    main()
