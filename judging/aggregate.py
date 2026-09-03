#!/usr/bin/env python3
"""Aggregate the per-set rubric JSONs from score_rubric.py into one table.

    python judging/aggregate.py judging/out/S*/rubric.json

Reports each set's 85-point subtotal, its components, the bonus gates and the
per-pair latency, plus the across-set mean and spread. The spread is the point:
a single set resolves nothing at this sample size, and a difference smaller
than the between-set spread is not a result.
"""

from __future__ import annotations

import argparse
import json
import statistics as st


def _fmt(vals, prec=2):
    return "  ".join(f"{v:>{prec + 5}.{prec}f}" for v in vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="primary runs -- must be one "
                    "machine configuration, since the latency columns are "
                    "pooled across them")
    ap.add_argument("--also", nargs="*", default=[],
                    help="sensitivity runs (different core type, diagnostic "
                         "variants) reported SEPARATELY rather than pooled")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    def read(paths):
        out = []
        for p in paths:
            with open(p) as fh:
                out.append(json.load(fh))
        out.sort(key=lambda r: r["label"])
        return out

    rs = read(a.jsons)
    extra = read(a.also)
    labels = [r["label"] for r in rs]

    # A pooled latency column is only meaningful across ONE machine
    # configuration, and a pooled accuracy column double-counts a set that
    # appears twice. Both happened when this was called with a bare S* glob
    # and it silently picked up the E-core rerun of S1, so refuse instead of
    # averaging: the title is derived from the data, never asserted.
    base = [l.split("_")[0] for l in labels]
    if len(set(base)) != len(base):
        dupe = sorted({b for b in base if base.count(b) > 1})
        raise SystemExit(
            f"refusing to pool: {dupe} appears more than once in the primary "
            f"runs ({labels}). A rerun of the same set on different cores is a "
            "sensitivity run -- pass it with --also.")
    if any("_" in l for l in labels):
        raise SystemExit(
            f"refusing to pool: {[l for l in labels if '_' in l]} look like "
            "variant runs. Pass variants with --also.")
    title = a.title or (f"{len(rs)} x {rs[0]['pairs']} spec-compliant pairs "
                        f"({', '.join(labels)})")

    def col(fn):
        return [fn(r) for r in rs]

    rows = [
        ("loc /40", col(lambda r: r["components"]["localisation_40"]["points"])),
        ("scale /10", col(lambda r: r["components"]["pose_scale_10"]["points"])),
        ("rot /10", col(lambda r: r["components"]["pose_rotation_10"]["points"])),
        ("reject /15", col(lambda r: r["components"]["rejection_15"]["points"])),
        ("calib /10", col(lambda r: r["components"]["calibration_10"]["points"])),
        ("SUBTOTAL /85", col(lambda r: r["subtotal_85"])),
        ("F1(reject)", col(lambda r: r["components"]["rejection_15"]["f1_reject"])),
        ("AUC", col(lambda r: r["components"]["calibration_10"]["auc"])),
        ("set A credit", col(lambda r: r["per_set"]["A"]["credit"])),
        ("set B credit", col(lambda r: r["per_set"]["B"]["credit"])),
        ("set C corr-rej", col(lambda r: r["per_set"]["C"]["credit"])),
        ("set D credit", col(lambda r: r["bonus"]["set_d_credit"])),
        ("BONUS pts", col(lambda r: float(r["bonus"]["points"]))),
    ]
    timed = all(r.get("timing") for r in rs)
    if timed:
        rows += [
            ("median s/pair", col(lambda r: r["timing"]["median_s"])),
            ("mean s/pair", col(lambda r: r["timing"]["mean_s"])),
            ("p90 s/pair", col(lambda r: r["timing"]["p90_s"])),
            ("max s/pair", col(lambda r: r["timing"]["max_s"])),
        ]

    w = 16
    head = f"{'metric':<{w}}" + "".join(f"{l:>9}" for l in labels) \
        + f"{'mean':>10}{'sd':>8}{'min':>9}{'max':>9}"
    out = ["=" * len(head), title, "=" * len(head), head, "-" * len(head)]
    for name, vals in rows:
        mean = st.fmean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        line = f"{name:<{w}}" + "".join(f"{v:>9.4f}" for v in vals) \
            + f"{mean:>10.4f}{sd:>8.4f}{min(vals):>9.4f}{max(vals):>9.4f}"
        out.append(line)
        if name in ("SUBTOTAL /85", "BONUS pts"):
            out.append("-" * len(head))
    out.append("=" * len(head))

    # Gate verdicts across every set, not on the mean: a gate that fails on one
    # set out of five has failed, and averaging hides exactly that.
    n = len(rs)
    b6 = sum(r["bonus"]["plus6_natural_reading"] for r in rs)
    b6s = sum(r["bonus"]["plus6_strict_reading"] for r in rs)
    b4 = sum(r["bonus"]["plus4_f1_gate"] for r in rs)
    out.append(f"+6 gate (natural reading)   met on {b6}/{n} sets")
    out.append(f"+6 gate (strict reading)    met on {b6s}/{n} sets")
    out.append(f"+4 gate (F1(reject)>=0.90)  met on {b4}/{n} sets")
    if timed:
        med = [r["timing"]["median_s"] for r in rs]
        budget = sum(not r["timing"]["over_median_budget"] for r in rs)
        target = sum(r["timing"]["meets_target_median"] for r in rs)
        to = sum(r["timing"]["over_hard_timeout"] for r in rs)
        allp = sum(r["timing"]["pairs_timed"] for r in rs)
        out.append(f"median <= 5 s (contract)    met on {budget}/{n} sets")
        out.append(f"median <= 2 s (target)      met on {target}/{n} sets  "
                   f"[worst set median {max(med):.3f} s]")
        out.append(f"pairs over the 20 s timeout {to} / {allp}")
        tot85 = [r["subtotal_85"] for r in rs]
        out.append("")
        out.append(f"ACROSS ALL {allp} PAIRS: subtotal {st.fmean(tot85):.2f} / 85 "
                   f"(sd {st.stdev(tot85) if n > 1 else 0:.2f}), "
                   f"bonus {st.fmean([float(r['bonus']['points']) for r in rs]):.1f} / 10, "
                   f"median {st.fmean(med):.3f} s/pair")
    if extra:
        out.append("")
        out.append("SENSITIVITY RUNS (reported separately, NOT pooled above)")
        out.append("-" * len(head))
        for r in extra:
            t = r.get("timing") or {}
            out.append(
                f"  {r['label']:<14} subtotal {r['subtotal_85']:>6.2f}/85   "
                f"bonus {r['bonus']['points']:>2}/10   "
                + (f"median {t['median_s']:.3f} s  p90 {t['p90_s']:.3f} s  "
                   f"max {t['max_s']:.3f} s" if t else "no timing"))
    out.append("=" * len(head))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
