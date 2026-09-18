#!/usr/bin/env python3
"""Phase 3 evaluation harness: generated CAD split -> decode -> rubric.

    # 1. a blind Phase 3 pairs.csv + ground_truth.csv from a generator split
    python scripts/phase3_eval.py prepare --split generator_i4c/output/cad_varied_dev

    # 2. decode every pair through phase3.predict_pair (the shipped path), in
    #    parallel; --set overrides one predict_pair argument (measurement only)
    python scripts/phase3_eval.py run --split ... --out runs/dev.csv
    python scripts/phase3_eval.py run --split ... --out runs/dev_image.csv --set use_cad=False

    # 3. the Phase 3 rubric: found from the run's own decision, or --threshold
    python scripts/phase3_eval.py score --split ... runs/dev.csv

`run` writes the RAW answer (pose and score for every pair, before the found
mask), so `score` can sweep thresholds and confidence definitions without
re-decoding. The rubric follows the Phase 3 brief: localisation 40 (tiered at
1/2/3/5 px, zero for a declined present pair), scale 10 and rotation 10 (on
pairs that earned localisation credit), rejection F1 15, calibration AUC 10.
Which class the F1 treats as positive is not settled by the brief; both are
printed and reject-positive (the harsher one) goes into the total, as in
scripts/grade_emulation.py.
"""

from __future__ import annotations

import argparse
import ast
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))
SCALE_TIERS = ((0.01, 1.00), (0.02, 0.60), (0.05, 0.30))
ROT_TIERS = ((0.25, 1.00), (0.50, 0.60), (1.00, 0.30))
PAIRS_FIELDS = ["pair_id", "search_path", "reference_gds_path", "search_gds_path",
                "reference_sem_path", "params_json_path"]


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(split: str) -> None:
    """pairs.csv (blind layout: the two withheld columns empty, paths relative
    to the split root) and ground_truth.csv, from the generator manifest."""
    rows = list(csv.DictReader(open(os.path.join(split, "manifest.csv"), newline="")))
    with open(os.path.join(split, "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PAIRS_FIELDS)
        for r in rows:
            w.writerow([r["id"], r["search_path"], r["reference_gds_path"],
                        r.get("search_gds_path", ""), "", ""])
    meta = [c for c in rows[0] if c not in ("id", "shard_id", "match_found", "gt_x", "gt_y", "seed",
                                            "gt_theta", "gt_scale") and not c.endswith("_path")
            and not c.startswith("gt_box")]
    with open(os.path.join(split, "ground_truth.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "present", "x", "y", "theta", "scale", *meta])
        for r in rows:
            p = r["match_found"] == "True"
            w.writerow([r["id"], int(p),
                        r["gt_x"] if p else 0, r["gt_y"] if p else 0,
                        (r.get("gt_theta") or 0.0) if p else 0,
                        (r.get("gt_scale") or 10.0) if p else 0,
                        *[r[c] for c in meta]])
    print(f"{len(rows)} pairs -> {split}/pairs.csv, ground_truth.csv")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

_W = {}


def _init(weights: str, overrides: dict) -> None:
    import register as R
    R.cap_threads(1)
    import infer as I
    import phase3
    model, device = I.load_model(weights) or (None, None)
    if model is None:
        raise SystemExit(f"weights failed to load: {weights}")
    _W.update(model=model, device=device, phase3=phase3, overrides=overrides)


def _one(item):
    pid, sea_path, gds_path, sgds_path = item
    t0 = time.perf_counter()
    try:
        res = _W["phase3"].predict_pair(_W["model"], _W["device"], gds_path, sea_path, sgds_path,
                                        **_W["overrides"])
        out = {"pair_id": pid, "error": ""}
        for k, v in res.items():
            if isinstance(v, (int, float, np.floating, np.integer, bool)):
                out[k] = float(v)
            elif isinstance(v, str):
                out[k] = v
    except Exception as e:  # noqa: BLE001 -- one pair's failure is a row, not a crash
        out = {"pair_id": pid, "error": f"{type(e).__name__}: {e}"}
    out["secs"] = time.perf_counter() - t0
    return out


def run(split: str, out: str, weights: str, workers: int, limit: int, seed: int,
        overrides: dict) -> None:
    rows = list(csv.DictReader(open(os.path.join(split, "pairs.csv"), newline="")))
    if limit:
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=min(limit, len(rows)), replace=False))]
    items = [(r["pair_id"], os.path.join(split, r["search_path"]),
              os.path.join(split, r["reference_gds_path"]),
              os.path.join(split, r["search_gds_path"]) if r.get("search_gds_path") else "")
             for r in rows]
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(weights, overrides)) as pool:
        for n, r in enumerate(pool.imap(_one, items, chunksize=1), 1):
            results.append(r)
            if n % 20 == 0 or n == len(items):
                el = time.time() - t0
                print(f"\r  {n}/{len(items)}  {n / el:.2f} pairs/s  ETA {(len(items) - n) / (n / el) / 60:.1f} min ",
                      end="", flush=True)
    print()
    keys = []
    for r in results:
        keys += [k for k in r if k not in keys]
    with open(out, "w", newline="") as f:
        f.write(f"# overrides={overrides!r}\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    errs = sum(1 for r in results if r.get("error"))
    print(f"wrote {len(results)} rows ({errs} errors) -> {out}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def tier(v: float, tiers) -> float:
    for bound, credit in tiers:
        if v <= bound:
            return credit
    return 0.0


def load(split: str, run_csv: str, score_col: str = "confidence"):
    import pandas as pd
    gt = pd.read_csv(os.path.join(split, "ground_truth.csv"))
    pr = pd.read_csv(run_csv, comment="#")
    df = gt.rename(columns={"present": "gt_found", "x": "gt_x", "y": "gt_y",
                            "theta": "gt_theta", "scale": "gt_scale"}).merge(pr, on="pair_id")
    df["score"] = df[score_col].fillna(0.0) if score_col in df else 0.0
    # The run's own found decision when it made one (the CAD path decides
    # found separately from its confidence); a threshold otherwise.
    df["found_raw"] = df["found"].fillna(0).astype(int) if "found" in df else -1
    for c in ("x", "y", "theta", "scale"):
        df[c] = df[c].fillna(0.0) if c in df else 0.0
    df["err"] = np.where(df.gt_found == 1, np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    return df


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                      # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def f1(found, gt, positive="reject") -> float:
    if positive == "reject":
        tp = np.sum((found == 0) & (gt == 0)); fp = np.sum((found == 0) & (gt == 1))
        fn = np.sum((found == 1) & (gt == 0))
    else:
        tp = np.sum((found == 1) & (gt == 1)); fp = np.sum((found == 1) & (gt == 0))
        fn = np.sum((found == 0) & (gt == 1))
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d else 0.0


def rubric(df, t: float | None) -> dict:
    """t=None: use the run's own found column."""
    gt = df.gt_found.to_numpy()
    score = df.score.to_numpy()
    found = df.found_raw.to_numpy() if t is None else (score >= t).astype(int)
    present = gt == 1
    err = df.err.to_numpy()
    loc = np.array([tier(e, LOC_TIERS) if p and f else 0.0 for e, p, f in zip(err, present, found)])
    ok = present & (loc > 0)
    s_err = np.abs(df.scale - df.gt_scale).to_numpy() / np.where(df.gt_scale > 0, df.gt_scale, 1.0)
    r_err = np.abs(df.theta - df.gt_theta).to_numpy()
    r = {
        "loc": float(loc[present].mean()) if present.any() else float("nan"),
        "scale": float(np.mean([tier(v, SCALE_TIERS) for v in s_err[ok]])) if ok.any() else float("nan"),
        "rot": float(np.mean([tier(v, ROT_TIERS) for v in r_err[ok]])) if ok.any() else float("nan"),
        "f1_reject": f1(found, gt, "reject"),
        "f1_found": f1(found, gt, "present"),
        "auc": auc(score[present & (err <= 5)], score[~(present & (err <= 5))]),
    }
    r["total"] = 40 * r["loc"] + 10 * r["scale"] + 10 * r["rot"] + 15 * r["f1_reject"] + 10 * r["auc"]
    r["n"], r["n_present"] = len(df), int(present.sum())
    r["tp_reject"] = int(np.sum((found == 0) & (gt == 0)))
    r["lost_real"] = int(np.sum((found == 0) & (gt == 1)))
    r["missed_absent"] = int(np.sum((found == 1) & (gt == 0)))
    return r


def report(df, t: float, label: str = "") -> dict:
    r = rubric(df, t)
    pres = df[df.gt_found == 1]
    e = pres.err
    print(f"== {label}  found={'run decision' if t is None else f'score >= {t:g}'}  "
          f"n={r['n']} present={r['n_present']}"
          + (f"  methods={df.method.value_counts().to_dict()}" if "method" in df else ""))
    print(f"  localisation  {r['loc']:.4f} -> {40 * r['loc']:6.2f}/40   "
          f"err median {e.median():.2f}  <=1 {np.mean(e <= 1):.1%}  <=2 {np.mean(e <= 2):.1%}  "
          f"<=5 {np.mean(e <= 5):.1%}  >20 {np.mean(e > 20):.1%}")
    rot_e = (pres.theta - pres.gt_theta).abs()[e <= 5]
    print(f"  scale         {r['scale']:.4f} -> {10 * r['scale']:6.2f}/10")
    print(f"  rotation      {r['rot']:.4f} -> {10 * r['rot']:6.2f}/10   "
          f"|dtheta| median {rot_e.median():.3f}  p90 {rot_e.quantile(.9):.3f}")
    print(f"  rejection F1  {r['f1_reject']:.4f} -> {15 * r['f1_reject']:6.2f}/15   "
          f"[found-positive {r['f1_found']:.4f}]  rejected-absent {r['tp_reject']}  "
          f"lost-real {r['lost_real']}  missed-absent {r['missed_absent']}")
    print(f"  calibration   {r['auc']:.4f} -> {10 * r['auc']:6.2f}/10")
    print(f"  TOTAL         {r['total']:.2f} / 85 measurable")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--split", required=True)
    p = sub.add_parser("run")
    p.add_argument("--split", required=True); p.add_argument("--out", required=True)
    p.add_argument("--weights", default=os.path.join(REPO, "weights", "driftsense.pt"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p = sub.add_parser("score")
    p.add_argument("--split", required=True); p.add_argument("runs", nargs="+")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--score-col", default="score")
    a = ap.parse_args()

    if a.cmd == "prepare":
        prepare(a.split)
    elif a.cmd == "run":
        overrides = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            try:
                overrides[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                overrides[k] = v
        run(a.split, a.out, a.weights, a.workers, a.limit, a.seed, overrides)
    else:
        for rc in a.runs:
            report(load(a.split, rc, a.score_col), a.threshold, os.path.basename(rc))


if __name__ == "__main__":
    main()
