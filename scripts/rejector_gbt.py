#!/usr/bin/env python3
"""Nonlinear (gradient-boosted trees) present/absent rejector.

REJECTOR_FINDINGS.md bounds the *linear logistic* family at an in-sample
oracle F1 of 0.8850 and explicitly scopes that bound: "It does not bound a
nonlinear rejector (trees, kernels) or genuinely new features."  This tests
that family, on the CURRENT wide checkpoint (the doc's numbers are from the
0.456M model at F1 0.8716).

Pure numpy so nothing new ships: a fitted ensemble exports to plain arrays.
Protocol matches scripts/rejector_cv.py -- fit on the training folds, pick the
threshold on those same folds, score the held-out fold, never the reverse.
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_threshold import points, prep  # noqa

ALL = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce"]
EXT = ALL + ["rank", "band", "margin"]


# ---------------- gradient-boosted trees (Newton / xgboost-style) ----------
def _best_split(X, g, h, lam, min_leaf):
    n, d = X.shape
    G, H = g.sum(), h.sum()
    base = G * G / (H + lam)
    best = None
    for j in range(d):
        v = X[:, j]
        order = np.argsort(v, kind="stable")
        vs, gs, hs = v[order], g[order], h[order]
        cg, ch = np.cumsum(gs), np.cumsum(hs)
        # candidate splits only where the value actually changes
        ok = np.where(vs[1:] != vs[:-1])[0]
        ok = ok[(ok + 1 >= min_leaf) & (n - ok - 1 >= min_leaf)]
        if not len(ok):
            continue
        GL, HL = cg[ok], ch[ok]
        GR, HR = G - GL, H - HL
        gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - base
        k = int(np.argmax(gain))
        if best is None or gain[k] > best[0]:
            best = (float(gain[k]), j, float((vs[ok[k]] + vs[ok[k] + 1]) / 2.0))
    return best


def _fit_tree(X, g, h, depth, lam, min_leaf):
    if depth == 0 or len(g) < 2 * min_leaf:
        return {"leaf": float(-g.sum() / (h.sum() + lam))}
    s = _best_split(X, g, h, lam, min_leaf)
    if s is None or s[0] <= 1e-9:
        return {"leaf": float(-g.sum() / (h.sum() + lam))}
    _, j, thr = s
    m = X[:, j] <= thr
    if m.all() or (~m).all():
        return {"leaf": float(-g.sum() / (h.sum() + lam))}
    return {"j": j, "thr": thr,
            "L": _fit_tree(X[m], g[m], h[m], depth - 1, lam, min_leaf),
            "R": _fit_tree(X[~m], g[~m], h[~m], depth - 1, lam, min_leaf)}


def _pred_tree(t, X):
    if "leaf" in t:
        return np.full(len(X), t["leaf"])
    m = X[:, t["j"]] <= t["thr"]
    out = np.empty(len(X))
    if m.any():  out[m]  = _pred_tree(t["L"], X[m])
    if (~m).any(): out[~m] = _pred_tree(t["R"], X[~m])
    return out


class GBM:
    def __init__(self, rounds=200, lr=0.06, depth=3, lam=1.0, min_leaf=25):
        self.r, self.lr, self.d, self.lam, self.ml = rounds, lr, depth, lam, min_leaf
    def fit(self, X, y):
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.b = float(np.log(p / (1 - p)))
        f = np.full(len(y), self.b); self.trees = []
        for _ in range(self.r):
            pr = 1.0 / (1.0 + np.exp(-f))
            g, h = pr - y, np.maximum(pr * (1 - pr), 1e-6)
            t = _fit_tree(X, g, h, self.d, self.lam, self.ml)
            f += self.lr * _pred_tree(t, X)
            self.trees.append(t)
        return self
    def decision(self, X):
        f = np.full(len(X), self.b)
        for t in self.trees:
            f += self.lr * _pred_tree(t, X)
        return f


# ---------------- linear logistic, for the like-for-like control -----------
class Logit:
    def __init__(self, iters=400, lr=0.4, l2=1e-3):
        self.i, self.lr, self.l2 = iters, lr, l2
    def fit(self, X, y):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        self.mu, self.sd = mu, sd
        Z = (X - mu) / sd
        w = np.zeros(Z.shape[1]); b = 0.0
        for _ in range(self.i):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            gw = Z.T @ (p - y) / len(y) + self.l2 * w
            gb = float((p - y).mean())
            w -= self.lr * gw; b -= self.lr * gb
        self.w, self.b = w, b
        return self
    def decision(self, X):
        return ((X - self.mu) / self.sd) @ self.w + self.b


def feats(d, cols):
    X = np.column_stack([pd.to_numeric(d[c], errors="coerce").values for c in cols])
    med = np.nanmedian(X, axis=0)
    ix = np.where(~np.isfinite(X))
    X[ix] = np.take(med, ix[1])
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = prep(pd.read_csv(a.csv))
    d = d[d["set"].isin(["A", "B", "C"])].reset_index(drop=True)
    cols = [c for c in EXT if c in d.columns
            and np.isfinite(pd.to_numeric(d[c], errors="coerce").values).mean() >= 0.5]
    print(f"{len(d)} pairs | features: {', '.join(cols)}")
    y = (d.gt_found.values == 0).astype(float)          # 1 = should REJECT
    X = feats(d, cols)

    rng = np.random.default_rng(a.seed)
    fold = rng.permutation(len(d)) % a.folds

    def sweep(stat, sub):
        best = (-1, 0.0)
        for t in np.unique(stat):
            tot = float(points(sub, stat, t))
            if tot > best[0]: best = (tot, float(t))
        return best[1]

    rows = {}
    for name in ("shipped min(score,zncc)", "linear logistic", "GBT (nonlinear)"):
        rows[name] = []
    for k in range(a.folds):
        tr, te = fold != k, fold == k
        dtr, dte = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        # 1. shipped statistic
        s_tr = np.minimum(dtr.score.values, dtr.zncc.values)
        s_te = np.minimum(dte.score.values, dte.zncc.values)
        t = sweep(s_tr, dtr); rows["shipped min(score,zncc)"].append(points(dte, s_te, t, True))
        # 2. linear  (decision is "reject"; negate so higher = more confident FOUND)
        m = Logit().fit(X[tr], y[tr])
        t = sweep(-m.decision(X[tr]), dtr)
        rows["linear logistic"].append(points(dte, -m.decision(X[te]), t, True))
        # 3. nonlinear
        m = GBM().fit(X[tr], y[tr])
        t = sweep(-m.decision(X[tr]), dtr)
        rows["GBT (nonlinear)"].append(points(dte, -m.decision(X[te]), t, True))

    print(f"\n{'statistic':>26} | {'held-out total':>14} | {'F1':>7} | {'AUC':>7}")
    print("-" * 66)
    for k, v in rows.items():
        tot = np.mean([x["total"] for x in v])
        f1  = np.mean([x["f1"]    for x in v])
        auc = np.mean([x["auc"]   for x in v])
        print(f"{k:>26} | {tot:14.2f} | {f1:7.4f} | {auc:7.4f}")

    # NOTE: an in-sample GBT oracle is meaningless (a deep ensemble memorises the
    # rows); it is NOT a ceiling in the sense the linear-family bound was. Omitted.


if __name__ == "__main__":
    main()
