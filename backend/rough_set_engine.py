"""Pillar 1 - Pawlak Rough Sets + QuickReduct feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd


class RoughSetSelector:
    """Equivalence-class / positive-region machinery over a discretised decision table."""

    def __init__(self, tolerance: float = 0.98, max_features: int | None = 8, min_support: int = 5,
                 min_features: int = 6, relevance_min_mi: float = 0.01):
        self.tolerance = tolerance
        self.relevance_min_mi = relevance_min_mi  # nats; screens out attributes that only fragment blocks
        self.min_features = min_features  # clinical floor so the network input is not starved
        self.min_support = min_support  # blocks smaller than this are not "certain" (stops singleton inflation of gamma)
        self.max_features = max_features
        self.reduct_: list[str] = []
        self.report_: dict = {}

    # ---- core rough-set operations -------------------------------------------------
    @staticmethod
    def partition(X: np.ndarray, cols: list[int]) -> np.ndarray:
        """IND(P): block id of every object (rows identical on P share a block)."""
        if not cols:
            return np.zeros(len(X), dtype=int)
        _, inv = np.unique(X[:, cols], axis=0, return_inverse=True)
        return inv.ravel()

    def positive_mask(self, X, y, cols) -> np.ndarray:
        """Objects whose IND(P)-block is wholly inside one decision class (lower approximations)."""
        inv = self.partition(X, cols)
        nb = inv.max() + 1
        ymin = np.full(nb, np.iinfo(np.int64).max)
        ymax = np.full(nb, np.iinfo(np.int64).min)
        np.minimum.at(ymin, inv, y)
        np.maximum.at(ymax, inv, y)
        size = np.bincount(inv, minlength=nb)
        return ((ymin == ymax) & (size >= self.min_support))[inv]

    def gamma(self, X, y, cols) -> float:
        """Dependency degree gamma_P(D) = |POS_P(D)| / |U|."""
        return float(self.positive_mask(X, y, cols).mean())

    def approximations(self, X, y, cols, labels=None) -> list[dict]:
        inv = self.partition(X, cols)
        size = np.bincount(inv)
        out = []
        for c in np.unique(y):
            cnt = np.bincount(inv, weights=(y == c).astype(float), minlength=len(size))
            lower = int(size[cnt == size].sum())
            upper = int(size[cnt > 0].sum())
            out.append({"class": (labels or {}).get(int(c), f"Class {int(c)}"), "size": int((y == c).sum()), "lower": lower,
                        "upper": upper, "accuracy": float(lower / upper) if upper else 0.0})
        return out

    @staticmethod
    def mutual_info(x, y) -> float:
        _, yi = np.unique(y, return_inverse=True)
        joint = np.zeros((int(x.max()) + 1, int(yi.max()) + 1))
        np.add.at(joint, (x, yi), 1)
        p = joint / joint.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        nz = p > 0
        return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())

    # ---- QuickReduct -----------------------------------------------------------------
    def fit(self, disc: pd.DataFrame, y, class_labels: dict | None = None) -> dict:
        """Greedy QuickReduct on the support-filtered dependency degree.

        With many attributes the table fragments into tiny blocks, so gamma(C) itself collapses
        (curse of dimensionality). The stopping reference is therefore gamma_peak, the best
        dependency reachable on the greedy path; the reduct is the shortest prefix with
        gamma >= tolerance * gamma_peak (floored at min_features, capped at max_features), then pruned of redundancy.
        """
        names = list(disc.columns)
        X = disc.to_numpy(dtype=int)
        y = np.asarray(y).astype(int)
        d = len(names)
        gamma_full = self.gamma(X, y, list(range(d)))
        mi = [self.mutual_info(X[:, c], y) for c in range(d)]
        cand = [c for c in range(d) if mi[c] >= self.relevance_min_mi]
        if len(cand) < self.min_features:
            cand = sorted(np.argsort(mi)[::-1][:self.min_features].tolist())
        depth = min(len(cand), max(12, (self.max_features or 0) + 2))

        path, gains, g = [], [], 0.0
        while len(path) < depth:
            best_g, best_c = -1.0, -1
            for c in cand:
                if c in path:
                    continue
                gc = self.gamma(X, y, path + [c])
                if gc > best_g + 1e-12:
                    best_g, best_c = gc, c
            gains.append({"step": len(path) + 1, "attribute": names[best_c], "gamma_before": g,
                          "gamma_after": best_g, "delta": best_g - g})
            path.append(best_c)
            g = best_g
        peak = max(t["gamma_after"] for t in gains)
        target = self.tolerance * peak
        k = next(i for i, t in enumerate(gains, 1) if t["gamma_after"] >= target - 1e-12)
        cap = self.max_features or d
        R = path[:min(max(k, self.min_features), cap)]
        trace = gains[:len(R)]

        for c in list(R):  # backward pruning -> minimal reduct for the same threshold
            rest = [r for r in R if r != c]
            if len(R) > self.min_features and self.gamma(X, y, rest) >= self.gamma(X, y, R) - 1e-12:
                R = rest
        g = self.gamma(X, y, R)

        self.reduct_ = [names[i] for i in R]
        sig = []
        for i, n in enumerate(names):
            loss = g - self.gamma(X, y, [r for r in R if r != i]) if i in R and len(R) > 1 else 0.0
            sig.append({"attribute": n, "single_gamma": self.gamma(X, y, [i]), "mutual_info": mi[i],
                        "drop_loss": loss, "selected": i in R})
        self.report_ = {
            "attributes": names, "reduct": self.reduct_,
            "dropped": [n for n in names if n not in self.reduct_],
            "gamma_full": gamma_full, "gamma_peak": peak, "gamma_reduct": g,
            "dependency_ratio": g / peak if peak else 0.0,
            "reached_threshold": bool(g >= target - 1e-12), "tolerance": self.tolerance,
            "min_support": self.min_support, "relevance_min_mi": self.relevance_min_mi, "min_features": self.min_features,
            "trace": trace, "significance": sig,
            "approximations": self.approximations(X, y, R, class_labels),
            "equivalence_classes": {"full": int(self.partition(X, list(range(d))).max() + 1),
                                    "reduct": int(self.partition(X, R).max() + 1)},
            "n_full": d, "n_reduct": len(R), "reduction_pct": 100.0 * (1 - len(R) / d),
        }
        return self.report_
