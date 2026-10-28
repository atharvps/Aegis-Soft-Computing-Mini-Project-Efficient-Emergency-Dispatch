"""Pillar 3 - Vectorised Mamdani FIS: clinical risk x traffic x shock -> Golden-Hour urgency U in [0,100].

Membership functions are implemented in pure NumPy (fast, no version conflicts);
scikit-fuzzy, when installed, is used only by verify_against_skfuzzy() as an independent cross-check.
"""
from __future__ import annotations

import numpy as np

try:  # optional cross-check library
    import skfuzzy as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None

EPS = 1e-12
_trapz = getattr(np, "trapezoid", None) or np.trapz


def trimf(x, a, b, c):
    x = np.asarray(x, float)
    return np.clip(np.minimum((x - a) / max(b - a, EPS), (c - x) / max(c - b, EPS)), 0, 1)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    up = np.where(x >= b, 1.0, (x - a) / max(b - a, EPS))
    down = np.where(x <= c, 1.0, (d - x) / max(d - c, EPS))
    return np.clip(np.minimum(up, down), 0, 1)


def _mf(kind, params, x):
    return trapmf(x, *params) if kind == "trap" else trimf(x, *params)


class MamdaniUrgencyEngine:
    RISK = {"Low": ("trap", [0, 0, .25, .45]), "Moderate": ("tri", [.30, .55, .75]),
            "Severe": ("trap", [.60, .80, 1, 1])}
    TRAFFIC = {"FreeFlow": ("trap", [1.0, 1.0, 1.3, 1.7]), "Congested": ("tri", [1.4, 1.9, 2.4]),
               "Gridlock": ("trap", [2.1, 2.5, 3.0, 3.0])}
    SHOCK = {"Stable": ("trap", [.3, .3, .7, .9]), "Worsening": ("tri", [.75, 1.05, 1.35]),
             "RapidCollapse": ("trap", [1.2, 1.5, 2.0, 2.0])}
    OUT = {"VeryLow": ("trap", [0, 0, 10, 25]), "Low": ("tri", [10, 25, 45]), "Medium": ("tri", [35, 50, 65]),
           "High": ("tri", [55, 75, 90]), "Critical": ("trap", [80, 92, 100, 100])}

    def __init__(self):
        self.u = np.linspace(0, 100, 201)
        self.out_names = list(self.OUT)
        self.out_mu = np.array([_mf(k, p, self.u) for k, p in self.OUT.values()])  # (5, U)
        self.rules = []
        for r in range(3):
            for t in range(3):
                for s in range(3):
                    self.rules.append({
                        "id": len(self.rules) + 1, "risk": list(self.RISK)[r], "traffic": list(self.TRAFFIC)[t],
                        "shock": list(self.SHOCK)[s], "consequent": self.out_names[self._level(r, t, s)],
                        "_idx": (r, t, s, self._level(r, t, s))})
        self.ridx, self.tidx, self.sidx, self.cidx = (np.array([x["_idx"][i] for x in self.rules]) for i in range(4))

    @staticmethod
    def _level(r, t, s) -> int:
        """Rule base: risk dominates; traffic and shock escalate. Low risk is capped at Medium, Severe floored at High."""
        base = int(np.floor((3 * r + 1.5 * t + 1.5 * s) / 12 * 4 + 0.5))
        if r == 2:
            base = max(base, 3)
        if r == 0:
            base = min(base, 2)
        return base

    @staticmethod
    def _memberships(table, x):
        return np.stack([_mf(k, p, x) for k, p in table.values()], axis=1)  # (n, 3)

    def fuzzify(self, p, tau, shock):
        p = np.clip(np.atleast_1d(p).astype(float), 0, 1)
        tau = np.clip(np.atleast_1d(tau).astype(float), 1, 3)
        shock = np.clip(np.atleast_1d(shock).astype(float), .3, 2)
        return self._memberships(self.RISK, p), self._memberships(self.TRAFFIC, tau), self._memberships(self.SHOCK, shock)

    def firing(self, p, tau, shock):
        mr, mt, ms = self.fuzzify(p, tau, shock)
        return np.minimum(np.minimum(mr[:, self.ridx], mt[:, self.tidx]), ms[:, self.sidx])  # (n, 27)

    def compute(self, p, tau, shock) -> np.ndarray:
        fire = self.firing(p, tau, shock)
        strength = np.stack([fire[:, self.cidx == k].max(axis=1) if (self.cidx == k).any()
                             else np.zeros(len(fire)) for k in range(5)], axis=1)  # max-aggregation per output set
        agg = np.minimum(strength[:, :, None], self.out_mu[None]).max(axis=1)  # min-implication, max-aggregation
        area = _trapz(agg, self.u, axis=1)
        cen = _trapz(agg * self.u, self.u, axis=1) / np.where(area > EPS, area, 1)
        return np.where(area > EPS, cen, 50.0)

    def explain(self, p, tau, shock, top: int = 5):
        f = self.firing([p], [tau], [shock])[0]
        order = np.argsort(-f)[:top]
        return [{"rule": self.rules[i]["id"], "strength": float(f[i]), "risk": self.rules[i]["risk"],
                 "traffic": self.rules[i]["traffic"], "shock": self.rules[i]["shock"],
                 "consequent": self.rules[i]["consequent"]} for i in order if f[i] > 0]

    def surface(self, shock: float = 0.9, resolution: int = 21):
        xs = np.linspace(0, 1, resolution)
        ys = np.linspace(1, 3, resolution)
        P, T = np.meshgrid(xs, ys)
        z = self.compute(P.ravel(), T.ravel(), np.full(P.size, shock)).reshape(P.shape)
        return xs, ys, z

    def membership_curves(self):
        def block(table, lo, hi):
            x = np.linspace(lo, hi, 101)
            return {"x": x.tolist(), "sets": {n: _mf(k, p, x).tolist() for n, (k, p) in table.items()}}
        return {"risk": block(self.RISK, 0, 1), "traffic": block(self.TRAFFIC, 1, 3),
                "shock": block(self.SHOCK, .3, 2), "urgency": block(self.OUT, 0, 100)}

    def public_rules(self):
        return [{k: v for k, v in r.items() if k != "_idx"} for r in self.rules]

    def verify_against_skfuzzy(self) -> dict:
        """Max abs. difference vs scikit-fuzzy memberships/centroid (None if library unavailable)."""
        if _fuzz is None:
            return {"available": False}
        try:
            x = np.linspace(0, 1, 101)
            d1 = np.abs(trapmf(x, 0, 0, .25, .45) - _fuzz.trapmf(x, [0, 0, .25, .45])).max()
            d2 = np.abs(trimf(x, .3, .55, .75) - _fuzz.trimf(x, [.3, .55, .75])).max()
            mf = self.out_mu[3]
            d3 = abs(_fuzz.defuzz(self.u, mf, "centroid") - _trapz(mf * self.u, self.u) / _trapz(mf, self.u))
            return {"available": True, "max_mf_diff": float(max(d1, d2)), "centroid_diff": float(d3)}
        except Exception as exc:  # pragma: no cover
            return {"available": False, "error": str(exc)}
