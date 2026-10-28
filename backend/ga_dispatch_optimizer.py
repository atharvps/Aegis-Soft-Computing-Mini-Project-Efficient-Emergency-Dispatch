"""Pillar 4 - Constrained elitist GA: ambulance permutation + hospital genes + bed-overflow repair."""
from __future__ import annotations

import time

import numpy as np

HOSPITALS = [  # beds per specialty: [General, Cardiac, Neuro, Trauma]
    {"name": "AIIMS Trauma Centre", "lat": 28.5672, "lon": 77.2100, "beds": [6, 3, 3, 5]},
    {"name": "Safdarjung Hospital", "lat": 28.5686, "lon": 77.2050, "beds": [6, 2, 2, 4]},
    {"name": "RML Hospital", "lat": 28.6258, "lon": 77.2000, "beds": [5, 3, 2, 2]},
    {"name": "LNJP Hospital", "lat": 28.6396, "lon": 77.2397, "beds": [6, 1, 2, 3]},
    {"name": "Max Saket", "lat": 28.5276, "lon": 77.2118, "beds": [4, 3, 2, 1]},
    {"name": "Medanta Gurugram", "lat": 28.4395, "lon": 77.0425, "beds": [4, 3, 3, 2]},
]
CENTERS = np.array([[28.63, 77.22], [28.57, 77.20], [28.66, 77.12], [28.59, 77.07], [28.54, 77.27], [28.70, 77.20]])


def city_points(rng, n, spread=0.02):
    c = CENTERS[rng.integers(0, len(CENTERS), n)]
    return c + rng.normal(0, spread, (n, 2))


def haversine_matrix(a, b):
    la1, lo1 = np.radians(a[:, 0])[:, None], np.radians(a[:, 1])[:, None]
    la2, lo2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def pmx(p1, p2, rng):
    """Partially Mapped Crossover on permutations: child is always a valid permutation."""
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    b += 1
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    seg = set(p1[a:b].tolist())
    pos2 = np.empty(n, dtype=int)
    pos2[p2] = np.arange(n)
    for i in range(a, b):
        v = p2[i]
        if v in seg:
            continue
        pos = i
        while a <= pos < b:
            pos = pos2[p1[pos]]
        child[pos] = v
    mask = child < 0
    child[mask] = p2[mask]
    return child


class ConstrainedDispatchGA:
    def __init__(self, urgency, spec, tau, p_crit, amb_xy, pat_xy, hosp_xy, beds, population=80,
                 generations=60, crossover=0.85, mutation=0.12, elite=6, tournament=4, seed=42,
                 penalties=(500.0, 350.0, 25.0), speed_kmh=40.0, road_factor=1.3):
        self.U = np.asarray(urgency, float)
        self.spec = np.asarray(spec, int)
        self.tau = np.asarray(tau, float)
        self.p = np.asarray(p_crit, float)
        self.N, self.M, self.K = len(self.U), len(amb_xy), len(hosp_xy)
        if self.M < self.N:
            raise ValueError("need at least as many ambulances as patients")
        self.beds = np.asarray(beds, int)
        self.S = self.beds.shape[1]
        self.P, self.G = int(population), int(generations)
        self.pc, self.pm, self.E, self.tk = crossover, mutation, min(elite, self.P - 1), tournament
        self.l1, self.l2, self.l3 = penalties
        self.rng = np.random.default_rng(seed)
        self.idx = np.arange(self.N)
        self.order = np.argsort(-self.U)
        # pre-computed travel-time matrices (minutes); tau_i scales both legs for patient i
        k = road_factor / speed_kmh * 60.0
        self.T_ap = haversine_matrix(amb_xy, pat_xy) * k * self.tau[None, :]   # (M, N)
        self.T_ph = haversine_matrix(pat_xy, hosp_xy) * k * self.tau[:, None]  # (N, K)

    # ---- vectorised fitness over a whole population ---------------------------------
    def evaluate(self, amb, hosp):
        """amb, hosp: (P, N) int arrays -> cost J (P,)."""
        P = amb.shape[0]
        total = self.T_ap[amb, self.idx] + self.T_ph[self.idx, hosp]
        wt = (self.U / 100.0 * total).sum(1)
        flat = np.arange(P)[:, None] * self.K * self.S + hosp * self.S + self.spec[None, :]
        counts = np.bincount(flat.ravel(), minlength=P * self.K * self.S).reshape(P, self.K, self.S)
        over = np.maximum(0, counts - self.beds[None]).sum((1, 2))
        mism = (self.beds[hosp, self.spec[None, :]] == 0).sum(1)
        breach = np.where(self.U >= 75, np.maximum(0, total - 45), 0).sum(1)
        return wt + self.l1 * over + self.l2 * mism + self.l3 * breach

    def breakdown(self, amb, hosp) -> dict:
        pick, trans = self.T_ap[amb, self.idx], self.T_ph[self.idx, hosp]
        total = pick + trans
        counts = np.zeros((self.K, self.S), int)
        np.add.at(counts, (hosp, self.spec), 1)
        crit = self.U >= 75
        return {
            "cost": float(self.evaluate(amb[None], hosp[None])[0]), "pickup": pick, "transit": trans, "total": total,
            "bed_violations": int(np.maximum(0, counts - self.beds).sum()),
            "specialty_mismatches": int((self.beds[hosp, self.spec] == 0).sum()),
            "critical_count": int(crit.sum()),
            "avg_critical_wait": float(total[crit].mean()) if crit.any() else 0.0,
            "golden_hour_survival_pct": float((total[crit] <= 45).mean() * 100) if crit.any() else 100.0,
            "mortality_exposure": float((self.U / 100 * total).sum()), "assigned": counts,
        }

    # ---- repair: greedy hospital capacity reallocation ------------------------------
    def repair(self, h):
        rem, out = self.beds.copy(), h.copy()
        for i in self.order:  # most urgent patients keep their choice first
            s, k = self.spec[i], out[i]
            if rem[k, s] > 0:
                rem[k, s] -= 1
            else:
                cand = np.where(rem[:, s] > 0)[0]
                if len(cand):
                    k2 = cand[np.argmin(self.T_ph[i, cand])]
                    out[i] = k2
                    rem[k2, s] -= 1
        return out

    # ---- baselines ----------------------------------------------------------------------
    def greedy_fcfs(self):
        free = np.ones(self.M, bool)
        amb = np.empty(self.N, int)
        for i in range(self.N):
            a = int(np.argmin(np.where(free, self.T_ap[:, i], np.inf)))
            amb[i], free[a] = a, False
        return amb, np.argmin(self.T_ph, axis=1)

    def _priority_greedy(self, order):
        free, rem = np.ones(self.M, bool), self.beds.copy()
        amb, hosp = np.empty(self.N, int), np.empty(self.N, int)
        for i in order:
            a = int(np.argmin(np.where(free, self.T_ap[:, i], np.inf)))
            amb[i], free[a] = a, False
            cand = np.where(rem[:, self.spec[i]] > 0)[0]
            k = int(cand[np.argmin(self.T_ph[i, cand])]) if len(cand) else int(np.argmin(self.T_ph[i]))
            hosp[i] = k
            rem[k, self.spec[i]] = max(0, rem[k, self.spec[i]] - 1)
        return amb, hosp

    def ann_only(self):
        """Priority by ANN risk only; ignores fuzzy traffic/shock urgency."""
        return self._priority_greedy(np.argsort(-self.p))

    # ---- evolution ------------------------------------------------------------------------
    def evolve(self) -> dict:
        t0 = time.perf_counter()
        rng, P, N, M, K, E = self.rng, self.P, self.N, self.M, self.K, self.E
        amb = np.array([rng.permutation(M) for _ in range(P)])
        hosp = rng.integers(0, K, (P, N))
        s_amb, s_h = self._priority_greedy(self.order)
        rest = np.setdiff1d(np.arange(M), s_amb)
        amb[0] = np.concatenate([s_amb, rng.permutation(rest)])
        hosp[0] = s_h
        for j in range(1, max(2, P // 4)):
            h = s_h.copy()
            m = rng.random(N) < 0.3
            h[m] = rng.integers(0, K, m.sum())
            hosp[j] = h
        hosp = np.array([self.repair(h) for h in hosp])

        best_hist, mean_hist = [], []
        for _ in range(self.G):
            fit = self.evaluate(amb[:, :N], hosp)
            order = np.argsort(fit)
            best_hist.append(float(fit[order[0]]))
            mean_hist.append(float(fit.mean()))
            cand = rng.integers(0, P, (P, self.tk))
            sel = cand[np.arange(P), np.argmin(fit[cand], axis=1)]  # tournament selection
            nc = P - E
            ca, ch = np.empty((nc, M), int), np.empty((nc, N), int)
            for j in range(nc):
                p1, p2 = sel[j], sel[(j + 1) % P]
                if rng.random() < self.pc:
                    a = pmx(amb[p1], amb[p2], rng)                       # PMX on ambulance permutation
                    h = np.where(rng.random(N) < 0.5, hosp[p1], hosp[p2])  # uniform crossover on hospitals
                else:
                    a, h = amb[p1].copy(), hosp[p1].copy()
                if rng.random() < self.pm:                               # swap mutation (ambulances)
                    i, k = rng.choice(M, 2, replace=False)
                    a[i], a[k] = a[k], a[i]
                if rng.random() < self.pm:                               # swap mutation (hospitals)
                    i, k = rng.choice(N, 2, replace=False)
                    h[i], h[k] = h[k], h[i]
                if rng.random() < self.pm:                               # reassignment mutation
                    h[rng.integers(N)] = rng.integers(K)
                ca[j], ch[j] = a, self.repair(h)
            amb = np.vstack([amb[order[:E]], ca])
            hosp = np.vstack([hosp[order[:E]], ch])

        fit = self.evaluate(amb[:, :N], hosp)
        b = int(np.argmin(fit))
        best_hist.append(float(fit[b]))
        mean_hist.append(float(fit.mean()))
        return {"amb": amb[b, :N].copy(), "hosp": hosp[b].copy(), "best_history": best_hist,
                "mean_history": mean_hist, "runtime_s": time.perf_counter() - t0}
