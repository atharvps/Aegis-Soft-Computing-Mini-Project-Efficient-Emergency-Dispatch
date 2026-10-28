"""Pillar 2 - Dual-head MLP triage network (critical-risk head + KTAS-level head)
trained on ALL features vs ROUGH-SET REDUCT features, benchmarked against nurses."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_loader import CONDITION_ATTRS, NURSE, TARGET

warnings.filterwarnings("ignore")


def _mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                         learning_rate_init=0.003, alpha=1e-4, max_iter=400, early_stopping=True,
                         n_iter_no_change=15, random_state=seed)


def _n_params(m: MLPClassifier) -> int:
    return int(sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_))


def _balance(X, y, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(small, len(big) - len(small), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


class DualHeadTriageMLP:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reduct: list[str] = []
        self.test_idx = np.array([], dtype=int)
        self.report_: dict = {}

    def _train(self, df, feats, tr, te):
        rng = np.random.default_rng(self.seed)
        X = df[feats].to_numpy(float)
        y = df["critical"].to_numpy(int)
        sc = StandardScaler().fit(X[tr])
        Xtr, ytr = _balance(sc.transform(X[tr]), y[tr], rng)  # balanced prior -> spread-out P_crit
        t0 = time.perf_counter()
        m = _mlp(self.seed).fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        Xte = sc.transform(X[te])
        p = m.predict_proba(Xte)[:, list(m.classes_).index(1)]
        pred = (p >= 0.5).astype(int)
        t0 = time.perf_counter()
        for _ in range(20):
            m.predict_proba(Xte)
        lat = (time.perf_counter() - t0) / 20 / max(len(te), 1) * 1000
        yt = y[te]
        metrics = {
            "n_features": len(feats), "params": _n_params(m), "train_time_s": train_s,
            "latency_ms_per_patient": lat, "accuracy": accuracy_score(yt, pred),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0), "f1": f1_score(yt, pred, zero_division=0),
            "roc_auc": roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else float("nan"),
        }
        return sc, m, p, pred, metrics

    def fit(self, df: pd.DataFrame, reduct: list[str]) -> dict:
        self.reduct = list(reduct)
        y = df["critical"].to_numpy(int)
        tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=self.seed, stratify=y)
        self.test_idx = te

        _, _, _, _, m_full = self._train(df, list(CONDITION_ATTRS), tr, te)
        self.scaler, self.crit, p, pred, m_red = self._train(df, self.reduct, tr, te)

        # Head 2: KTAS level (1..5) on reduct features
        Xr = self.scaler.transform(df[self.reduct].to_numpy(float))
        self.ktas = _mlp(self.seed).fit(Xr[tr], df[TARGET].to_numpy(int)[tr])
        k_pred = self.ktas.predict(Xr[te])
        k_true = df[TARGET].to_numpy(int)[te]

        # ROC + confusion matrix (reduct model)
        yt = y[te]
        fpr, tpr, _ = roc_curve(yt, p)
        sel = np.unique(np.linspace(0, len(fpr) - 1, 40).astype(int))
        cm = confusion_matrix(yt, pred, labels=[0, 1])

        # Nurse benchmark (only rows where a nurse label exists)
        nurse = {"available": False}
        if NURSE in df.columns and df[NURSE].notna().any():
            rn = df[NURSE].to_numpy(float)[te]
            ok = ~np.isnan(rn)
            rn_i = rn[ok].astype(int)
            kt, kp, yc, pc = k_true[ok], k_pred[ok], yt[ok], p[ok]
            n_crit = rn_i <= 2
            m_crit = pc >= 0.5
            crit = yc == 1
            nurse = {
                "available": True, "n": int(ok.sum()),
                "nurse_ktas_accuracy": float((rn_i == kt).mean()),
                "model_ktas_accuracy": float((kp == kt).mean()),
                "nurse_critical_f1": float(f1_score(yc, n_crit.astype(int), zero_division=0)),
                "model_critical_f1": float(f1_score(yc, m_crit.astype(int), zero_division=0)),
                "nurse_critical_recall": float(n_crit[crit].mean()) if crit.any() else 0.0,
                "model_critical_recall": float(m_crit[crit].mean()) if crit.any() else 0.0,
                "nurse_under_triage_pct": float((rn_i > kt).mean() * 100),
                "nurse_over_triage_pct": float((rn_i < kt).mean() * 100),
                "model_under_triage_pct": float((kp > kt).mean() * 100),
                "model_over_triage_pct": float((kp < kt).mean() * 100),
            }

        self.report_ = {
            "architecture": "Input -> Linear(32)+ReLU -> Linear(16)+ReLU -> [P_crit head | KTAS-level head]",
            "train_size": int(len(tr)), "test_size": int(len(te)), "base_rate": float(y.mean()),
            "models": {"full": m_full, "reduct": m_red},
            "ablation": {
                "param_reduction_pct": 100 * (1 - m_red["params"] / m_full["params"]),
                "latency_reduction_pct": 100 * (1 - m_red["latency_ms_per_patient"] / max(m_full["latency_ms_per_patient"], 1e-9)),
                "f1_delta": m_red["f1"] - m_full["f1"], "auc_delta": m_red["roc_auc"] - m_full["roc_auc"],
            },
            "ktas_head_accuracy": float((k_pred == k_true).mean()),
            "roc": [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in sel],
            "confusion": cm.tolist(), "nurse": nurse, "reduct": self.reduct,
        }
        return self.report_

    def predict(self, rows: pd.DataFrame):
        """Returns (P_crit in [0,1], predicted KTAS level 1..5)."""
        X = self.scaler.transform(rows[self.reduct].to_numpy(float))
        p = self.crit.predict_proba(X)[:, list(self.crit.classes_).index(1)]
        return p, self.ktas.predict(X)
