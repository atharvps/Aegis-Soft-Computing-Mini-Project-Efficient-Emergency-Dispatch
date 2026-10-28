"""Aegis-SC data layer: Kaggle KTAS loader, imputation, MEWS binning, synthetic fallback.

load_dataset() NEVER raises: if data.csv is missing/corrupt it falls back to a
deterministic synthetic generator with the same schema.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

SPECIALTIES = ["General", "Cardiac", "Neuro", "Trauma"]
TARGET, NURSE = "KTAS_expert", "KTAS_RN"

# Condition attributes available to a paramedic / triage nurse (no outcome leakage:
# Disposition, Length of stay, KTAS_RN, Error_group, mistriage are deliberately excluded).
CONDITION_ATTRS = [
    "Sex", "Age", "Patients number per hour", "Arrival mode", "Injury", "Mental",
    "Pain", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation", "Group",
    "Shock_Index", "Complaint_group",
]
NUMERIC_RAW = [
    "Group", "Sex", "Age", "Patients number per hour", "Arrival mode", "Injury", "Mental",
    "Pain", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation", NURSE, TARGET,
]
CATEGORICAL = ["Group", "Sex", "Arrival mode", "Injury", "Mental", "Pain"]
DEFAULTS = {
    "Group": 1, "Sex": 1, "Age": 55, "Patients number per hour": 10, "Arrival mode": 3,
    "Injury": 1, "Mental": 1, "Pain": 0, "NRS_pain": 0, "SBP": 125, "DBP": 78, "HR": 82,
    "RR": 18, "BT": 36.7, "Saturation": 97,
}
RANGES = {
    "Age": (0, 110), "NRS_pain": (0, 10), "SBP": (40, 260), "DBP": (20, 160),
    "HR": (20, 220), "RR": (4, 60), "BT": (32, 43), "Saturation": (50, 100),
}
NA_TOKENS = ["??", "#BO?", "?", "", "NA", "N/A", "nan", "NaN", "-"]
_KNOWN = NUMERIC_RAW + ["Chief_complain", "Disposition", "Error_group", "Length of stay_min",
                        "KTAS duration_min", "mistriage"]
CANON = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in _KNOWN}

KEYWORDS = {
    1: ["chest", "cardiac", "palpit", "syncope", "angina", "heart", "dyspnea", "shortness", "breath"],
    2: ["head", "dizz", "seiz", "stroke", "neuro", "conscious", "weak", "numb", "vertigo",
        "convuls", "mental", "speech", "paralys", "coma"],
    3: ["trauma", "fall", "injur", "fract", "lacer", "burn", "wound", "accident", "crash",
        "bleed", "contusion", "sprain", "stab", "cut"],
}


def complaint_group(text) -> int:
    t = str(text or "").lower()
    for g in (1, 2, 3):
        if any(k in t for k in KEYWORDS[g]):
            return g
    return 0


# --------------------------------------------------------------------------- loading
def _read_csv(path: str):
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                first = f.readline()
            sep = ";" if first.count(";") > first.count(",") else ","
            df = pd.read_csv(path, sep=sep, encoding=enc, na_values=NA_TOKENS, low_memory=False)
            return df, sep
        except Exception as exc:  # try next encoding
            last = exc
    raise last  # type: ignore[misc]


def _clip_ranges(df: pd.DataFrame) -> pd.DataFrame:
    for c, (lo, hi) in RANGES.items():
        df[c] = df[c].clip(lo, hi)
    df["Mental"] = df["Mental"].round().clip(1, 4)
    df["Injury"] = df["Injury"].round().clip(1, 2)
    df["Pain"] = df["Pain"].round().clip(0, 1)
    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df["Shock_Index"] = (df["HR"] / df["SBP"]).clip(0.2, 2.5)
    df["Complaint_group"] = df["Chief_complain"].map(complaint_group).astype(int)
    df["Specialty"] = np.where((df["Injury"] == 2) | (df["Complaint_group"] == 3), 3,
                               df["Complaint_group"]).astype(int)
    return df


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [CANON.get(re.sub(r"[^a-z0-9]", "", str(c).lower()), str(c).strip()) for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} column not found")
    for c in NUMERIC_RAW:
        if c in df.columns:
            s = df[c]
            if s.dtype == object:
                s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
            df[c] = pd.to_numeric(s, errors="coerce")
        else:
            df[c] = np.nan
    df = df[df[TARGET].between(1, 5)].copy()
    if len(df) < 150:
        raise ValueError("too few valid rows")
    df.loc[df["Pain"].eq(0) & df["NRS_pain"].isna(), "NRS_pain"] = 0
    for c, d in DEFAULTS.items():
        if c in CATEGORICAL:
            m = df[c].mode()
            fill = m.iloc[0] if len(m) else d
        else:
            fill = df[c].median()
            if pd.isna(fill):
                fill = d
        df[c] = df[c].fillna(fill)
    df["Chief_complain"] = (df["Chief_complain"] if "Chief_complain" in df.columns
                            else pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    df = _clip_ranges(df)
    df = _derive(df)
    df["critical"] = (df[TARGET] <= 2).astype(int)
    return df.reset_index(drop=True)


def row_from_patient(p: dict) -> pd.DataFrame:
    """Build one prepared row from an API patient payload (lower-case keys)."""
    g = lambda k, d: p.get(k) if p.get(k) is not None else d
    row = {
        "Group": 1, "Sex": g("sex", 1), "Age": g("age", 55), "Patients number per hour": 10,
        "Arrival mode": g("arrival_mode", 3), "Injury": g("injury", 1), "Mental": g("mental", 1),
        "Pain": g("pain", 0), "NRS_pain": g("nrs_pain", 0), "SBP": g("sbp", 120), "DBP": g("dbp", 80),
        "HR": g("hr", 80), "RR": g("rr", 18), "BT": g("bt", 36.8), "Saturation": g("saturation", 97),
        "Chief_complain": str(g("chief_complain", "")).lower(), NURSE: np.nan, TARGET: np.nan,
    }
    df = _clip_ranges(pd.DataFrame([row]))
    df = _derive(df)
    df["critical"] = 0
    return df


# --------------------------------------------------------------------------- MEWS discretisation
def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Modified-Early-Warning-style bands: 0 normal, 1 borderline, 2 critical."""
    d = pd.DataFrame(index=df.index)
    a = lambda c: df[c].to_numpy(float)
    sat, sbp, dbp, hr, rr, bt = a("Saturation"), a("SBP"), a("DBP"), a("HR"), a("RR"), a("BT")
    d["Saturation"] = np.select([sat >= 95, sat >= 90], [0, 1], 2)
    d["SBP"] = np.select([(sbp >= 100) & (sbp <= 139), (sbp >= 90) & (sbp <= 179)], [0, 1], 2)
    d["DBP"] = np.select([(dbp >= 60) & (dbp <= 89), (dbp >= 50) & (dbp <= 109)], [0, 1], 2)
    d["HR"] = np.select([(hr >= 60) & (hr <= 100), (hr >= 50) & (hr <= 129)], [0, 1], 2)
    d["RR"] = np.select([(rr >= 12) & (rr <= 20), (rr > 20) & (rr <= 28)], [0, 1], 2)
    d["BT"] = np.select([(bt >= 36) & (bt < 38), (bt >= 35) & (bt < 39)], [0, 1], 2)
    d["Mental"] = np.select([a("Mental") <= 1, a("Mental") == 2], [0, 1], 2)
    d["Age"] = np.select([a("Age") < 45, a("Age") < 70], [0, 1], 2)
    d["NRS_pain"] = np.select([a("NRS_pain") <= 3, a("NRS_pain") <= 6], [0, 1], 2)
    d["Shock_Index"] = np.select([a("Shock_Index") < 0.7, a("Shock_Index") <= 1.0], [0, 1], 2)
    d["Patients number per hour"] = np.select(
        [a("Patients number per hour") <= 10, a("Patients number per hour") <= 20], [0, 1], 2)
    for c in ("Sex", "Arrival mode", "Injury", "Pain", "Group", "Complaint_group"):
        d[c] = df[c].astype(int)
    return d[CONDITION_ATTRS].astype(int)


# --------------------------------------------------------------------------- synthetic fallback
def synthetic(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = rng.beta(1.2, 2.6, n)
    ktas = np.select([sev > 0.88, sev > 0.70, sev > 0.42, sev > 0.20], [1, 2, 3, 4], 5)
    sgn = np.where(rng.random(n) < 0.8, -1, 1)
    sbp = 125 + rng.normal(0, 13, n) + sgn * 58 * sev ** 2
    mental = 1 + (rng.random(n) < 1.1 * sev ** 2).astype(int) + (rng.random(n) < np.where(sev > 0.85, 0.7, 0.02)).astype(int)
    pain = (rng.random(n) < 0.55).astype(int)
    grp = rng.choice(4, n, p=[0.55, 0.15, 0.12, 0.18])
    texts = {
        0: ["abdominal pain", "fever", "vomiting", "back pain", "cough"],
        1: ["chest pain", "palpitation", "dyspnea", "syncope"],
        2: ["headache", "dizziness", "seizure", "weakness", "stroke symptoms"],
        3: ["fall injury", "laceration", "road accident", "fracture", "burn"],
    }
    chief = [texts[g][rng.integers(len(texts[g]))] for g in grp]
    noise = rng.choice([-1, 0, 1], n, p=[0.10, 0.78, 0.12])
    raw = pd.DataFrame({
        "Group": rng.choice([1, 2], n), "Sex": rng.choice([1, 2], n),
        "Age": np.clip(rng.normal(52, 20, n) + 10 * sev, 1, 95).round(),
        "Patients number per hour": rng.integers(2, 25, n), "Arrival mode": rng.integers(1, 8, n),
        "Injury": np.where(rng.random(n) < 0.22 + 0.25 * sev, 2, 1), "Mental": np.clip(mental, 1, 4),
        "Pain": pain, "NRS_pain": np.where(pain == 1, np.clip(rng.normal(4 + 3 * sev, 1.8), 1, 10).round(), 0),
        "SBP": sbp, "DBP": sbp * 0.62 + rng.normal(0, 7, n),
        "HR": 80 + rng.normal(0, 9, n) + 75 * sev ** 1.6 - (rng.random(n) < 0.05) * 45,
        "RR": 17 + rng.normal(0, 2.2, n) + 17 * sev ** 1.5,
        "BT": 36.7 + rng.normal(0, 0.35, n) + 1.8 * sev * (rng.random(n) < 0.45),
        "Saturation": np.minimum(100, 98.5 - 17 * sev ** 1.8 + rng.normal(0, 1.1, n)),
        "Chief_complain": chief, NURSE: np.clip(ktas + noise, 1, 5), TARGET: ktas,
    })
    return raw


def load_dataset(path: str | None = None):
    """Returns (prepared_df, meta). Never raises."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path, os.getenv("DATA_PATH"), "data.csv", "data/data.csv",
                  os.path.join(here, "data.csv"), os.path.join(here, "data", "data.csv")]
    reason = "data.csv not found"
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                raw, sep = _read_csv(p)
                df = prepare(raw)
                return df, {"source": "kaggle_ktas", "path": p, "delimiter": sep,
                            "raw_rows": int(len(raw)), "raw_columns": int(raw.shape[1]),
                            "rows": int(len(df)), "has_nurse": bool(df[NURSE].notna().any())}
            except Exception as exc:  # fall through to synthetic
                reason = f"failed to parse {p}: {exc}"
    df = prepare(synthetic())
    return df, {"source": "synthetic_fallback", "reason": reason, "rows": int(len(df)),
                "raw_columns": 24, "has_nurse": True, "delimiter": None}
