"""Aegis-SC FastAPI service: Rough Sets -> Neural Net -> Mamdani Fuzzy -> Genetic Algorithm."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_loader import NURSE, SPECIALTIES, TARGET, discretize, load_dataset, row_from_patient
from fuzzy_urgency_controller import MamdaniUrgencyEngine
from ga_dispatch_optimizer import HOSPITALS, ConstrainedDispatchGA, city_points
from neural_triage_net import DualHeadTriageMLP
from rough_set_engine import RoughSetSelector

STATE: dict = {"ctx": None}
LOCK = threading.Lock()
SCENARIOS = {
    "normal": {"label": "Normal City Load", "traffic": 1.3, "injury_boost": 1.0, "bed_factor": 1.0},
    "peak": {"label": "Peak Monsoon Gridlock", "traffic": 2.2, "injury_boost": 1.0, "bed_factor": 0.85},
    "pileup": {"label": "Multi-Vehicle Highway Pileup", "traffic": 2.6, "injury_boost": 3.0, "bed_factor": 0.75},
}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    return o


@asynccontextmanager
async def lifespan(app: FastAPI):
    df, meta = load_dataset()
    disc = discretize(df)
    rs = RoughSetSelector(tolerance=0.995, max_features=8, min_support=5, min_features=6)
    rs_report = rs.fit(disc, df["critical"].to_numpy(),
                       {0: "Non-critical (KTAS 3-5)", 1: "Critical (KTAS 1-2)"})
    nn = DualHeadTriageMLP(seed=42)
    nn_report = nn.fit(df, rs.reduct_)
    fuzzy = MamdaniUrgencyEngine()
    dist = df[TARGET].value_counts().sort_index()
    STATE.update(
        df=df, nn=nn, fuzzy=fuzzy,
        status={
            "ready": True,
            "dataset": {**meta, "ktas_distribution": {f"KTAS {int(k)}": int(v) for k, v in dist.items()},
                        "critical_rate": float(df["critical"].mean())},
            "rough_set": rs_report, "neural": nn_report,
            "fuzzy": {"rules": len(fuzzy.rules), "skfuzzy_check": fuzzy.verify_against_skfuzzy()},
            "hospitals": [{**h, "beds_by_specialty": dict(zip(SPECIALTIES, h["beds"]))} for h in HOSPITALS],
            "specialties": SPECIALTIES, "scenarios": {k: v["label"] for k, v in SCENARIOS.items()},
        },
    )
    yield


app = FastAPI(title="Aegis-SC API", version="1.0.0", lifespan=lifespan)
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"],
                   allow_credentials=False)


class SimRequest(BaseModel):
    scenario: str = "peak"
    n_calls: int = Field(15, ge=5, le=40)
    traffic: Optional[float] = Field(None, ge=1.0, le=3.0)
    seed: int = 42
    population: int = Field(80, ge=10, le=300)
    generations: int = Field(60, ge=5, le=300)
    mutation_rate: float = Field(0.12, ge=0.0, le=1.0)


class PatientIn(BaseModel):
    age: float = 55
    sex: int = 1
    sbp: float = 120
    dbp: float = 80
    hr: float = 80
    rr: float = 18
    bt: float = 36.8
    saturation: float = 97
    mental: int = Field(1, ge=1, le=4)
    pain: int = Field(0, ge=0, le=1)
    nrs_pain: float = 0
    injury: int = Field(1, ge=1, le=2)
    arrival_mode: int = 3
    chief_complain: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    sim: Optional[SimRequest] = None


def _build_context(req: SimRequest) -> dict:
    sc = SCENARIOS.get(req.scenario)
    if sc is None:
        raise HTTPException(400, f"unknown scenario '{req.scenario}'")
    df, nn = STATE["df"], STATE["nn"]
    rng = np.random.default_rng(req.seed)
    pool = nn.test_idx  # patients unseen by the networks
    w = np.where(df.loc[pool, "Injury"].to_numpy() == 2, sc["injury_boost"], 1.0)
    pick = rng.choice(pool, size=min(req.n_calls, len(pool)), replace=False, p=w / w.sum())
    rows = df.loc[pick].reset_index(drop=True)
    n = len(rows)
    base = req.traffic if req.traffic is not None else sc["traffic"]
    beds = np.ceil(np.array([h["beds"] for h in HOSPITALS]) * sc["bed_factor"]).astype(int)
    beds[:, 0] = np.maximum(beds[:, 0], 1)
    return {
        "rows": rows, "pids": [f"P{i + 1:02d}" for i in range(n)], "pat_xy": city_points(rng, n),
        "amb_xy": city_points(rng, max(n + 4, 18), 0.04),
        "tau": np.clip(base * rng.normal(1, 0.12, n), 1.0, 3.0), "beds": beds, "base": base,
        "req": req, "rng_seed": req.seed, "injected": 0,
    }


def _run(ctx: dict, injected_id: Optional[str] = None) -> dict:
    nn, fuzzy, req = STATE["nn"], STATE["fuzzy"], ctx["req"]
    rows = ctx["rows"]
    n = len(rows)
    p, ktas = nn.predict(rows)
    shock = (rows["HR"] / rows["SBP"]).clip(0.3, 2.0).to_numpy()
    U = fuzzy.compute(p, ctx["tau"], shock)
    spec = rows["Specialty"].to_numpy(int)
    hosp_xy = np.array([[h["lat"], h["lon"]] for h in HOSPITALS])
    ga = ConstrainedDispatchGA(U, spec, ctx["tau"], p, ctx["amb_xy"], ctx["pat_xy"], hosp_xy, ctx["beds"],
                               population=req.population, generations=req.generations,
                               mutation=req.mutation_rate, seed=ctx["rng_seed"])
    res = ga.evolve()
    bd = {"aegis": ga.breakdown(res["amb"], res["hosp"]),
          "greedy_fcfs": ga.breakdown(*ga.greedy_fcfs()),
          "ann_only": ga.breakdown(*ga.ann_only())}
    a, h = res["amb"], res["hosp"]
    A = bd["aegis"]

    patients = []
    for i in range(n):
        r = rows.iloc[i]
        k = int(h[i])
        crit = bool(U[i] >= 75)
        patients.append({
            "id": ctx["pids"][i], "lat": ctx["pat_xy"][i][0], "lon": ctx["pat_xy"][i][1],
            "age": r["Age"], "sbp": r["SBP"], "hr": r["HR"], "rr": r["RR"], "bt": r["BT"],
            "spo2": r["Saturation"], "mental": r["Mental"], "injury": r["Injury"],
            "chief_complain": r["Chief_complain"], "p_crit": p[i], "ktas_model": int(ktas[i]),
            "ktas_expert": r.get(TARGET), "ktas_nurse": r.get(NURSE), "specialty": SPECIALTIES[spec[i]],
            "tau": ctx["tau"][i], "shock_index": shock[i], "urgency": U[i], "critical": crit,
            "amb_id": f"A{int(a[i]) + 1:02d}", "amb_lat": ctx["amb_xy"][a[i]][0], "amb_lon": ctx["amb_xy"][a[i]][1],
            "hospital": HOSPITALS[k]["name"], "hosp_lat": HOSPITALS[k]["lat"], "hosp_lon": HOSPITALS[k]["lon"],
            "pickup_min": A["pickup"][i], "transit_min": A["transit"][i], "total_min": A["total"][i],
            "golden_hour_ok": bool(A["total"][i] <= 45), "fuzzy_rules": fuzzy.explain(p[i], ctx["tau"][i], shock[i], 3),
        })
    patients.sort(key=lambda x: -x["urgency"])
    for rank, pt in enumerate(patients, 1):
        pt["rank"] = rank

    assigned = {int(x): ctx["pids"][i] for i, x in enumerate(a)}
    ambulances = [{"id": f"A{j + 1:02d}", "lat": ctx["amb_xy"][j][0], "lon": ctx["amb_xy"][j][1],
                   "assigned_patient": assigned.get(j)} for j in range(len(ctx["amb_xy"]))]
    hospitals = [{"name": H["name"], "lat": H["lat"], "lon": H["lon"],
                  "beds": dict(zip(SPECIALTIES, ctx["beds"][k].tolist())),
                  "assigned": dict(zip(SPECIALTIES, A["assigned"][k].tolist()))} for k, H in enumerate(HOSPITALS)]

    def metrics(b):
        return {k: b[k] for k in ("cost", "avg_critical_wait", "bed_violations", "specialty_mismatches",
                                  "golden_hour_survival_pct", "mortality_exposure")}

    g, an = bd["greedy_fcfs"], bd["ann_only"]
    imp = lambda ref, new: 100.0 * (ref - new) / ref if ref else 0.0
    return _clean({
        "scenario": req.scenario, "scenario_label": SCENARIOS[req.scenario]["label"],
        "params": req.model_dump(), "traffic_base": ctx["base"], "injected_id": injected_id,
        "patients": patients, "ambulances": ambulances, "hospitals": hospitals,
        "kpis": {
            "n_calls": n, "critical_count": A["critical_count"],
            "critical_served_pct": A["golden_hour_survival_pct"],
            "max_critical_total_min": float(A["total"][U >= 75].max()) if (U >= 75).any() else 0.0,
            "cost_improvement_vs_greedy_pct": imp(g["cost"], A["cost"]),
            "exposure_reduction_vs_greedy_pct": imp(g["mortality_exposure"], A["mortality_exposure"]),
            "exposure_reduction_vs_ann_pct": imp(an["mortality_exposure"], A["mortality_exposure"]),
            "ga_runtime_s": res["runtime_s"],
        },
        "ablation": {"greedy_fcfs": metrics(g), "ann_only": metrics(an), "aegis": metrics(A)},
        "convergence": {"best": res["best_history"], "mean": res["mean_history"]},
    })


@app.get("/api/health")
def health():
    return {"ok": "status" in STATE}


@app.get("/api/status")
def status():
    if "status" not in STATE:
        raise HTTPException(503, "models are still training")
    return _clean(STATE["status"])


@app.post("/api/simulate")
def simulate(req: SimRequest):
    if "status" not in STATE:
        raise HTTPException(503, "models are still training")
    with LOCK:
        ctx = _build_context(req)
        STATE["ctx"] = ctx
        return _run(ctx)


@app.post("/api/inject-patient")
def inject_patient(body: PatientIn):
    if "status" not in STATE:
        raise HTTPException(503, "models are still training")
    with LOCK:
        if STATE["ctx"] is None or body.sim is not None:
            STATE["ctx"] = _build_context(body.sim or SimRequest())
        ctx = STATE["ctx"]
        if len(ctx["rows"]) >= 40:
            raise HTTPException(400, "maximum of 40 active calls reached; run a new simulation")
        payload = body.model_dump(exclude={"lat", "lon", "sim"})
        row = row_from_patient(payload)
        ctx["injected"] += 1
        pid = f"X{ctx['injected']:02d}"
        rng = np.random.default_rng(ctx["rng_seed"] + 1000 + ctx["injected"])
        xy = np.array([[body.lat, body.lon]]) if body.lat is not None and body.lon is not None \
            else city_points(rng, 1)
        ctx["rows"] = pd.concat([ctx["rows"], row], ignore_index=True)
        ctx["pids"].append(pid)
        ctx["pat_xy"] = np.vstack([ctx["pat_xy"], xy])
        ctx["tau"] = np.append(ctx["tau"], np.clip(ctx["base"] * rng.normal(1, 0.1), 1.0, 3.0))
        if len(ctx["amb_xy"]) < len(ctx["rows"]) + 3:
            ctx["amb_xy"] = np.vstack([ctx["amb_xy"], city_points(rng, 3, 0.04)])
        return _run(ctx, injected_id=pid)


@app.get("/api/fuzzy-surface")
def fuzzy_surface(shock: float = Query(0.9, ge=0.3, le=2.0), resolution: int = Query(21, ge=5, le=41)):
    if "status" not in STATE:
        raise HTTPException(503, "models are still training")
    f = STATE["fuzzy"]
    xs, ys, z = f.surface(shock, resolution)
    return _clean({"shock": shock, "x_label": "P_crit", "y_label": "Traffic multiplier",
                   "x": xs, "y": ys, "z": z, "memberships": f.membership_curves(), "rules": f.public_rules()})
