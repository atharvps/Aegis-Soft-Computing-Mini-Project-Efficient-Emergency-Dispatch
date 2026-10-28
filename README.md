# Aegis-SC — Rough → Neuro → Fuzzy → Evolutionary Emergency Dispatch

Pipeline: **Rough Set QuickReduct** (feature reduction) → **MLP** (critical risk + KTAS level) →
**Mamdani FIS** (risk × traffic × shock → urgency U) → **Constrained GA** (ambulance permutation + hospital beds).

```
backend/   FastAPI + pandas + scikit-learn   (data_loader, rough_set_engine, neural_triage_net,
                                              fuzzy_urgency_controller, ga_dispatch_optimizer, main)
frontend/  React + Vite + Tailwind + Recharts + Leaflet
```

## Dataset
Put the Kaggle **Emergency Service Triage Application (KTAS)** file at `backend/data/data.csv`
(`;` or `,` delimited, any encoding). Missing values (`??`, `#BO?`) are imputed. If the file is absent or
unparseable the backend silently falls back to a deterministic synthetic generator with the same schema
(the UI shows an amber "Synthetic fallback" badge). Outcome columns (KTAS_RN, Disposition, Length of stay…)
are never used as model inputs; KTAS_RN is used only for the nurse benchmark.

## Run locally
```bash
# backend  (Python 3.10+)
cd backend && python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# frontend (Node 18+)
cd frontend && cp .env.example .env && npm install && npm run dev   # http://localhost:5173
```
Models train once at backend start-up (a few seconds); each simulation then takes well under a second.

## API
| Endpoint | Purpose |
|---|---|
| `GET /api/status` | dataset info, rough-set report, ANN ablation + nurse benchmark |
| `POST /api/simulate` | `{scenario: normal\|peak\|pileup, n_calls, traffic?, seed, population, generations, mutation_rate}` → full dispatch plan, KPIs, 3-way ablation, GA convergence |
| `POST /api/inject-patient` | add a live patient (vitals + complaint) to the current scenario and re-run all pillars |
| `GET /api/fuzzy-surface?shock=0.9&resolution=21` | membership curves, 27 rules, control surface |
| `GET /api/health` | liveness probe |

## Deploy
**Backend → Render (Docker)**: push the repo, create a Blueprint from `render.yaml` (or a Web Service with
root `backend/`, runtime Docker). Set `CORS_ORIGINS` to your frontend URL. Free instances sleep; the UI retries
for ~2 minutes while the service wakes and trains. Railway/Fly/any Docker host works the same
(`PORT` env var is honoured). Commit `backend/data/data.csv` (or mount it and set `DATA_PATH`).

**Frontend → Vercel/Netlify**: root `frontend/`, build `npm run build`, output `dist`,
env `VITE_API_URL=https://<your-backend>` (no trailing slash).

**Single-container alternative**: build the frontend (`npm run build`) and serve `dist/` with any static host.

## Design notes (viva)
* Rough-set γ counts only pure equivalence blocks with ≥ 5 patients, because with many attributes the table
  shatters into singletons that are trivially "certain". Candidates are screened by mutual information, the stopping
  reference is the peak γ along the greedy path, and the reduct is floored at 6 / capped at 8 attributes.
* Fuzzy engine is pure vectorised NumPy; `scikit-fuzzy` (if installed) only cross-checks memberships and centroid.
* GA: ambulance genes are a full permutation (PMX ⇒ no duplicate ambulance by construction); hospital genes are
  repaired greedily so bed overflow and specialty mismatches are eliminated before fitness is evaluated.
* Specialty (General/Cardiac/Neuro/Trauma) is derived by rules from chief complaint + injury flag, since the Kaggle
  data has no specialty label.
