import React, { useCallback, useEffect, useState } from "react";
import { Siren, Filter, Brain, Waves, Dna, AlertTriangle, Loader2 } from "lucide-react";
import { api } from "./api.js";
import ControlPanel, { DEFAULT_CONTROLS } from "./components/ControlPanel.jsx";
import DashboardMap from "./components/DashboardMap.jsx";
import RoughSetTab from "./components/RoughSetTab.jsx";
import NeuralNetTab from "./components/NeuralNetTab.jsx";
import FuzzyTab from "./components/FuzzyTab.jsx";
import GaTab from "./components/GaTab.jsx";

const TABS = [
  { id: "map", label: "Live Dispatch Map", icon: Siren },
  { id: "rough", label: "Rough Set Reduction", icon: Filter },
  { id: "nn", label: "Neural Triage Net", icon: Brain },
  { id: "fuzzy", label: "Fuzzy Urgency", icon: Waves },
  { id: "ga", label: "GA & Ablation", icon: Dna },
];

export default function App() {
  const [tab, setTab] = useState("map");
  const [status, setStatus] = useState(null);
  const [sim, setSim] = useState(null);
  const [controls, setControls] = useState({ ...DEFAULT_CONTROLS });
  const [loading, setLoading] = useState(false);
  const [booting, setBooting] = useState(true);
  const [error, setError] = useState("");

  const run = useCallback(async (c) => {
    setLoading(true); setError("");
    try { setSim(await api.simulate(c)); } catch (e) { setError(e.message); } finally { setLoading(false); }
  }, []);

  // Boot: wait for the backend to finish training (retries while it answers 503 / is cold-starting)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let i = 0; i < 40 && !cancelled; i++) {
        try {
          const s = await api.status();
          if (cancelled) return;
          setStatus(s); setBooting(false);
          await run(DEFAULT_CONTROLS);
          return;
        } catch (e) {
          if (i === 39) { setError(`Backend unreachable: ${e.message}`); setBooting(false); }
          await new Promise((r) => setTimeout(r, 3000));
        }
      }
    })();
    return () => { cancelled = true; };
  }, [run]);

  const inject = async (patient) => {
    setLoading(true); setError("");
    try { setSim(await api.inject(patient)); setTab("map"); } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const src = status?.dataset?.source;
  return (
    <div className="mx-auto max-w-[1500px] p-4 lg:p-6">
      <header className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-red-500/15 p-2.5"><Siren className="text-red-400" size={24} /></div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">AEGIS-SC <span className="font-normal text-slate-400">Golden-Hour Command Center</span></h1>
            <p className="text-xs text-slate-500">Rough Sets &rarr; Neural Net &rarr; Mamdani Fuzzy &rarr; Genetic Algorithm</p>
          </div>
        </div>
        {status && (
          <div className={`rounded-lg px-3 py-1.5 text-xs ${src === "kaggle_ktas" ? "bg-emerald-500/15 text-emerald-300" : "bg-amber-500/15 text-amber-300"}`}>
            {src === "kaggle_ktas" ? `Kaggle KTAS data - ${status.dataset.rows} patients` : "Synthetic fallback data (data.csv not found)"}
          </div>
        )}
      </header>

      {error && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" /> <span>{error}</span>
        </div>
      )}

      {booting ? (
        <div className="flex items-center justify-center gap-3 rounded-xl border border-slate-800 p-16 text-slate-400">
          <Loader2 className="animate-spin" /> Connecting to backend and training models (first start can take ~30 s on free hosting)...
        </div>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[300px_1fr]">
          <ControlPanel status={status} controls={controls} setControls={setControls} loading={loading}
            onRun={() => run(controls)} onInject={inject} onReset={() => run({ ...DEFAULT_CONTROLS })} />
          <main className="min-w-0">
            <nav className="mb-4 flex flex-wrap gap-2">
              {TABS.map(({ id, label, icon: Icon }) => (
                <button key={id} onClick={() => setTab(id)}
                  className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition ${tab === id ? "bg-cyan-500 text-slate-950" : "bg-slate-900 text-slate-300 hover:bg-slate-800"}`}>
                  <Icon size={15} /> {label}
                </button>
              ))}
              {loading && <span className="ml-auto flex items-center gap-2 text-xs text-slate-400"><Loader2 size={14} className="animate-spin" /> Running pipeline...</span>}
            </nav>
            {tab === "map" && <DashboardMap sim={sim} status={status} />}
            {tab === "rough" && <RoughSetTab status={status} />}
            {tab === "nn" && <NeuralNetTab status={status} />}
            {tab === "fuzzy" && <FuzzyTab sim={sim} />}
            {tab === "ga" && <GaTab sim={sim} />}
          </main>
        </div>
      )}
    </div>
  );
}
