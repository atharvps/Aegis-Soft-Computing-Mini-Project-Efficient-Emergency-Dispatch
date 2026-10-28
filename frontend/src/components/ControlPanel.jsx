import React, { useState } from "react";
import { Play, Plus, RotateCcw, Loader2, SlidersHorizontal, Stethoscope } from "lucide-react";
import { Card } from "./ui.jsx";

export const DEFAULT_CONTROLS = { scenario: "peak", n_calls: 15, traffic: null, seed: 42, population: 80, generations: 60, mutation_rate: 0.12 };

const DEFAULT_PATIENT = { age: 58, sbp: 118, dbp: 76, hr: 96, rr: 20, bt: 36.9, saturation: 96, mental: 1, injury: 1, chief_complain: "chest pain" };

function Slider({ label, value, min, max, step = 1, onChange, unit = "" }) {
  return (
    <label className="block text-xs text-slate-400">
      <div className="mb-1 flex justify-between"><span>{label}</span><span className="font-mono text-slate-200">{value}{unit}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-cyan-400" />
    </label>
  );
}

export default function ControlPanel({ status, controls, setControls, onRun, onInject, onReset, loading }) {
  const [useTraffic, setUseTraffic] = useState(false);
  const [showGa, setShowGa] = useState(false);
  const [pt, setPt] = useState(DEFAULT_PATIENT);
  const set = (k, v) => setControls((c) => ({ ...c, [k]: v }));
  const setP = (k, v) => setPt((p) => ({ ...p, [k]: v }));
  const scenarios = status?.scenarios || { normal: "Normal City Load", peak: "Peak Monsoon Gridlock", pileup: "Multi-Vehicle Highway Pileup" };

  return (
    <aside className="space-y-4">
      <Card title="Scenario Controls" icon={SlidersHorizontal}>
        <div className="space-y-3">
          <select value={controls.scenario} onChange={(e) => { set("scenario", e.target.value); setUseTraffic(false); set("traffic", null); }}
            className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
            {Object.entries(scenarios).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <Slider label="Active distress calls" value={controls.n_calls} min={8} max={25} onChange={(v) => set("n_calls", v)} />
          <label className="flex items-center gap-2 text-xs text-slate-400">
            <input type="checkbox" checked={useTraffic} onChange={(e) => { setUseTraffic(e.target.checked); set("traffic", e.target.checked ? 2.0 : null); }} className="accent-cyan-400" />
            Override scenario traffic multiplier
          </label>
          {useTraffic && <Slider label="Traffic multiplier" value={controls.traffic ?? 2.0} min={1} max={3} step={0.1} unit="x" onChange={(v) => set("traffic", v)} />}
          <button onClick={() => setShowGa((s) => !s)} className="text-xs text-cyan-400 hover:underline">{showGa ? "Hide" : "Show"} GA hyper-parameters</button>
          {showGa && (
            <div className="space-y-3 rounded-lg border border-slate-800 p-3">
              <Slider label="Population" value={controls.population} min={20} max={200} step={10} onChange={(v) => set("population", v)} />
              <Slider label="Generations" value={controls.generations} min={10} max={200} step={10} onChange={(v) => set("generations", v)} />
              <Slider label="Mutation rate" value={controls.mutation_rate} min={0.01} max={0.5} step={0.01} onChange={(v) => set("mutation_rate", v)} />
            </div>
          )}
          <button disabled={loading} onClick={onRun}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-500 px-3 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50">
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />} Run Full 4-Pillar Pipeline
          </button>
          <button disabled={loading} onClick={() => { setControls({ ...DEFAULT_CONTROLS }); setUseTraffic(false); onReset(); }}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50">
            <RotateCcw size={14} /> Reset Seed &amp; Telemetry
          </button>
        </div>
      </Card>

      <Card title="Live Patient Injector" icon={Stethoscope}>
        <div className="space-y-3">
          <Slider label="Age" value={pt.age} min={1} max={95} onChange={(v) => setP("age", v)} />
          <Slider label="Systolic BP" value={pt.sbp} min={50} max={230} unit=" mmHg" onChange={(v) => setP("sbp", v)} />
          <Slider label="Diastolic BP" value={pt.dbp} min={25} max={140} unit=" mmHg" onChange={(v) => setP("dbp", v)} />
          <Slider label="Heart rate" value={pt.hr} min={30} max={190} unit=" bpm" onChange={(v) => setP("hr", v)} />
          <Slider label="Resp. rate" value={pt.rr} min={6} max={50} unit="/min" onChange={(v) => setP("rr", v)} />
          <Slider label="Body temp" value={pt.bt} min={34} max={41} step={0.1} unit=" C" onChange={(v) => setP("bt", v)} />
          <Slider label="SpO2" value={pt.saturation} min={60} max={100} unit="%" onChange={(v) => setP("saturation", v)} />
          <Slider label="Mental (1 alert - 4 unresponsive)" value={pt.mental} min={1} max={4} onChange={(v) => setP("mental", v)} />
          <div className="flex items-center gap-2">
            <select value={pt.injury} onChange={(e) => setP("injury", Number(e.target.value))} className="rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs">
              <option value={1}>Non-injury</option><option value={2}>Injury</option>
            </select>
            <input value={pt.chief_complain} onChange={(e) => setP("chief_complain", e.target.value)} placeholder="Chief complaint"
              className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-950 px-2 py-1.5 text-xs" />
          </div>
          <button disabled={loading} onClick={() => onInject(pt)}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-violet-500 px-3 py-2.5 text-sm font-semibold text-white hover:bg-violet-400 disabled:opacity-50">
            <Plus size={16} /> Inject Custom Live Patient
          </button>
        </div>
      </Card>
    </aside>
  );
}
