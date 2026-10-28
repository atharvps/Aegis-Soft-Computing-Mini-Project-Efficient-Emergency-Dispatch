import React from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Dna, Scale } from "lucide-react";
import { Card, Empty, Stat, chartTheme, fmt } from "./ui.jsx";

const NAMES = { greedy_fcfs: "Greedy FCFS", ann_only: "ANN-only", aegis: "Full Aegis-SC" };
const COLORS = { greedy_fcfs: "#64748b", ann_only: "#a78bfa", aegis: "#22d3ee" };

function Mini({ title, metric, ab, digits = 1 }) {
  const data = Object.keys(NAMES).map((k) => ({ name: NAMES[k], value: ab[k][metric], key: k }));
  return (
    <div>
      <div className="mb-1 text-xs font-medium text-slate-400">{title}</div>
      <div className="h-44">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ left: -15, right: 5 }}>
            <CartesianGrid stroke={chartTheme.grid} /><XAxis dataKey="name" stroke={chartTheme.axis} tick={{ fontSize: 10 }} />
            <YAxis stroke={chartTheme.axis} tick={{ fontSize: 10 }} /><Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => Number(v).toFixed(digits)} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} shape={(p) => <rect {...p} fill={COLORS[p.payload.key]} />} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function GaTab({ sim }) {
  if (!sim) return <Empty text="Run the pipeline first." />;
  const { ablation: ab, kpis: k, params } = sim;
  const conv = sim.convergence.best.map((b, i) => ({ gen: i, Best: b, Mean: sim.convergence.mean[i] }));
  const metrics = [
    ["Total cost J", "cost", 1], ["Weighted mortality exposure", "mortality_exposure", 1], ["Avg critical total time (min)", "avg_critical_wait", 1],
    ["Bed overflow violations", "bed_violations", 0], ["Specialty mismatches", "specialty_mismatches", 0], ["Golden-hour survival (%)", "golden_hour_survival_pct", 0],
  ];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Stat label="Cost vs Greedy FCFS" value={`-${fmt(k.cost_improvement_vs_greedy_pct)}%`} sub="multi-objective J(C)" />
        <Stat label="Mortality exposure vs FCFS" value={`${k.exposure_reduction_vs_greedy_pct >= 0 ? "-" : "+"}${fmt(Math.abs(k.exposure_reduction_vs_greedy_pct))}%`} sub="sum of U/100 x time" tone="green" />
        <Stat label="Mortality exposure vs ANN-only" value={`${k.exposure_reduction_vs_ann_pct >= 0 ? "-" : "+"}${fmt(Math.abs(k.exposure_reduction_vs_ann_pct))}%`} sub="value of fuzzy urgency" tone="violet" />
        <Stat label="GA runtime" value={`${fmt(k.ga_runtime_s, 2)} s`} sub={`pop ${params.population} x ${params.generations} gens`} tone="amber" />
      </div>

      <Card title="GA Fitness Convergence" icon={Dna}>
        <div className="h-64">
          <ResponsiveContainer>
            <LineChart data={conv} margin={{ left: -5, right: 10 }}>
              <CartesianGrid stroke={chartTheme.grid} /><XAxis dataKey="gen" stroke={chartTheme.axis} tick={{ fontSize: 11 }} label={{ value: "generation", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }} />
              <YAxis stroke={chartTheme.axis} tick={{ fontSize: 11 }} /><Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => v.toFixed(1)} /><Legend />
              <Line dataKey="Best" stroke="#22d3ee" strokeWidth={2} dot={false} /><Line dataKey="Mean" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-slate-500">Chromosome = ambulance permutation (PMX crossover, zero duplicate dispatches) + hospital genes (uniform crossover, swap mutation, greedy bed-capacity repair). Elitism keeps the top 6.</p>
      </Card>

      <Card title="3-Way Ablation: Greedy FCFS vs ANN-only vs Full Aegis-SC" icon={Scale}>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <Mini title="Avg critical total time (min)" metric="avg_critical_wait" ab={ab} />
          <Mini title="Bed overflow violations" metric="bed_violations" ab={ab} digits={0} />
          <Mini title="Golden-hour survival (%)" metric="golden_hour_survival_pct" ab={ab} digits={0} />
        </div>
        <table className="mt-4 w-full text-left text-sm">
          <thead className="text-xs text-slate-400"><tr><th className="py-1.5">Metric</th>{Object.values(NAMES).map((n) => <th key={n}>{n}</th>)}</tr></thead>
          <tbody>
            {metrics.map(([label, key, d]) => (
              <tr key={key} className="border-t border-slate-800"><td className="py-1.5">{label}</td>
                {Object.keys(NAMES).map((n) => <td key={n} className={`font-mono ${n === "aegis" ? "text-cyan-300" : ""}`}>{fmt(ab[n][key], d)}</td>)}</tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-slate-500">All three plans are scored with the same true-traffic cost function and the same urgency scores. Results vary by scenario and seed; they are measured on this simulation, not on real dispatch data.</p>
      </Card>
    </div>
  );
}
