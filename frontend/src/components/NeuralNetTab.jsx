import React from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, Stethoscope } from "lucide-react";
import { Card, Empty, Stat, chartTheme, fmt } from "./ui.jsx";

export default function NeuralNetTab({ status }) {
  const nn = status?.neural;
  if (!nn) return <Empty />;
  const { full, reduct } = nn.models;
  const ab = nn.ablation;
  const n = nn.nurse;

  const rows = [
    ["Input features", full.n_features, reduct.n_features],
    ["Parameters", full.params, reduct.params],
    ["Accuracy", fmt(full.accuracy * 100) + "%", fmt(reduct.accuracy * 100) + "%"],
    ["F1 (critical)", fmt(full.f1, 3), fmt(reduct.f1, 3)],
    ["Recall (critical)", fmt(full.recall, 3), fmt(reduct.recall, 3)],
    ["ROC-AUC", fmt(full.roc_auc, 3), fmt(reduct.roc_auc, 3)],
    ["Train time (s)", fmt(full.train_time_s, 2), fmt(reduct.train_time_s, 2)],
    ["Latency (ms/patient)", fmt(full.latency_ms_per_patient, 4), fmt(reduct.latency_ms_per_patient, 4)],
  ];
  const chart = [
    { metric: "Accuracy", All: full.accuracy, Reduct: reduct.accuracy },
    { metric: "F1", All: full.f1, Reduct: reduct.f1 },
    { metric: "Recall", All: full.recall, Reduct: reduct.recall },
    { metric: "ROC-AUC", All: full.roc_auc, Reduct: reduct.roc_auc },
  ];
  const [[tn, fp], [fn, tp]] = nn.confusion;
  const cmCell = (v, good) => (
    <div className={`rounded-lg p-3 text-center ${good ? "bg-emerald-500/15 text-emerald-300" : "bg-red-500/15 text-red-300"}`}>
      <div className="text-xl font-bold">{v}</div>
    </div>
  );
  const nurseChart = n.available ? [
    { metric: "KTAS-level accuracy %", Nurse: n.nurse_ktas_accuracy * 100, Model: n.model_ktas_accuracy * 100 },
    { metric: "Critical recall %", Nurse: n.nurse_critical_recall * 100, Model: n.model_critical_recall * 100 },
    { metric: "Critical F1 %", Nurse: n.nurse_critical_f1 * 100, Model: n.model_critical_f1 * 100 },
    { metric: "Under-triage %", Nurse: n.nurse_under_triage_pct, Model: n.model_under_triage_pct },
    { metric: "Over-triage %", Nurse: n.nurse_over_triage_pct, Model: n.model_over_triage_pct },
  ] : [];

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Stat label="Parameter reduction" value={`-${fmt(ab.param_reduction_pct)}%`} sub={`${full.params} -> ${reduct.params}`} />
        <Stat label="Latency change" value={`${ab.latency_reduction_pct >= 0 ? "-" : "+"}${fmt(Math.abs(ab.latency_reduction_pct))}%`} sub="per-patient inference" tone="green" />
        <Stat label="F1 delta (reduct - all)" value={`${ab.f1_delta >= 0 ? "+" : ""}${fmt(ab.f1_delta * 100, 2)} pt`} sub="on held-out patients" tone="violet" />
        <Stat label="AUC delta" value={`${ab.auc_delta >= 0 ? "+" : ""}${fmt(ab.auc_delta, 4)}`} sub={`test n = ${nn.test_size}`} tone="amber" />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Ablation: All Attributes vs Rough-Set Reduct" icon={Activity}>
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-slate-400"><tr><th className="py-1">Metric</th><th>All features</th><th className="text-cyan-300">Reduct</th></tr></thead>
            <tbody>{rows.map(([k, a, b]) => <tr key={k} className="border-t border-slate-800"><td className="py-1.5">{k}</td><td className="font-mono">{a}</td><td className="font-mono text-cyan-300">{b}</td></tr>)}</tbody>
          </table>
          <div className="mt-4 h-52">
            <ResponsiveContainer>
              <BarChart data={chart}><CartesianGrid stroke={chartTheme.grid} /><XAxis dataKey="metric" stroke={chartTheme.axis} tick={{ fontSize: 11 }} />
                <YAxis stroke={chartTheme.axis} domain={[0, 1]} tick={{ fontSize: 11 }} /><Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => v.toFixed(3)} /><Legend />
                <Bar dataKey="All" fill="#64748b" /><Bar dataKey="Reduct" fill="#22d3ee" /></BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="ROC Curve (Reduct model)">
            <div className="h-52">
              <ResponsiveContainer>
                <LineChart data={nn.roc} margin={{ left: -10 }}><CartesianGrid stroke={chartTheme.grid} />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke={chartTheme.axis} tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} stroke={chartTheme.axis} tick={{ fontSize: 11 }} /><Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => v.toFixed(3)} />
                  <Line dataKey="tpr" stroke="#22d3ee" strokeWidth={2} dot={false} /></LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card title="Confusion Matrix (critical = KTAS 1-2)">
            <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-2 text-xs text-slate-400">
              <span /><span className="text-center">Pred. non-critical</span><span className="text-center">Pred. critical</span>
              <span>Actual non-critical</span>{cmCell(tn, true)}{cmCell(fp, false)}
              <span>Actual critical</span>{cmCell(fn, false)}{cmCell(tp, true)}
            </div>
          </Card>
        </div>
      </div>

      <Card title="Human Nurse Benchmark (test patients with nurse label)" icon={Stethoscope}>
        {!n.available ? <p className="text-sm text-slate-500">No KTAS_RN column found in the dataset.</p> : (
          <div className="h-64">
            <ResponsiveContainer>
              <BarChart data={nurseChart}><CartesianGrid stroke={chartTheme.grid} /><XAxis dataKey="metric" stroke={chartTheme.axis} tick={{ fontSize: 11 }} />
                <YAxis stroke={chartTheme.axis} tick={{ fontSize: 11 }} /><Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => v.toFixed(1)} /><Legend />
                <Bar dataKey="Nurse" fill="#f59e0b" /><Bar dataKey="Model" fill="#22d3ee" /></BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <p className="mt-2 text-xs text-slate-500">
          Ground truth is the expert KTAS label. Under-triage = assigned less urgent than the expert; over-triage = more urgent. The KTAS-level head uses only the reduct features.
          {n.available && ` Evaluated on ${n.n} patients.`}
        </p>
      </Card>
    </div>
  );
}
