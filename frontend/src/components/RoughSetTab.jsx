import React from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Filter, Table2 } from "lucide-react";
import { Badge, Card, Empty, Stat, chartTheme, fmt } from "./ui.jsx";

export default function RoughSetTab({ status }) {
  const rs = status?.rough_set;
  if (!rs) return <Empty />;
  const sig = [...rs.significance].sort((a, b) => b.mutual_info - a.mutual_info);
  const path = rs.trace.map((t) => ({ step: `${t.step}. ${t.attribute}`, gamma: t.gamma_after }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Stat label="Attributes kept" value={`${rs.n_full} -> ${rs.n_reduct}`} sub={`-${fmt(rs.reduction_pct)}%`} />
        <Stat label="gamma (reduct)" value={fmt(rs.gamma_reduct, 3)} sub={`${fmt(rs.dependency_ratio * 100)}% of peak ${fmt(rs.gamma_peak, 3)}`} tone="green" />
        <Stat label="gamma (all attrs)" value={fmt(rs.gamma_full, 3)} sub="over-fragmented: blocks below min support" tone="amber" />
        <Stat label="Equivalence classes" value={`${rs.equivalence_classes.full} -> ${rs.equivalence_classes.reduct}`} sub={`min support ${rs.min_support}`} tone="violet" />
      </div>

      <Card title="QuickReduct Trace" icon={Table2}>
        <div className="grid gap-4 lg:grid-cols-2">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-slate-400"><tr><th className="py-1">Step</th><th>Attribute added</th><th>gamma before</th><th>gamma after</th><th>Delta</th></tr></thead>
            <tbody>
              {rs.trace.map((t) => (
                <tr key={t.step} className="border-t border-slate-800">
                  <td className="py-1.5">{t.step}</td><td className="font-medium text-cyan-300">{t.attribute}</td>
                  <td className="font-mono">{fmt(t.gamma_before, 3)}</td><td className="font-mono">{fmt(t.gamma_after, 3)}</td>
                  <td className={`font-mono ${t.delta >= 0 ? "text-emerald-400" : "text-red-400"}`}>{t.delta >= 0 ? "+" : ""}{fmt(t.delta, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart data={path} margin={{ left: -10, right: 10 }}>
                <CartesianGrid stroke={chartTheme.grid} /><XAxis dataKey="step" stroke={chartTheme.axis} tick={{ fontSize: 9 }} interval={0} angle={-20} height={55} textAnchor="end" />
                <YAxis stroke={chartTheme.axis} domain={[0, 1]} tick={{ fontSize: 11 }} /><Tooltip contentStyle={chartTheme.tooltip} />
                <Line type="monotone" dataKey="gamma" stroke="#22d3ee" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Decision attribute: high acuity (KTAS 1-2). Dependency gamma counts only equivalence blocks that are pure <i>and</i> hold at least {rs.min_support} patients,
          and candidates are pre-screened by mutual information &ge; {rs.relevance_min_mi} nats so attributes that merely fragment the table cannot win.
          {rs.n_reduct > rs.trace.length ? "" : ` The reduct is floored at ${rs.min_features} attributes so the network is not starved.`}
        </p>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Attribute Relevance (mutual information with decision)" icon={Filter}
          right={<div className="flex gap-2"><Badge color="cyan">reduct</Badge><Badge>dropped</Badge></div>}>
          <div className="h-80">
            <ResponsiveContainer>
              <BarChart data={sig} layout="vertical" margin={{ left: 40, right: 10 }}>
                <CartesianGrid stroke={chartTheme.grid} horizontal={false} /><XAxis type="number" stroke={chartTheme.axis} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="attribute" stroke={chartTheme.axis} tick={{ fontSize: 11 }} width={130} />
                <Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="mutual_info" name="MI (nats)">{sig.map((s) => <Cell key={s.attribute} fill={s.selected ? "#22d3ee" : "#475569"} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Pawlak Approximations under the Reduct">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-slate-400"><tr><th className="py-1">Decision class</th><th>|X|</th><th>Lower</th><th>Upper</th><th>Accuracy</th></tr></thead>
            <tbody>
              {rs.approximations.map((a) => (
                <tr key={a.class} className="border-t border-slate-800"><td className="py-1.5">{a.class}</td><td>{a.size}</td><td className="font-mono">{a.lower}</td><td className="font-mono">{a.upper}</td><td className="font-mono text-cyan-300">{fmt(a.accuracy * 100)}%</td></tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 text-xs text-slate-400">Selected reduct</div>
          <div className="mt-1 flex flex-wrap gap-1.5">{rs.reduct.map((r) => <Badge key={r} color="cyan">{r}</Badge>)}</div>
          <div className="mt-3 text-xs text-slate-400">Dropped ({rs.dropped.length})</div>
          <div className="mt-1 flex flex-wrap gap-1.5">{rs.dropped.map((r) => <Badge key={r}>{r}</Badge>)}</div>
        </Card>
      </div>
    </div>
  );
}
