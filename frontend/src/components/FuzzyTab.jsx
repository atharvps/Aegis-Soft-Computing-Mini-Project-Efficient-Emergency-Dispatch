import React, { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Grid3x3, ListChecks, Waves } from "lucide-react";
import { api } from "../api.js";
import { Badge, Card, Empty, chartTheme, fmt } from "./ui.jsx";

const COLORS = ["#22d3ee", "#a78bfa", "#f59e0b", "#ef4444", "#22c55e"];
const toSeries = (m) => m.x.map((x, i) => ({ x: Number(x.toFixed(3)), ...Object.fromEntries(Object.entries(m.sets).map(([k, v]) => [k, v[i]])) }));

function MfChart({ title, data, sets }) {
  return (
    <div>
      <div className="mb-1 text-xs font-medium text-slate-400">{title}</div>
      <div className="h-40">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ left: -20, right: 8 }}>
            <CartesianGrid stroke={chartTheme.grid} /><XAxis dataKey="x" stroke={chartTheme.axis} tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 1]} stroke={chartTheme.axis} tick={{ fontSize: 10 }} /><Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => v.toFixed(2)} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {sets.map((s, i) => <Line key={s} dataKey={s} dot={false} stroke={COLORS[i % COLORS.length]} strokeWidth={2} />)}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function FuzzyTab({ sim }) {
  const [shock, setShock] = useState(0.9);
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    const t = setTimeout(() => { api.surface(shock, 21).then(setData).catch((e) => setErr(e.message)); }, 150);
    return () => clearTimeout(t);
  }, [shock]);

  if (err) return <Empty text={err} />;
  if (!data) return <Empty />;
  const m = data.memberships;
  const zmax = 100;
  const color = (z) => `hsl(${Math.round(130 - (z / zmax) * 130)} 75% ${38 + (z / zmax) * 6}%)`;
  const rows = [...data.y].map((_, i) => data.z.length - 1 - i); // traffic high at the top
  const top = sim?.patients?.slice(0, 5) || [];

  return (
    <div className="space-y-4">
      <Card title="Mamdani Membership Functions" icon={Waves}>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <MfChart title="Clinical risk P_crit" data={toSeries(m.risk)} sets={Object.keys(m.risk.sets)} />
          <MfChart title="Traffic multiplier tau" data={toSeries(m.traffic)} sets={Object.keys(m.traffic.sets)} />
          <MfChart title="Shock index HR/SBP" data={toSeries(m.shock)} sets={Object.keys(m.shock.sets)} />
          <MfChart title="Output urgency U" data={toSeries(m.urgency)} sets={Object.keys(m.urgency.sets)} />
        </div>
      </Card>

      <Card title="Control Surface: P_crit x Traffic -> Urgency" icon={Grid3x3}
        right={<label className="flex items-center gap-2 text-xs text-slate-400">Shock index
          <input type="range" min={0.3} max={2} step={0.05} value={shock} onChange={(e) => setShock(Number(e.target.value))} className="accent-cyan-400" />
          <span className="w-10 font-mono text-slate-200">{shock.toFixed(2)}</span></label>}>
        <div className="flex gap-3">
          <div className="flex flex-col justify-between py-1 text-[10px] text-slate-500"><span>tau 3.0</span><span>tau 1.0</span></div>
          <div className="grid flex-1 gap-px" style={{ gridTemplateColumns: `repeat(${data.x.length}, minmax(0, 1fr))` }}>
            {rows.map((ri) => data.x.map((x, ci) => (
              <div key={`${ri}-${ci}`} title={`P_crit ${x.toFixed(2)} | tau ${data.y[ri].toFixed(2)} | U ${data.z[ri][ci].toFixed(1)}`}
                className="aspect-square" style={{ background: color(data.z[ri][ci]) }} />
            )))}
          </div>
        </div>
        <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500"><span className="pl-12">P_crit 0</span><span>P_crit 1</span></div>
        <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
          <span>U = 0</span><div className="h-2 flex-1 rounded" style={{ background: "linear-gradient(90deg, hsl(130 75% 38%), hsl(65 75% 40%), hsl(0 75% 44%))" }} /><span>U = 100</span>
        </div>
        <p className="mt-2 text-xs text-slate-500">Drag the shock index: the same clinical risk becomes more urgent when the street is gridlocked and vitals are collapsing.</p>
      </Card>

      {top.length > 0 && (
        <Card title="Top-Urgency Patients: Rules That Fired" icon={ListChecks}>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {top.map((p) => (
              <div key={p.id} className="rounded-lg border border-slate-800 p-3 text-xs">
                <div className="mb-1 flex items-center justify-between"><b>{p.id}</b><span className="font-mono text-amber-300">U {fmt(p.urgency)}</span></div>
                <div className="mb-2 text-slate-500">P_crit {fmt(p.p_crit, 2)} | tau {fmt(p.tau, 2)} | shock {fmt(p.shock_index, 2)}</div>
                {p.fuzzy_rules.map((r) => <div key={r.rule} className="text-slate-300">R{r.rule} ({fmt(r.strength, 2)}): {r.risk} &amp; {r.traffic} &amp; {r.shock} &rarr; <Badge color="amber">{r.consequent}</Badge></div>)}
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title={`27-Rule Mamdani Rule Base`}>
        <div className="max-h-80 overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-slate-900 text-slate-400"><tr><th className="py-1.5">#</th><th>IF risk</th><th>AND traffic</th><th>AND shock</th><th>THEN urgency</th></tr></thead>
            <tbody>
              {data.rules.map((r) => (
                <tr key={r.id} className="border-t border-slate-800"><td className="py-1 text-slate-500">{r.id}</td><td>{r.risk}</td><td>{r.traffic}</td><td>{r.shock}</td>
                  <td><Badge color={r.consequent === "Critical" ? "red" : r.consequent === "High" ? "amber" : r.consequent === "Medium" ? "cyan" : "green"}>{r.consequent}</Badge></td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
