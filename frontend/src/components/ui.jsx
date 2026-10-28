import React from "react";

export const urgencyColor = (u) => (u >= 75 ? "#ef4444" : u >= 50 ? "#f59e0b" : "#22c55e");
export const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? "-" : `${(v * 100).toFixed(d)}%`);

export function Card({ title, icon: Icon, right, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-slate-900/70 p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-300">
            {Icon && <Icon size={16} className="text-cyan-400" />}{title}
          </h3>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, tone = "cyan" }) {
  const tones = { cyan: "text-cyan-300", red: "text-red-400", green: "text-emerald-400", amber: "text-amber-300", violet: "text-violet-300" };
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${tones[tone]}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function Badge({ children, color = "slate" }) {
  const c = { slate: "bg-slate-800 text-slate-300", red: "bg-red-500/15 text-red-300", amber: "bg-amber-500/15 text-amber-300",
    green: "bg-emerald-500/15 text-emerald-300", cyan: "bg-cyan-500/15 text-cyan-300", violet: "bg-violet-500/15 text-violet-300" };
  return <span className={`rounded-md px-2 py-0.5 text-xs font-medium ${c[color]}`}>{children}</span>;
}

export function Empty({ text = "Waiting for backend..." }) {
  return <div className="rounded-xl border border-dashed border-slate-700 p-10 text-center text-slate-500">{text}</div>;
}

export const chartTheme = { grid: "#1e293b", axis: "#64748b", tooltip: { background: "#0f172a", border: "1px solid #334155", borderRadius: 8, color: "#e2e8f0" } };
