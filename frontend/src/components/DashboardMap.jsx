import React, { useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from "react-leaflet";
import { Download, Hospital, ListOrdered } from "lucide-react";
import { Badge, Card, Empty, Stat, fmt, urgencyColor } from "./ui.jsx";

function exportCsv(sim) {
  const head = ["Rank", "Patient", "Urgency", "P_crit", "Specialty", "Ambulance", "Hospital", "Pickup_min", "Transit_min", "Total_min", "Golden_hour_ok"];
  const rows = sim.patients.map((p) => [p.rank, p.id, fmt(p.urgency), fmt(p.p_crit, 3), p.specialty, p.amb_id, p.hospital,
    fmt(p.pickup_min), fmt(p.transit_min), fmt(p.total_min), p.golden_hour_ok]);
  const csv = [head, ...rows].map((r) => r.map((c) => `"${c}"`).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "aegis_dispatch_manifest.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function DashboardMap({ sim, status }) {
  const [sel, setSel] = useState(null);
  if (!sim) return <Empty text="Run the pipeline to see live dispatch." />;
  const rs = status?.rough_set, nn = status?.neural?.models?.reduct;
  const k = sim.kpis;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Stat label="Rough-set reduction" value={`${rs?.n_full} -> ${rs?.n_reduct}`} sub={`-${fmt(rs?.reduction_pct)}% attributes`} />
        <Stat label="ANN ROC-AUC / F1" value={`${fmt(nn?.roc_auc, 3)} / ${fmt(nn?.f1 * 100)}%`} sub="held-out test patients" tone="violet" />
        <Stat label="Critical golden-hour" value={`${fmt(k.critical_served_pct, 0)}% < 45 min`} sub={`${k.critical_count} critical of ${k.n_calls} calls`} tone="green" />
        <Stat label="GA cost vs FCFS" value={`-${fmt(k.cost_improvement_vs_greedy_pct)}%`} sub={`GA ran in ${fmt(k.ga_runtime_s, 2)} s`} tone="amber" />
      </div>

      <Card title={`Spatial Dispatch Grid - ${sim.scenario_label} (traffic x${fmt(sim.traffic_base, 2)})`}
        right={<div className="flex gap-2 text-[11px] text-slate-400"><span className="text-red-400">&bull; critical</span><span className="text-amber-400">&bull; urgent</span><span className="text-emerald-400">&bull; stable</span><span className="text-sky-400">&bull; ambulance</span><span className="text-violet-400">&bull; hospital</span></div>}>
        <div className="h-[520px] overflow-hidden rounded-lg">
          <MapContainer center={[28.6, 77.17]} zoom={10} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
            <TileLayer attribution='&copy; OpenStreetMap &copy; CARTO' url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
            {sim.patients.map((p) => {
              const c = urgencyColor(p.urgency);
              const on = sel === p.id;
              return (
                <React.Fragment key={p.id}>
                  <Polyline positions={[[p.amb_lat, p.amb_lon], [p.lat, p.lon]]} pathOptions={{ color: "#38bdf8", weight: on ? 3 : 1.2, dashArray: "5 6", opacity: sel && !on ? 0.2 : 0.75 }} />
                  <Polyline positions={[[p.lat, p.lon], [p.hosp_lat, p.hosp_lon]]} pathOptions={{ color: c, weight: on ? 4 : 1.8, opacity: sel && !on ? 0.2 : 0.8 }} />
                  <CircleMarker center={[p.lat, p.lon]} radius={on ? 12 : 8} pathOptions={{ color: p.id === sim.injected_id ? "#fff" : c, fillColor: c, fillOpacity: 0.9, weight: p.id === sim.injected_id ? 3 : 1 }}
                    eventHandlers={{ click: () => setSel(p.id) }}>
                    <Tooltip>
                      <div className="text-xs">
                        <b>{p.id}</b> - {p.specialty}<br />SpO2 {fmt(p.spo2, 0)} | HR {fmt(p.hr, 0)} | SBP {fmt(p.sbp, 0)}<br />
                        P_crit {fmt(p.p_crit, 2)} &rarr; U {fmt(p.urgency)}<br />{p.amb_id} &rarr; {p.hospital} ({fmt(p.total_min, 0)} min)
                      </div>
                    </Tooltip>
                  </CircleMarker>
                </React.Fragment>
              );
            })}
            {sim.ambulances.map((a) => (
              <CircleMarker key={a.id} center={[a.lat, a.lon]} radius={a.assigned_patient ? 5 : 3} pathOptions={{ color: "#38bdf8", fillColor: "#38bdf8", fillOpacity: a.assigned_patient ? 0.9 : 0.25, weight: 1 }}>
                <Tooltip>{a.id} {a.assigned_patient ? `-> ${a.assigned_patient}` : "(standby)"}</Tooltip>
              </CircleMarker>
            ))}
            {sim.hospitals.map((h) => (
              <CircleMarker key={h.name} center={[h.lat, h.lon]} radius={11} pathOptions={{ color: "#a78bfa", fillColor: "#7c3aed", fillOpacity: 0.75, weight: 2 }}>
                <Tooltip>
                  <div className="text-xs"><b>{h.name}</b>{Object.keys(h.beds).map((s) => <div key={s}>{s}: {h.assigned[s]}/{h.beds[s]} beds</div>)}</div>
                </Tooltip>
              </CircleMarker>
            ))}
          </MapContainer>
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <Card title="Prioritised Dispatch Manifest" icon={ListOrdered}
          right={<button onClick={() => exportCsv(sim)} className="flex items-center gap-1.5 rounded-lg border border-slate-700 px-2.5 py-1 text-xs hover:bg-slate-800"><Download size={13} /> Export CSV</button>}>
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-slate-900 text-slate-400">
                <tr>{["#", "Patient", "U", "P_crit", "Model KTAS", "Expert / Nurse", "Specialty", "Ambulance", "Hospital", "ETA"].map((h) => <th key={h} className="px-2 py-2 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody>
                {sim.patients.map((p) => (
                  <tr key={p.id} onClick={() => setSel(sel === p.id ? null : p.id)} className={`cursor-pointer border-t border-slate-800 hover:bg-slate-800/60 ${sel === p.id ? "bg-slate-800" : ""}`}>
                    <td className="px-2 py-1.5 text-slate-500">{p.rank}</td>
                    <td className="px-2 py-1.5 font-semibold">{p.id}{p.id === sim.injected_id && <Badge color="violet">live</Badge>}</td>
                    <td className="px-2 py-1.5 font-mono font-bold" style={{ color: urgencyColor(p.urgency) }}>{fmt(p.urgency)}</td>
                    <td className="px-2 py-1.5 font-mono">{fmt(p.p_crit, 2)}</td>
                    <td className="px-2 py-1.5">{p.ktas_model}</td>
                    <td className="px-2 py-1.5 text-slate-400">{p.ktas_expert ?? "-"} / {p.ktas_nurse ?? "-"}</td>
                    <td className="px-2 py-1.5"><Badge color={p.specialty === "Cardiac" ? "red" : p.specialty === "Trauma" ? "amber" : p.specialty === "Neuro" ? "violet" : "slate"}>{p.specialty}</Badge></td>
                    <td className="px-2 py-1.5">{p.amb_id}</td>
                    <td className="px-2 py-1.5">{p.hospital}</td>
                    <td className="px-2 py-1.5"><span className={p.critical && !p.golden_hour_ok ? "text-red-400" : ""}>{fmt(p.total_min, 0)} m</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Hospital Bed Utilisation" icon={Hospital}>
          <div className="space-y-3">
            {sim.hospitals.map((h) => (
              <div key={h.name}>
                <div className="mb-1 text-xs font-medium text-slate-300">{h.name}</div>
                <div className="grid grid-cols-4 gap-1.5">
                  {Object.keys(h.beds).map((s) => {
                    const over = h.assigned[s] > h.beds[s];
                    const ratio = h.beds[s] ? Math.min(h.assigned[s] / h.beds[s], 1) : 0;
                    return (
                      <div key={s} className="rounded-md bg-slate-800 p-1.5 text-center">
                        <div className="text-[10px] text-slate-400">{s}</div>
                        <div className={`text-xs font-bold ${over ? "text-red-400" : "text-slate-100"}`}>{h.assigned[s]}/{h.beds[s]}</div>
                        <div className="mt-1 h-1 rounded bg-slate-700"><div className="h-1 rounded bg-violet-400" style={{ width: `${ratio * 100}%` }} /></div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
