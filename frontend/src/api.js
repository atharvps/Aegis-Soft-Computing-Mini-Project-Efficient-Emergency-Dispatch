const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function req(path, opts = {}) {
  const res = await fetch(BASE + path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* ignore */ }
    throw new Error(`${res.status}: ${msg}`);
  }
  return res.json();
}

export const api = {
  health: () => req("/api/health"),
  status: () => req("/api/status"),
  simulate: (body) => req("/api/simulate", { method: "POST", body: JSON.stringify(body) }),
  inject: (body) => req("/api/inject-patient", { method: "POST", body: JSON.stringify(body) }),
  surface: (shock, resolution = 21) => req(`/api/fuzzy-surface?shock=${shock}&resolution=${resolution}`),
};
