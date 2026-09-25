import { useCallback, useEffect, useState } from "react";
import { api, formatBytes, formatDate } from "../api.js";
import { useAuth } from "../auth.jsx";

const TABS = [
  ["overview", "Overview"],
  ["users", "Users"],
  ["logs", "Processing log"],
  ["errors", "Error log"],
];

export default function Admin() {
  const [tab, setTab] = useState("overview");
  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Administration</h1>
          <p className="muted">Manage users and review system reports.</p>
        </div>
        <div className="seg">
          {TABS.map(([k, label]) => (
            <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>
          ))}
        </div>
      </div>
      {tab === "overview" && <Overview />}
      {tab === "users" && <Users />}
      {tab === "logs" && <Logs />}
      {tab === "errors" && <Logs status="failed" />}
    </div>
  );
}

/* ---------- User Activity + Processing Time Analysis reports ---------- */
function Overview() {
  const [s, setS] = useState(null);
  const [rep, setRep] = useState(null);
  const [days, setDays] = useState(14);
  const [error, setError] = useState("");

  useEffect(() => {
    api("/api/admin/stats").then(setS).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    api(`/api/admin/reports/processing-time?days=${days}`).then(setRep).catch((e) => setError(e.message));
  }, [days]);

  if (error) return <div className="alert error">{error}</div>;
  if (!s) return <div className="muted">Loading…</div>;
  const maxAvg = Math.max(0.001, ...(rep?.per_day || []).map((d) => d.avg));

  return (
    <>
      <h3>User activity</h3>
      <div className="stats">
        <Stat label="Total users" value={s.total_users} />
        <Stat label="Active today" value={s.active_today} />
        <Stat label="Images processed" value={s.total_images} sub={`${s.images_today} today`} />
        <Stat label="Storage used" value={formatBytes(s.total_storage_bytes)} />
        <Stat label="Avg. processing time" value={s.avg_processing_time ? `${s.avg_processing_time}s` : "—"} />
        <Stat label="Failed jobs" value={s.status_counts.failed} tone={s.status_counts.failed ? "bad" : ""} />
      </div>

      <div className="page-head">
        <h3>Processing time analysis</h3>
        <select value={days} onChange={(e) => setDays(+e.target.value)}>
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
      </div>
      {rep && (
        <div className="two-col">
          <div className="card">
            <h4>By model</h4>
            <table className="table">
              <thead><tr><th>Model</th><th>Jobs</th><th>Avg</th><th>Min</th><th>Max</th></tr></thead>
              <tbody>
                {rep.per_model.length === 0 && <tr><td colSpan={5} className="muted">No completed jobs</td></tr>}
                {rep.per_model.map((m) => (
                  <tr key={m.model}><td>{m.model}</td><td>{m.count}</td><td>{m.avg}s</td><td>{m.min}s</td><td>{m.max}s</td></tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">{rep.failed} failed job(s) in this period.</p>
          </div>
          <div className="card">
            <h4>Average time per day</h4>
            {rep.per_day.length === 0 ? <p className="muted">No data</p> : (
              <div className="bars">
                {rep.per_day.map((d) => (
                  <div className="bar-row" key={d.date} title={`${d.count} job(s)`}>
                    <span className="small muted">{d.date.slice(5)}</span>
                    <div className="bar"><div style={{ width: `${(d.avg / maxAvg) * 100}%` }} /></div>
                    <span className="small">{d.avg}s</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function Stat({ label, value, sub, tone }) {
  return (
    <div className={`card stat ${tone || ""}`}>
      <div className="muted small">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}

/* ---------- Manage users ---------- */
function Users() {
  const { user: me } = useAuth();
  const [items, setItems] = useState([]);
  const [q, setQ] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(() => {
    api(`/api/admin/users?q=${encodeURIComponent(q)}`).then((d) => setItems(d.items)).catch((e) => setError(e.message));
  }, [q]);
  useEffect(() => { const t = setTimeout(load, 250); return () => clearTimeout(t); }, [load]);

  const patch = async (u, body) => {
    setError("");
    try { await api(`/api/admin/users/${u.user_id}`, { method: "PATCH", json: body }); load(); }
    catch (e) { setError(e.message); }
  };
  const del = async (u) => {
    if (!window.confirm(`Delete ${u.email} and all their images?`)) return;
    try { await api(`/api/admin/users/${u.user_id}`, { method: "DELETE" }); load(); }
    catch (e) { setError(e.message); }
  };

  return (
    <div className="card">
      <input className="search" placeholder="Search by name or email…" value={q} onChange={(e) => setQ(e.target.value)} />
      {error && <div className="alert error">{error}</div>}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr><th>ID</th><th>User</th><th>Joined</th><th>Last login</th><th>Images</th><th>Storage</th><th>Role</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {items.map((u) => (
              <tr key={u.user_id}>
                <td>{u.user_id}</td>
                <td><div>{u.username}</div><div className="muted small">{u.email}</div></td>
                <td className="small">{formatDate(u.created_at)}</td>
                <td className="small">{formatDate(u.last_login_at)}</td>
                <td>{u.image_count}</td>
                <td>{formatBytes(u.storage_bytes)}</td>
                <td>{u.is_admin ? <span className="badge done">admin</span> : "user"}</td>
                <td>{u.is_active ? <span className="badge done">active</span> : <span className="badge failed">disabled</span>}</td>
                <td className="row sm-gap nowrap">
                  {u.user_id !== me.user_id && (
                    <>
                      <button className="btn ghost sm" onClick={() => patch(u, { is_active: !u.is_active })}>
                        {u.is_active ? "Disable" : "Enable"}
                      </button>
                      <button className="btn ghost sm" onClick={() => patch(u, { is_admin: !u.is_admin })}>
                        {u.is_admin ? "Revoke admin" : "Make admin"}
                      </button>
                      <button className="btn ghost sm danger" onClick={() => del(u)}>Delete</button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ---------- Processing Log Report / Failure-Error Logs ---------- */
function Logs({ status }) {
  const [data, setData] = useState({ items: [], page: 1, pages: 1, total: 0 });
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");

  useEffect(() => { setPage(1); }, [status]);
  useEffect(() => {
    const qs = new URLSearchParams({ page, per_page: 25 });
    if (status) qs.set("status", status);
    api(`/api/admin/logs?${qs}`).then(setData).catch((e) => setError(e.message));
  }, [page, status]);

  const exportCsv = () => {
    const head = ["Image ID", "Date", "Time Taken (s)", "Status", "Model", "User", "Email", "Error"];
    const rows = data.items.map((r) => [r.img_id, r.upload_time, r.processing_time ?? "", r.status,
      r.model_used ?? "", r.username, r.email, (r.error_message || "").replace(/\n/g, " ")]);
    const csv = [head, ...rows].map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = status === "failed" ? "pixfix_error_log.csv" : "pixfix_processing_log.csv";
    a.click();
  };

  return (
    <div className="card">
      <div className="page-head">
        <span className="muted">{data.total} record(s)</span>
        <button className="btn ghost sm" onClick={exportCsv} disabled={!data.items.length}>Export CSV</button>
      </div>
      {error && <div className="alert error">{error}</div>}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Image ID</th><th>Date</th><th>Time taken</th><th>Status</th><th>Model</th><th>User</th>
              {status === "failed" && <th>Error</th>}
            </tr>
          </thead>
          <tbody>
            {data.items.length === 0 && <tr><td colSpan={7} className="muted">No records</td></tr>}
            {data.items.map((r) => (
              <tr key={r.img_id}>
                <td>{r.img_id}</td>
                <td className="small">{formatDate(r.upload_time)}</td>
                <td>{r.processing_time != null ? `${r.processing_time.toFixed(2)}s` : "—"}</td>
                <td><span className={`badge ${r.status}`}>{r.status}</span></td>
                <td>{r.model_used || "—"}</td>
                <td>{r.username}</td>
                {status === "failed" && <td className="small mono">{r.error_message}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.pages > 1 && (
        <div className="pager">
          <button className="btn ghost sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>← Prev</button>
          <span className="muted small">Page {data.page} of {data.pages}</span>
          <button className="btn ghost sm" disabled={page >= data.pages} onClick={() => setPage(page + 1)}>Next →</button>
        </div>
      )}
    </div>
  );
}
