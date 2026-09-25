import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, downloadFile, fetchImageUrl, formatDate } from "../api.js";
import AuthImage from "../components/AuthImage.jsx";
import CompareSlider from "../components/CompareSlider.jsx";

const FILTERS = ["all", "done", "processing", "pending", "failed"];

export default function History() {
  const nav = useNavigate();
  const [data, setData] = useState({ items: [], page: 1, pages: 1, total: 0 });
  const [page, setPage] = useState(1);
  const [status, setStatus] = useState("all");
  const [error, setError] = useState("");
  const [viewing, setViewing] = useState(null);

  const load = useCallback(async () => {
    try {
      const q = new URLSearchParams({ page, per_page: 12 });
      if (status !== "all") q.set("status", status);
      setData(await api(`/api/images?${q}`));
    } catch (e) {
      setError(e.message);
    }
  }, [page, status]);

  useEffect(() => { load(); }, [load]);

  // Refresh while jobs are still running.
  useEffect(() => {
    if (!data.items.some((i) => i.status === "pending" || i.status === "processing")) return;
    const t = setTimeout(load, 1500);
    return () => clearTimeout(t);
  }, [data, load]);

  const remove = async (item) => {
    if (!window.confirm("Delete this image and its result permanently?")) return;
    try {
      await api(`/api/images/${item.img_id}`, { method: "DELETE" });
      load();
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p className="muted">{data.total} processed image{data.total === 1 ? "" : "s"}</p>
        </div>
        <div className="seg">
          {FILTERS.map((f) => (
            <button key={f} className={status === f ? "active" : ""} onClick={() => { setStatus(f); setPage(1); }}>
              {f}
            </button>
          ))}
        </div>
      </div>
      {error && <div className="alert error">{error}</div>}

      {data.items.length === 0 ? (
        <div className="card center muted empty">
          Nothing here yet. <a onClick={() => nav("/")}>Fix your first image →</a>
        </div>
      ) : (
        <div className="grid">
          {data.items.map((item) => (
            <div className="card history-item" key={item.img_id}>
              <button className="thumb" onClick={() => item.result_url && setViewing(item)}>
                <AuthImage path={item.result_url || item.original_url} alt={item.original_filename} />
                <span className={`badge ${item.status}`}>{item.status}</span>
              </button>
              <div className="hi-body">
                <div className="hi-title" title={item.original_filename}>{item.original_filename || `Image #${item.img_id}`}</div>
                <div className="muted small">
                  {formatDate(item.upload_time)}
                  {item.processing_time != null && ` · ${item.processing_time.toFixed(2)}s`}
                  {item.model_used && ` · ${item.model_used}`}
                </div>
                {item.status === "failed" && <div className="field-error small">{item.error_message}</div>}
                <div className="row sm-gap">
                  {item.result_url && (
                    <button className="btn sm" onClick={() => downloadFile(item.result_url)}>Download</button>
                  )}
                  <button className="btn ghost sm" onClick={() => nav(`/?source=${item.img_id}`)}>Edit again</button>
                  <button className="btn ghost sm danger" onClick={() => remove(item)}>Delete</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {data.pages > 1 && (
        <div className="pager">
          <button className="btn ghost sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>← Prev</button>
          <span className="muted small">Page {data.page} of {data.pages}</span>
          <button className="btn ghost sm" disabled={page >= data.pages} onClick={() => setPage(page + 1)}>Next →</button>
        </div>
      )}

      {viewing && <Viewer item={viewing} onClose={() => setViewing(null)} />}
    </div>
  );
}

function Viewer({ item, onClose }) {
  const [urls, setUrls] = useState(null);
  useEffect(() => {
    let made = [];
    Promise.all([fetchImageUrl(item.original_url), fetchImageUrl(item.result_url)]).then((u) => {
      made = u;
      setUrls(u);
    });
    const esc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => { made.forEach(URL.revokeObjectURL); window.removeEventListener("keydown", esc); };
  }, [item, onClose]);

  return (
    <div className="modal" onClick={onClose}>
      <div className="modal-body card" onClick={(e) => e.stopPropagation()}>
        <div className="page-head">
          <strong>{item.original_filename}</strong>
          <button className="btn ghost sm" onClick={onClose}>Close ✕</button>
        </div>
        {urls ? <CompareSlider before={urls[0]} after={urls[1]} /> : <div className="img-placeholder shimmer tall" />}
      </div>
    </div>
  );
}
