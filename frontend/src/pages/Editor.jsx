import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, downloadFile, fetchImageUrl } from "../api.js";
import CompareSlider from "../components/CompareSlider.jsx";
import MaskEditor from "../components/MaskEditor.jsx";

const MAX_MB = 20;
const MODEL_LABELS = {
  auto: "Auto (best available)",
  lama: "LaMa – pretrained deep model",
  gan: "PixFix GAN – custom trained",
  patchmatch: "PatchMatch – classical, copies texture",
  opencv: "OpenCV Telea – classical, fastest",
};

/**
 * Activity flow (Fig. 8): Upload → validate → draw mask → Process → AI inference
 * → display result → satisfied? download : undo/redraw mask.
 */
export default function Editor() {
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const editorRef = useRef(null);
  const fileInput = useRef(null);

  const [file, setFile] = useState(null);         // new upload
  const [sourceId, setSourceId] = useState(null); // or: re-edit a previous job
  const [baseUrl, setBaseUrl] = useState(null);   // what the user paints on
  const [hasMask, setHasMask] = useState(false);
  const [models, setModels] = useState(["auto"]);
  const [model, setModel] = useState("auto");
  const [job, setJob] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [phase, setPhase] = useState("empty");    // empty | edit | processing | result
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    api("/api/health")
      .then((h) => setModels(["auto", ...h.ai_backends]))
      .catch(() => {});
  }, []);

  // Open a previous job from History: /?source=<id>
  useEffect(() => {
    const src = params.get("source");
    if (!src) return;
    (async () => {
      try {
        const { image } = await api(`/api/images/${src}`);
        const url = await fetchImageUrl(image.result_url || image.original_url);
        resetResult();
        setFile(null);
        setSourceId(image.img_id);
        setBaseUrl(url);
        setPhase("edit");
      } catch (e) {
        setError(e.message);
      } finally {
        setParams({}, { replace: true });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);

  const resetResult = () => {
    setJob(null);
    setResultUrl(null);
  };

  const acceptFile = (f) => {
    setError("");
    if (!f) return;
    if (!["image/jpeg", "image/png"].includes(f.type) || !/\.(jpe?g|png)$/i.test(f.name)) {
      setError("Please choose a JPG or PNG image.");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File is larger than ${MAX_MB} MB.`);
      return;
    }
    resetResult();
    setFile(f);
    setSourceId(null);
    setBaseUrl(URL.createObjectURL(f));
    setPhase("edit");
  };

  const poll = async (id) => {
    for (;;) {
      const { image } = await api(`/api/images/${id}`);
      setJob(image);
      if (image.status === "done" || image.status === "failed") return image;
      await new Promise((r) => setTimeout(r, 800));
    }
  };

  const process = async () => {
    setError("");
    setPhase("processing");
    try {
      const mask = await editorRef.current.getMaskBlob();
      const form = new FormData();
      if (sourceId) form.append("source_id", sourceId);
      else form.append("image", file, file.name);
      form.append("mask", mask, "mask.png");
      form.append("model", model);

      const { image } = await api("/api/images/process", { method: "POST", form });
      setJob(image);
      const done = image.status === "done" || image.status === "failed" ? image : await poll(image.img_id);
      if (done.status === "failed") throw new Error(done.error_message || "Processing failed");
      setResultUrl(await fetchImageUrl(done.result_url));
      setPhase("result");
    } catch (e) {
      setError(e.message);
      setPhase("edit");
    }
  };

  const continueOnResult = () => {
    setSourceId(job.img_id);
    setFile(null);
    setBaseUrl(resultUrl);
    resetResult();
    setPhase("edit");
  };

  const startOver = () => {
    resetResult();
    setFile(null);
    setSourceId(null);
    setBaseUrl(null);
    setPhase("empty");
    setError("");
  };

  return (
    <div className="editor-page">
      <div className="page-head">
        <div>
          <h1>Remove objects</h1>
          <p className="muted">Upload a photo, paint over what you don't want, and let the AI fill it in.</p>
        </div>
        {phase !== "empty" && (
          <button className="btn ghost" onClick={startOver}>New image</button>
        )}
      </div>

      {error && <div className="alert error">{error}</div>}

      {phase === "empty" && (
        <div
          className={`dropzone ${dragOver ? "over" : ""}`}
          onClick={() => fileInput.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); acceptFile(e.dataTransfer.files[0]); }}
        >
          <div className="dz-icon">⬆</div>
          <strong>Drop an image here or click to browse</strong>
          <span className="muted small">JPG or PNG, up to {MAX_MB} MB</span>
          <input ref={fileInput} type="file" accept="image/jpeg,image/png" hidden
                 onChange={(e) => acceptFile(e.target.files[0])} />
        </div>
      )}

      {/* Kept mounted while viewing the result so "Redraw mask" keeps the strokes. */}
      {phase !== "empty" && baseUrl && (
        <div className="card editor-card" hidden={phase === "result"}>
          <MaskEditor ref={editorRef} src={baseUrl} disabled={phase === "processing"} onChange={setHasMask} />
          <div className="action-bar">
            <label className="model-select">
              Model
              <select value={model} onChange={(e) => setModel(e.target.value)} disabled={phase === "processing"}>
                {models.map((m) => <option key={m} value={m}>{MODEL_LABELS[m] || m}</option>)}
              </select>
            </label>
            <button className="btn primary lg" onClick={process} disabled={!hasMask || phase === "processing"}>
              {phase === "processing"
                ? job?.status === "processing" ? "AI is filling the area…" : "Uploading…"
                : "✨ Remove object"}
            </button>
          </div>
          {phase === "processing" && <div className="progress"><div /></div>}
        </div>
      )}

      {phase === "result" && resultUrl && (
        <div className="card editor-card">
          <CompareSlider before={baseUrl} after={resultUrl} />
          <div className="result-meta muted small">
            Model: <b>{job.model_used}</b> · {job.width}×{job.height}px · processed in{" "}
            <b>{job.processing_time?.toFixed(2)}s</b>
          </div>
          <div className="action-bar">
            <span className="muted">Happy with it?</span>
            <div className="row">
              <button className="btn ghost" onClick={() => { resetResult(); setPhase("edit"); }}>
                ↶ Redraw mask
              </button>
              <button className="btn ghost" onClick={continueOnResult}>Keep editing this result</button>
              <button className="btn primary" onClick={() => downloadFile(job.result_url)}>⬇ Download</button>
            </div>
          </div>
          <p className="muted small">Saved to your <a onClick={() => nav("/history")}>history</a>.</p>
        </div>
      )}
    </div>
  );
}
