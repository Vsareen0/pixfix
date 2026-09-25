import { forwardRef, useCallback, useEffect, useImperativeHandle, useReducer, useRef, useState } from "react";

// Pure reducer (safe under React StrictMode double-invocation).
function historyReducer(state, action) {
  switch (action.type) {
    case "add":
      return { strokes: [...state.strokes, action.stroke], redo: [] };
    case "undo":
      if (!state.strokes.length) return state;
      return { strokes: state.strokes.slice(0, -1), redo: [...state.redo, state.strokes[state.strokes.length - 1]] };
    case "redo":
      if (!state.redo.length) return state;
      return { strokes: [...state.strokes, state.redo[state.redo.length - 1]], redo: state.redo.slice(0, -1) };
    case "clear":
      return state.strokes.length ? { strokes: [], redo: [] } : state;
    case "reset":
      return { strokes: [], redo: [] };
    default:
      return state;
  }
}

/**
 * Module 2: Image Input & Interaction.
 *
 * The image is shown in an <img>; a transparent <canvas> with the image's
 * *natural* pixel size sits on top of it (scaled by CSS). Brush strokes are
 * stored as vectors so undo/redo just re-renders the list. The exported mask is
 * the canvas itself as PNG: painted pixels are opaque (= remove), the rest
 * transparent (= keep) – the backend binarises the alpha channel.
 */
const MaskEditor = forwardRef(function MaskEditor({ src, disabled, onChange }, ref) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const current = useRef(null);
  const [natural, setNatural] = useState(null);
  const [{ strokes, redo }, dispatch] = useReducer(historyReducer, { strokes: [], redo: [] });
  const [tool, setTool] = useState("brush");
  const [size, setSize] = useState(36);
  const [cursor, setCursor] = useState(null);

  const drawStroke = (ctx, s) => {
    ctx.save();
    ctx.globalCompositeOperation = s.tool === "erase" ? "destination-out" : "source-over";
    ctx.strokeStyle = ctx.fillStyle = "rgb(255, 45, 95)";
    ctx.lineWidth = s.size;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const pts = s.points;
    if (pts.length === 1) {
      ctx.beginPath();
      ctx.arc(pts[0][0], pts[0][1], s.size / 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();
    }
    ctx.restore();
  };

  const redraw = useCallback((list) => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    list.forEach((s) => drawStroke(ctx, s));
  }, []);

  useEffect(() => {
    dispatch({ type: "reset" });
    setNatural(null);
  }, [src]);

  useEffect(() => {
    redraw(strokes);
    onChange?.(strokes.some((s) => s.tool === "brush"));
  }, [strokes, natural, redraw, onChange]);

  const undo = useCallback(() => dispatch({ type: "undo" }), []);
  const redoFn = useCallback(() => dispatch({ type: "redo" }), []);
  const clear = useCallback(() => dispatch({ type: "clear" }), []);

  useImperativeHandle(ref, () => ({
    getMaskBlob: () => new Promise((res) => canvasRef.current.toBlob(res, "image/png")),
    clear,
  }), [clear]);

  // Keyboard shortcuts: Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z, [ ], B, E
  useEffect(() => {
    const onKey = (e) => {
      if (disabled || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        e.shiftKey ? redoFn() : undo();
      } else if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redoFn();
      } else if (e.key === "[") setSize((s) => Math.max(4, s - 4));
      else if (e.key === "]") setSize((s) => Math.min(200, s + 4));
      else if (e.key.toLowerCase() === "b") setTool("brush");
      else if (e.key.toLowerCase() === "e") setTool("erase");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redoFn, disabled]);

  const toCanvas = (e) => {
    const c = canvasRef.current;
    const r = c.getBoundingClientRect();
    const sx = c.width / r.width;
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * (c.height / r.height), scale: sx };
  };

  const down = (e) => {
    if (disabled || e.button > 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = toCanvas(e);
    current.current = { tool, size: size * p.scale, points: [[p.x, p.y]] };
    drawStroke(canvasRef.current.getContext("2d"), current.current);
  };

  const move = (e) => {
    const r = canvasRef.current.getBoundingClientRect();
    setCursor({ x: e.clientX - r.left, y: e.clientY - r.top });
    if (!current.current) return;
    const p = toCanvas(e);
    const s = current.current;
    const last = s.points[s.points.length - 1];
    if (Math.hypot(p.x - last[0], p.y - last[1]) < 1) return;
    s.points.push([p.x, p.y]);
    // incremental draw of the newest segment
    drawStroke(canvasRef.current.getContext("2d"), { ...s, points: [last, [p.x, p.y]] });
  };

  const up = () => {
    if (!current.current) return;
    const s = current.current;
    current.current = null;
    dispatch({ type: "add", stroke: s });
  };

  return (
    <div className="mask-editor">
      <div className="toolbar">
        <div className="seg">
          <button className={tool === "brush" ? "active" : ""} onClick={() => setTool("brush")} title="Brush (B)">
            ✎ Brush
          </button>
          <button className={tool === "erase" ? "active" : ""} onClick={() => setTool("erase")} title="Eraser (E)">
            ⌫ Eraser
          </button>
        </div>
        <label className="size">
          Size
          <input type="range" min="4" max="200" value={size} onChange={(e) => setSize(+e.target.value)} />
          <span className="muted small">{size}px</span>
        </label>
        <div className="seg">
          <button onClick={undo} disabled={!strokes.length} title="Undo (Ctrl+Z)">↶ Undo</button>
          <button onClick={redoFn} disabled={!redo.length} title="Redo (Ctrl+Shift+Z)">↷ Redo</button>
          <button onClick={clear} disabled={!strokes.length}>Clear</button>
        </div>
      </div>

      <div className="stage">
        <div className="stage-inner">
          <img
            ref={imgRef}
            src={src}
            alt="Your upload"
            draggable={false}
            onLoad={(e) => setNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
          />
          {natural && (
            <canvas
              ref={canvasRef}
              width={natural.w}
              height={natural.h}
              className={`mask-canvas ${disabled ? "disabled" : ""}`}
              onPointerDown={down}
              onPointerMove={move}
              onPointerUp={up}
              onPointerCancel={up}
              onPointerLeave={() => setCursor(null)}
            />
          )}
          {cursor && !disabled && (
            <div
              className={`brush-cursor ${tool}`}
              style={{ width: size, height: size, left: cursor.x - size / 2, top: cursor.y - size / 2 }}
            />
          )}
        </div>
      </div>
      <p className="muted small hint">
        Paint over the object you want to remove. Shortcuts: <kbd>B</kbd> brush, <kbd>E</kbd> eraser,{" "}
        <kbd>[</kbd>/<kbd>]</kbd> size, <kbd>Ctrl</kbd>+<kbd>Z</kbd> undo.
      </p>
    </div>
  );
});

export default MaskEditor;
