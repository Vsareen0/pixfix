import { useState } from "react";

/** Before/after slider: the "after" image is revealed from the left. */
export default function CompareSlider({ before, after }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="compare">
      <img src={before} alt="Before" draggable={false} />
      <img src={after} alt="After" className="after" draggable={false}
           style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
      <div className="divider" style={{ left: `${pos}%` }}><span>⇆</span></div>
      <span className="tag left">After</span>
      <span className="tag right">Before</span>
      <input type="range" min="0" max="100" value={pos} aria-label="Compare"
             onChange={(e) => setPos(+e.target.value)} />
    </div>
  );
}
