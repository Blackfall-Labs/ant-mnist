import { useRef, useEffect } from "react";
import type { Snap } from "@/types";

const DIGIT_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
];

interface Props {
  history: Snap[];
  mode: "overall" | "per-detector";
  height?: number;
}

/** Line chart for accuracy history — overall or per-detector. */
export function AccuracyChart({ history, mode, height = 90 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || history.length === 0) return;
    cv.width = cv.clientWidth;
    cv.height = height;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const w = cv.width;
    const h = cv.height;

    ctx.fillStyle = "#0e0e0e";
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 0.5;
    for (const p of [0.25, 0.5, 0.75, 1]) {
      const y = h - p * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillStyle = "#333";
      ctx.font = "8px monospace";
      ctx.fillText(`${p * 100}%`, 1, y - 1);
    }

    const xOf = (i: number) =>
      history.length === 1 ? w / 2 : (i / (history.length - 1)) * w;

    if (mode === "overall") {
      ctx.strokeStyle = "#ff6b35";
      ctx.lineWidth = 2;
      ctx.beginPath();
      history.forEach((snap, i) => {
        const x = xOf(i);
        const y = h - snap.test_acc * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    } else {
      for (let d = 0; d < 10; d++) {
        ctx.strokeStyle = DIGIT_COLORS[d];
        ctx.lineWidth = 1;
        ctx.beginPath();
        history.forEach((snap, i) => {
          if (!snap.det_accs || snap.det_accs.length <= d) return;
          const x = xOf(i);
          const y = h - (snap.det_accs[d] ?? 0) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
      // Legend
      for (let d = 0; d < 10; d++) {
        ctx.fillStyle = DIGIT_COLORS[d];
        ctx.fillText(d.toString(), w - 105 + d * 10, 10);
      }
    }
  }, [history, mode, height]);

  return (
    <canvas
      ref={ref}
      height={height}
      className="w-full border border-gray-800"
      style={{ height, background: "#0e0e0e" }}
    />
  );
}
