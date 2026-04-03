import { useRef, useEffect } from "react";

const DIGIT_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
];

interface Props {
  data: number[];
  digit: number;
  width?: number;
  height?: number;
}

/** Tiny line chart for a single detector's confidence history. */
export function Sparkline({ data, digit, width = 200, height = 16 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || data.length < 2) return;
    cv.width = width;
    cv.height = height;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, width, height);

    const mx = Math.max(1, ...data.map((v) => Math.abs(v)));
    const mid = height / 2;

    // Zero line
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    // Data line
    ctx.strokeStyle = DIGIT_COLORS[digit] ?? "#888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * width;
      const y = mid - (data[i] / mx) * (mid - 1);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [data, digit, width, height]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      className="w-full"
      style={{ height }}
    />
  );
}
