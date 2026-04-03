import { useRef, useEffect } from "react";
import { useCtrl } from "@/context";

/** 10 side-by-side 28×28 heatmaps showing each detector's last correlation output.
 *  Green = positive correlation (agreement), Red = negative (disagreement), Dark = silent. */
export function CorrelationGrid() {
  const { state } = useCtrl();
  if (!state.detectors || state.detectors.length === 0) return null;

  const hasData = state.detectors.some((d) => d.correlation && d.correlation.length > 0);
  if (!hasData) return null;

  return (
    <div
      className="rounded border p-1.5"
      style={{ background: "var(--bg-secondary)", borderColor: "var(--border)" }}
    >
      <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
        Detector Correlation Outputs (last sample)
      </h2>
      <div className="flex gap-1 justify-center">
        {state.detectors.map((d) => (
          <div key={d.digit} className="text-center">
            <CorrelationHeatmap values={d.correlation} size={56} />
            <span className="text-[8px] text-gray-500">d{d.digit}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CorrelationHeatmap({ values, size }: { values: number[]; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !values || values.length < 784) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(28, 28);
    const maxAbs = Math.max(1, ...values.map((v) => Math.abs(v)));

    for (let j = 0; j < 784; j++) {
      const v = values[j] ?? 0;
      const norm = Math.abs(v) / maxAbs;
      const idx = j * 4;

      if (v > 0) {
        // Positive correlation — green
        img.data[idx] = 0;
        img.data[idx + 1] = Math.floor(40 + norm * 215);
        img.data[idx + 2] = 0;
      } else if (v < 0) {
        // Negative correlation — red
        img.data[idx] = Math.floor(40 + norm * 215);
        img.data[idx + 1] = 0;
        img.data[idx + 2] = 0;
      } else {
        img.data[idx] = 10;
        img.data[idx + 1] = 10;
        img.data[idx + 2] = 10;
      }
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values]);

  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
      className="border border-gray-700"
    />
  );
}
