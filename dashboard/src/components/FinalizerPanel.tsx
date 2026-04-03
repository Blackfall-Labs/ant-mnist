import { useCtrl } from "@/context";

const DIGIT_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
];

/** Finalizer ANT status — per-class weight magnitude bars + mastery stats. */
export function FinalizerPanel() {
  const { state } = useCtrl();
  const f = state.finalizer;
  if (!f || !f.class_magnitudes) return null;

  const maxMag = Math.max(1, ...f.class_magnitudes);
  const maxNz = Math.max(1, ...f.class_nonzero);

  return (
    <div
      className="rounded border p-1.5"
      style={{ background: "var(--bg-secondary)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-4 mb-1">
        <h2 className="text-xs text-gray-500 border-b border-gray-800 pb-0.5">
          Finalizer ANT (10×7840)
        </h2>
        <span className="text-[10px] text-gray-600">
          Weights: {f.nonzero_weights.toLocaleString()}/{f.total_weights.toLocaleString()} nonzero
        </span>
        <span className="text-[10px] text-gray-600">
          Steps: {f.steps.toLocaleString()}
        </span>
        <span className="text-[10px] text-gray-600">
          Transitions: {f.transitions.toLocaleString()}
        </span>
      </div>

      <div className="flex gap-2">
        {/* Per-class magnitude bars */}
        <div className="flex-1">
          <span className="text-[9px] text-gray-600">Per-class weight magnitude</span>
          <div className="flex gap-0.5 items-end h-12 mt-0.5">
            {f.class_magnitudes.map((mag: number, i: number) => (
              <div key={i} className="flex-1 flex flex-col items-center justify-end h-full">
                <div
                  style={{
                    width: "100%",
                    height: `${(mag / maxMag) * 100}%`,
                    background: DIGIT_COLORS[i],
                    minHeight: mag > 0 ? 2 : 0,
                  }}
                  title={`d${i}: ${mag.toLocaleString()}`}
                />
                <span className="text-[8px] text-gray-500 mt-0.5">{i}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Per-class nonzero weight counts */}
        <div className="flex-1">
          <span className="text-[9px] text-gray-600">Nonzero weights per class</span>
          <div className="flex gap-0.5 items-end h-12 mt-0.5">
            {f.class_nonzero.map((nz: number, i: number) => (
              <div key={i} className="flex-1 flex flex-col items-center justify-end h-full">
                <div
                  style={{
                    width: "100%",
                    height: `${(nz / maxNz) * 100}%`,
                    background: DIGIT_COLORS[i],
                    opacity: 0.7,
                    minHeight: nz > 0 ? 2 : 0,
                  }}
                  title={`d${i}: ${nz.toLocaleString()}/7840`}
                />
                <span className="text-[8px] text-gray-500 mt-0.5">{i}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
