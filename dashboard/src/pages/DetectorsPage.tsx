import { useCtrl } from "@/context";
import { TemplateHeatmap } from "@/components/TemplateHeatmap";
import { Sparkline } from "@/components/Sparkline";

export function DetectorsPage() {
  const { state, sendCmd } = useCtrl();

  return (
    <div className="flex flex-col gap-2">
      <h1 className="text-sm" style={{ color: "var(--accent-orange)" }}>
        Detector Details
      </h1>

      {state.detectors.map((det) => {
        const accPct = (det.eval_accuracy * 100).toFixed(1);
        return (
          <div
            key={det.digit}
            className="rounded border p-2 flex gap-3 items-start"
            style={{
              background: "var(--bg-secondary)",
              borderColor: "var(--border)",
            }}
          >
            {/* Template */}
            <div className="flex flex-col items-center gap-1">
              <span
                className="text-sm font-bold"
                style={{ color: "var(--accent-orange)" }}
              >
                {det.digit}
              </span>
              <TemplateHeatmap template={det.template} size={84} />
              <span className="text-[10px] text-gray-500">{det.phase}</span>
            </div>

            {/* Stats + sparkline */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-3 mb-1 text-xs">
                <span className="text-gray-500">
                  Accuracy:{" "}
                  <span
                    style={{
                      color:
                        det.eval_accuracy > 0.8
                          ? "var(--accent-green)"
                          : det.eval_accuracy > 0.5
                            ? "#ff8833"
                            : "#888",
                    }}
                  >
                    {accPct}%
                  </span>
                </span>
                <span className="text-gray-500">
                  History: {det.confidence_history.length} points
                </span>
              </div>

              <Sparkline
                data={det.confidence_history}
                digit={det.digit}
                height={32}
              />

              {/* Weight bars */}
              {det.weight_currents && det.weight_currents.length > 0 && (
                <div className="mt-1">
                  <span className="text-[9px] text-gray-600">Weights</span>
                  <WeightBar values={det.weight_currents} maxRef={3000} />
                </div>
              )}
              {det.weight_pressures && det.weight_pressures.length > 0 && (
                <div className="mt-0.5">
                  <span className="text-[9px] text-gray-600">Pressure</span>
                  <WeightBar values={det.weight_pressures} maxRef={200} />
                </div>
              )}

              {/* Controls */}
              <div className="flex gap-1 mt-2">
                <Btn label="Imprint 50" onClick={() => sendCmd("imprint", det.digit, 50)} />
                <Btn label="Imprint 200" onClick={() => sendCmd("imprint", det.digit, 200)} />
                <Btn label="Train 50" onClick={() => sendCmd("train_cycle", det.digit, 50)} />
                <Btn label="Train 200" onClick={() => sendCmd("train_cycle", det.digit, 200)} />
                <Btn label="Freeze" onClick={() => sendCmd("freeze", det.digit)} />
                <Btn label="Thaw" onClick={() => sendCmd("unfreeze", det.digit)} />
                <Btn label="Flush" onClick={() => sendCmd("flush", det.digit)} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function WeightBar({ values, maxRef }: { values: number[]; maxRef: number }) {
  return (
    <div className="flex h-2 bg-[#0a0a0a] rounded overflow-hidden">
      {values.map((v, i) => {
        const opacity = 0.3 + Math.min(1, Math.abs(v) / maxRef) * 0.7;
        const color = v > 0 ? "#4caf50" : v < 0 ? "#f44336" : "#222";
        return (
          <div
            key={i}
            className="flex-1 min-w-0"
            style={{ background: color, opacity }}
            title={`w${i}=${v}`}
          />
        );
      })}
    </div>
  );
}

function Btn({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      className="text-[10px] px-1.5 py-0.5 border border-gray-700 rounded bg-[#282828] text-gray-300 hover:bg-gray-700 cursor-pointer"
      onClick={onClick}
    >
      {label}
    </button>
  );
}
