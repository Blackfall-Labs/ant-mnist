import { useCtrl } from "@/context";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";
import { AccuracyChart } from "@/components/AccuracyChart";
import { SampleGrid } from "@/components/SampleGrid";
import { DetectorCard } from "@/components/DetectorCard";

export function OverviewPage() {
  const { state, sendCmd } = useCtrl();

  return (
    <div className="flex flex-col gap-2 h-full">
      {/* Global controls */}
      <div
        className="flex gap-1.5 items-center flex-wrap p-1.5 rounded border"
        style={{
          background: "var(--bg-secondary)",
          borderColor: "var(--border)",
        }}
      >
        <CmdBtn label="Imprint All (50)" onClick={() => sendCmd("imprint", undefined, 50)} cls="action" />
        <CmdBtn label="Train Cycle (50/digit)" onClick={() => sendCmd("train_cycle", undefined, 50)} cls="action" />
        <CmdBtn label="Train Cycle (200/digit)" onClick={() => sendCmd("train_cycle", undefined, 200)} cls="action" />
        <CmdBtn label="Evaluate (1000)" onClick={() => sendCmd("evaluate", undefined, 1000)} />
        <CmdBtn label="Freeze All" onClick={() => sendCmd("freeze")} cls="freeze" />
      </div>

      {/* Detector grid */}
      <div
        className="rounded border p-1.5"
        style={{
          background: "var(--bg-secondary)",
          borderColor: "var(--border)",
        }}
      >
        <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
          Detectors
        </h2>
        <div className="grid grid-cols-5 gap-1">
          {state.detectors.map((d) => (
            <DetectorCard key={d.digit} det={d} />
          ))}
        </div>
      </div>

      {/* Bottom row: samples + charts */}
      <div className="flex gap-2 min-h-0 flex-1">
        {/* Samples */}
        <div
          className="flex-[2] rounded border p-1.5 overflow-auto"
          style={{
            background: "var(--bg-secondary)",
            borderColor: "var(--border)",
          }}
        >
          <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
            Sample Predictions
          </h2>
          <SampleGrid samples={state.samples} />
        </div>

        {/* Charts + confusion */}
        <div
          className="flex-1 flex flex-col gap-1.5 rounded border p-1.5"
          style={{
            background: "var(--bg-secondary)",
            borderColor: "var(--border)",
          }}
        >
          <div>
            <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
              Confusion Matrix
            </h2>
            <ConfusionMatrix confusion={state.confusion} />
          </div>
          <div>
            <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
              Accuracy History
            </h2>
            <AccuracyChart history={state.history} mode="overall" />
          </div>
          <div>
            <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
              Per-Detector
            </h2>
            <AccuracyChart history={state.history} mode="per-detector" />
          </div>
        </div>
      </div>

      {/* Log */}
      <div
        className="rounded border p-1.5 max-h-24 overflow-y-auto text-[11px] text-gray-400"
        style={{
          background: "var(--bg-secondary)",
          borderColor: "var(--border)",
        }}
      >
        <h2 className="text-xs text-gray-500 border-b border-gray-800 mb-1 pb-0.5">
          Log
        </h2>
        {state.log.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
    </div>
  );
}

function CmdBtn({
  label,
  cls,
  onClick,
}: {
  label: string;
  cls?: string;
  onClick: () => void;
}) {
  const base =
    "text-[11px] px-2 py-0.5 border rounded cursor-pointer bg-[#282828] text-gray-300";
  const extra =
    cls === "action"
      ? "border-green-800 hover:bg-green-950"
      : cls === "freeze"
        ? "border-blue-800 hover:bg-blue-950"
        : "border-gray-600 hover:bg-gray-800";
  return (
    <button className={`${base} ${extra}`} onClick={onClick}>
      {label}
    </button>
  );
}
