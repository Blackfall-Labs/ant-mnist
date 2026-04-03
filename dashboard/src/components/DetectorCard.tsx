import type { DetViz } from "@/types";
import { TemplateHeatmap } from "./TemplateHeatmap";
import { ThermalMatrix } from "./ThermalMatrix";
import { ActivationMatrix } from "./ActivationMatrix";
import { Sparkline } from "./Sparkline";
import { useCtrl } from "@/context";

const PHASE_STYLES: Record<string, string> = {
  Cold: "bg-gray-800 text-gray-500",
  Imprinted: "bg-yellow-950 text-yellow-400",
  Training: "bg-green-950 text-green-400",
  Frozen: "bg-blue-950 text-blue-400",
};

interface Props {
  det: DetViz;
}

export function DetectorCard({ det }: Props) {
  const { sendCmd } = useCtrl();
  const accPct = (det.eval_accuracy * 100).toFixed(0);
  const accColor =
    det.eval_accuracy > 0.8
      ? "var(--accent-green)"
      : det.eval_accuracy > 0.5
        ? "#ff8833"
        : "#888";

  return (
    <div className="bg-[#161616] border border-gray-800 rounded p-1.5">
      {/* Header */}
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-bold" style={{ color: "var(--accent-orange)" }}>
          Digit {det.digit}
        </span>
        <span
          className={`text-[9px] px-1 rounded uppercase ${PHASE_STYLES[det.phase] ?? ""}`}
        >
          {det.phase}
        </span>
        <span className="text-xs font-bold" style={{ color: accColor }}>
          {accPct}%
        </span>
      </div>

      {/* Controls */}
      <div className="flex gap-0.5 mb-1">
        <Btn label="Imprint" cls="action" onClick={() => sendCmd("imprint", det.digit, 50)} />
        <Btn label="Train" cls="action" onClick={() => sendCmd("train_cycle", det.digit, 50)} />
        <Btn label="Freeze" cls="freeze" onClick={() => sendCmd("freeze", det.digit)} />
        <Btn label="Thaw" onClick={() => sendCmd("unfreeze", det.digit)} />
        <Btn label="Flush" cls="danger" onClick={() => sendCmd("flush", det.digit)} />
      </div>

      {/* Heatmaps row: template | thermals | activations */}
      <div className="flex gap-1">
        <div className="text-center">
          <TemplateHeatmap template={det.template} size={56} />
          <span className="text-[8px] text-gray-600">Template</span>
        </div>
        <div className="text-center">
          <ThermalMatrix thermals={det.thermals} size={56} />
          <span className="text-[8px] text-gray-600">Thermal</span>
        </div>
        <div className="text-center">
          <ActivationMatrix activations={det.activations} size={56} />
          <span className="text-[8px] text-gray-600">Activity</span>
        </div>
      </div>

      {/* Sparkline */}
      <div className="mt-1">
        <Sparkline data={det.confidence_history} digit={det.digit} height={16} />
      </div>
    </div>
  );
}

function Btn({
  label,
  cls,
  onClick,
}: {
  label: string;
  cls?: string;
  onClick: () => void;
}) {
  const base =
    "text-[10px] px-1.5 py-0.5 border rounded cursor-pointer bg-[#282828] text-gray-300";
  const extra =
    cls === "action"
      ? "border-green-800 hover:bg-green-950"
      : cls === "freeze"
        ? "border-blue-800 hover:bg-blue-950"
        : cls === "danger"
          ? "border-red-800 hover:bg-red-950"
          : "border-gray-600 hover:bg-gray-800";
  return (
    <button className={`${base} ${extra}`} onClick={onClick}>
      {label}
    </button>
  );
}
