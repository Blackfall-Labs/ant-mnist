import { MnistImage } from "./MnistImage";
import type { SampleViz } from "@/types";

interface Props {
  samples: SampleViz[];
}

/** Grid of sample predictions — 10 columns, color-coded correct/wrong. */
export function SampleGrid({ samples }: Props) {
  if (!samples || samples.length === 0) return null;

  return (
    <div className="grid grid-cols-10 gap-1">
      {samples.map((s, i) => (
        <div
          key={i}
          className={`text-center p-0.5 rounded ${
            s.correct
              ? "bg-green-950 border border-green-900"
              : "bg-red-950 border border-red-900"
          }`}
        >
          <MnistImage
            pixels={s.pixels}
            size={42}
            borderColor={s.correct ? "#4caf50" : "#f44336"}
          />
          <div className="text-[10px] mt-0.5">
            <span className="text-gray-500">{s.label}</span>
            <span className="text-gray-600">→</span>
            <span style={{ color: s.correct ? "#4caf50" : "#f44336" }}>
              {s.predicted}
            </span>
          </div>
          <ConfidenceBars confidences={s.confidences} predicted={s.predicted} label={s.label} />
        </div>
      ))}
    </div>
  );
}

function ConfidenceBars({
  confidences,
  predicted,
  label,
}: {
  confidences: number[];
  predicted: number;
  label: number;
}) {
  const mx = Math.max(1, ...confidences.map((c) => Math.abs(c)));
  return (
    <div className="flex h-5 gap-px mt-0.5">
      {confidences.map((c, i) => {
        const heightPct = (Math.abs(c) / mx) * 100;
        let color = "#444";
        if (i === predicted) color = "#ff6b35";
        else if (i === label) color = "#4caf50";
        const opacity = c < 0 ? 0.3 : 1;
        return (
          <div key={i} className="flex-1 flex flex-col justify-end" title={`d${i}:${c}`}>
            <div
              style={{
                height: `${heightPct}%`,
                background: color,
                opacity,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}
