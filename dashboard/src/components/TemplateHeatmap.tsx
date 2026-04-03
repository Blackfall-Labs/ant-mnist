import { useRef, useEffect } from "react";

interface Props {
  template: number[];
  size?: number;
}

/** 28x28 heatmap of a detector's imprinted template.
 *  Green = excitatory, Red = inhibitory, Dark = silent. */
export function TemplateHeatmap({ template, size = 56 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || template.length < 784) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(28, 28);
    const maxT = Math.max(1, ...template.map((v) => Math.abs(v)));

    for (let j = 0; j < 784; j++) {
      const v = template[j] ?? 0;
      const norm = Math.abs(v) / maxT;
      const idx = j * 4;
      if (v > 0) {
        img.data[idx] = 0;
        img.data[idx + 1] = Math.floor(norm * 255);
        img.data[idx + 2] = 0;
      } else if (v < 0) {
        img.data[idx] = Math.floor(norm * 255);
        img.data[idx + 1] = 0;
        img.data[idx + 2] = 0;
      } else {
        img.data[idx] = 20;
        img.data[idx + 1] = 20;
        img.data[idx + 2] = 20;
      }
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [template]);

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
