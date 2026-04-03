import { useRef, useEffect } from "react";

interface Props {
  activations: number[];   // 784 glow values (0-255, decays over time)
  size?: number;
}

/** 28x28 golden phosphor display showing recent weight activations.
 *  Bright gold when freshly activated, decays to dim like an old mainframe CRT. */
export function ActivationMatrix({ activations, size = 56 }: Props) {
  const ref_ = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref_.current;
    if (!cv || activations.length < 784) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(28, 28);
    for (let j = 0; j < 784; j++) {
      const a = activations[j] ?? 0;
      const idx = j * 4;

      // Phosphor display: dim amber baseline → bright gold on spike
      const i = a / 255;
      img.data[idx] = Math.floor(18 + 237 * i);       // R: 18 (dim) → 255 (bright)
      img.data[idx + 1] = Math.floor(12 + 183 * i);   // G: 12 → 195
      img.data[idx + 2] = Math.floor(4 + 31 * i);     // B: 4 → 35
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [activations]);

  return (
    <canvas
      ref={ref_}
      width={28}
      height={28}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
      className="border border-gray-700"
    />
  );
}
