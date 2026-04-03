import { useRef, useEffect } from "react";

interface Props {
  thermals: number[];   // 784 temperature values (0-255)
  size?: number;
}

/** 28x28 heatmap showing thermal state of each weight.
 *  HOT (192+) = bright red, WARM (128-191) = orange,
 *  COOL (64-127) = blue, COLD (0-63) = gray. */
export function ThermalMatrix({ thermals, size = 56 }: Props) {
  const ref_ = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref_.current;
    if (!cv || thermals.length < 784) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(28, 28);
    for (let j = 0; j < 784; j++) {
      const t = thermals[j] ?? 0;
      const idx = j * 4;

      let r: number, g: number, b: number;
      if (t >= 192) {
        // HOT — bright red to white
        const i = (t - 192) / 63;
        r = 180 + Math.floor(75 * i);
        g = 30 + Math.floor(60 * i);
        b = 20 + Math.floor(40 * i);
      } else if (t >= 128) {
        // WARM — orange
        const i = (t - 128) / 63;
        r = 120 + Math.floor(80 * i);
        g = 60 + Math.floor(40 * i);
        b = 15;
      } else if (t >= 64) {
        // COOL — blue
        const i = (t - 64) / 63;
        r = 15;
        g = 30 + Math.floor(30 * i);
        b = 80 + Math.floor(100 * i);
      } else {
        // COLD — gray (darker = colder)
        const i = t / 63;
        const v = 20 + Math.floor(40 * i);
        r = v;
        g = v;
        b = v;
      }

      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [thermals]);

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
