import { useRef, useEffect } from "react";

interface Props {
  pixels: number[];
  size?: number;
  borderColor?: string;
}

/** Renders a 28x28 grayscale MNIST image to a canvas. */
export function MnistImage({ pixels, size = 42, borderColor }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || pixels.length < 784) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = pixels[i];
      const idx = i * 4;
      img.data[idx] = v;
      img.data[idx + 1] = v;
      img.data[idx + 2] = v;
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      style={{
        width: size,
        height: size,
        imageRendering: "pixelated",
        borderColor: borderColor ?? "#333",
      }}
      className="border"
    />
  );
}
