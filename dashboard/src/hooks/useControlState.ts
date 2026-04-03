import { useState, useEffect, useCallback, useRef } from "react";
import type { ControlState } from "@/types";

const EMPTY_STATE: ControlState = {
  detectors: [],
  finalizer: { class_magnitudes: [], class_nonzero: [], total_weights: 0, nonzero_weights: 0, steps: 0, transitions: 0 },
  samples: [],
  confusion: [],
  history: [],
  test_acc: 0,
  best_test_acc: 0,
  cycle: 0,
  log: [],
};

export function useControlState(pollInterval = 2000) {
  const [state, setState] = useState<ControlState>(EMPTY_STATE);
  const [connected, setConnected] = useState(false);
  const mountedRef = useRef(true);

  const poll = useCallback(async () => {
    try {
      const r = await fetch("/api/state");
      if (!mountedRef.current) return;
      const data: ControlState = await r.json();
      setState(data);
      setConnected(true);
    } catch {
      if (mountedRef.current) setConnected(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    poll();
    const id = setInterval(poll, pollInterval);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [poll, pollInterval]);

  const sendCmd = useCallback(
    async (action: string, digit?: number, count?: number) => {
      const body: Record<string, unknown> = { action };
      if (digit !== undefined) body.digit = digit;
      if (count !== undefined) body.count = count;
      await fetch("/api/cmd", {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      });
      // Poll shortly after command completes
      setTimeout(poll, 300);
    },
    [poll]
  );

  return { state, connected, sendCmd };
}
