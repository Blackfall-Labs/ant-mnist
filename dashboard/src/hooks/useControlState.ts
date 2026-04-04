import { useState, useEffect, useCallback, useRef } from "react";
import type { ControlState } from "@/types";

const EMPTY_STATE: ControlState = {
  detectors: [],
  finalizer: { class_magnitudes: [], class_nonzero: [], total_weights: 0, nonzero_weights: 0, steps: 0, transitions: 0 },
  signal_flow: { label: 0, predicted: 0, detector_winner: 0, raw_confidences: [], inhibited_confidences: [], finalizer_scores: [] },
  samples: [],
  confusion: [],
  history: [],
  test_acc: 0,
  best_test_acc: 0,
  cycle: 0,
  log: [],
};

export function useControlState() {
  const [state, setState] = useState<ControlState>(EMPTY_STATE);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      if (cancelled) return;
      const wsUrl = `ws://${window.location.hostname}:8091`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!cancelled) setConnected(true);
      };

      ws.onmessage = (event) => {
        if (cancelled) return;
        try {
          const data: ControlState = JSON.parse(event.data);
          setState(data);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        wsRef.current = null;
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on cleanup
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  const sendCmd = useCallback(
    async (action: string, digit?: number, count?: number) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const body: Record<string, unknown> = { action };
      if (digit !== undefined) body.digit = digit;
      if (count !== undefined) body.count = count;
      ws.send(JSON.stringify(body));
    },
    []
  );

  return { state, connected, sendCmd };
}
