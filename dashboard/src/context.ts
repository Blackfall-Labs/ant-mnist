import { createContext, useContext } from "react";
import type { ControlState } from "@/types";

export interface ControlHandle {
  state: ControlState;
  connected: boolean;
  sendCmd: (action: string, digit?: number, count?: number) => Promise<void>;
}

export const StateContext = createContext<ControlHandle>(null!);

export function useCtrl(): ControlHandle {
  return useContext(StateContext);
}
