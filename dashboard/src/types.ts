export type Phase = "Cold" | "Imprinted" | "Training" | "Frozen";

export interface DetViz {
  digit: number;
  phase: Phase;
  template: number[];       // 784 values (frozen imprint snapshot)
  learned: number[];        // 784 live weight currents (changes during training)
  correlation: number[];    // 784 last correlation output
  eval_accuracy: number;
  confidence_history: number[];
  thermals: number[];       // 784 temperature values (0-255, HOT→COLD)
  activations: number[];    // 784 activation glow values (0-255, decays)
  mastery_steps: number;
  mastery_transitions: number;
  weight_currents?: number[];
  weight_pressures?: number[];
}

export interface SampleViz {
  pixels: number[];
  label: number;
  predicted: number;
  correct: boolean;
  confidences: number[];    // 10 scalar confidences
}

export interface Snap {
  cycle: number;
  test_acc: number;
  det_accs: number[];
}

export interface FinalizerViz {
  class_magnitudes: number[];
  class_nonzero: number[];
  total_weights: number;
  nonzero_weights: number;
  steps: number;
  transitions: number;
}

export interface SignalFlow {
  label: number;
  predicted: number;
  detector_winner: number;
  raw_confidences: number[];
  inhibited_confidences: number[];
  finalizer_scores: number[];
}

export interface ControlState {
  detectors: DetViz[];
  finalizer: FinalizerViz;
  signal_flow: SignalFlow;
  samples: SampleViz[];
  confusion: number[][];    // 10x10
  history: Snap[];
  test_acc: number;
  best_test_acc: number;
  cycle: number;
  log: string[];
}
