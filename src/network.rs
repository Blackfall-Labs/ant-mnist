//! 3-layer ThermalWeightMatrix network: 784 → 300 → 100 → 10.

use atomic_neural_transistors::core::{ThermalWeightMatrix, ThermalMasteryConfig};
use atomic_neural_transistors::core::weight_matrix::{packed_from_current, relu_packed};
use atomic_neural_transistors::thermal_mastery::ThermalMasteryState;
use atomic_neural_transistors::neuromod::{Chemical, NeuromodState};
use ternary_signal::PackedSignal;
use std::path::Path;

pub const INPUT_DIM: usize = 784;
pub const HIDDEN1_DIM: usize = 300;
pub const HIDDEN2_DIM: usize = 100;
pub const OUTPUT_DIM: usize = 10;

pub struct AntNetwork {
    pub w1: ThermalWeightMatrix, // 300×784
    pub w2: ThermalWeightMatrix, // 100×300
    pub w3: ThermalWeightMatrix, // 10×100
    pub m1: ThermalMasteryState,
    pub m2: ThermalMasteryState,
    pub m3: ThermalMasteryState,
    pub neuromod: NeuromodState,
    pub total_correct: u64,
    pub total_seen: u64,
}

impl AntNetwork {
    pub fn new() -> Self {
        let config = ThermalMasteryConfig {
            pressure_threshold: 4,
            decay_rate: 1,
            participation_gate: 5,
            cooling_rate: 50,
            warming_step: 10,
            warming_threshold: 20,
        };

        Self {
            w1: ThermalWeightMatrix::random_hot(HIDDEN1_DIM, INPUT_DIM, 42),
            w2: ThermalWeightMatrix::random_hot(HIDDEN2_DIM, HIDDEN1_DIM, 7919),
            w3: ThermalWeightMatrix::random_hot(OUTPUT_DIM, HIDDEN2_DIM, 104729),
            m1: ThermalMasteryState::new(config.clone()),
            m2: ThermalMasteryState::new(config.clone()),
            m3: ThermalMasteryState::new(config),
            neuromod: NeuromodState::new(),
            total_correct: 0,
            total_seen: 0,
        }
    }

    /// Forward pass. Returns (h1, h2, output, predicted_class).
    pub fn forward(&self, input: &[PackedSignal]) -> (Vec<PackedSignal>, Vec<PackedSignal>, Vec<PackedSignal>, usize) {
        let h1_raw = self.w1.matmul(input);
        let h1 = relu_packed(&h1_raw);

        let h2_raw = self.w2.matmul(&h1);
        let h2 = relu_packed(&h2_raw);

        let out = self.w3.matmul(&h2);

        // Winner = highest current
        let mut best_idx = 0;
        let mut best_val = i32::MIN;
        for (i, s) in out.iter().enumerate() {
            let c = s.current();
            if c > best_val { best_val = c; best_idx = i; }
        }

        (h1, h2, out, best_idx)
    }

    /// Classify without learning. Returns predicted class.
    pub fn predict(&self, input: &[PackedSignal]) -> usize {
        self.forward(input).3
    }

    /// Learn from one image. Returns predicted class.
    pub fn learn(&mut self, input: &[PackedSignal], correct_label: u8) -> usize {
        let (h1, h2, out, predicted) = self.forward(input);
        let correct = predicted == correct_label as usize;

        self.total_seen += 1;
        if correct { self.total_correct += 1; }

        // Build target: correct class excitatory, others inhibitory
        let mut target = vec![packed_from_current(-40); OUTPUT_DIM];
        target[correct_label as usize] = packed_from_current(60);

        // Neuromodulator feedback
        if correct {
            self.neuromod.inject(Chemical::Dopamine, 25);
        } else {
            self.neuromod.inject(Chemical::Dopamine, -15);
            self.neuromod.inject(Chemical::Norepinephrine, 20);
        }

        // Layer 3: output layer mastery
        self.m3.update(&mut self.w3, &h2, &out, &target, correct);

        // Layer 2: hidden2 — derive target from output error
        let mut h2_target = Vec::with_capacity(HIDDEN2_DIM);
        for i in 0..HIDDEN2_DIM {
            let mut drive = 0i32;
            for c in 0..OUTPUT_DIM {
                let err = target[c].current() - out[c].current();
                let w = self.w3.get(c, i).signal.current();
                drive += (err.signum() * w.signum()) as i32;
            }
            h2_target.push(packed_from_current(drive.clamp(-60, 60)));
        }
        self.m2.update(&mut self.w2, &h1, &h2, &h2_target, correct);

        // Layer 1: hidden1 — derive target from hidden2 error
        let mut h1_target = Vec::with_capacity(HIDDEN1_DIM);
        for i in 0..HIDDEN1_DIM {
            let mut drive = 0i32;
            for j in 0..HIDDEN2_DIM {
                let err = h2_target[j].current() - h2[j].current();
                let w = self.w2.get(j, i).signal.current();
                drive += (err.signum() * w.signum()) as i32;
            }
            h1_target.push(packed_from_current(drive.clamp(-60, 60)));
        }
        self.m1.update(&mut self.w1, input, &h1, &h1_target, correct);

        // Decay + tick
        self.m3.decay(&mut self.w3);
        self.m2.decay(&mut self.w2);
        self.m1.decay(&mut self.w1);
        self.neuromod.tick();

        predicted
    }

    /// Accuracy so far.
    pub fn accuracy(&self) -> f32 {
        if self.total_seen > 0 { self.total_correct as f32 / self.total_seen as f32 } else { 0.0 }
    }

    /// Reset counters (between cycles).
    pub fn reset_counters(&mut self) {
        self.total_correct = 0;
        self.total_seen = 0;
    }

    /// Thermal summary per layer.
    pub fn thermal_summary(&self) -> String {
        let (h1, w1, c1, f1) = self.w1.temp_summary();
        let (h2, w2, c2, f2) = self.w2.temp_summary();
        let (h3, w3, c3, f3) = self.w3.temp_summary();
        format!(
            "L1[H{}W{}C{}F{}] L2[H{}W{}C{}F{}] L3[H{}W{}C{}F{}]",
            h1, w1, c1, f1, h2, w2, c2, f2, h3, w3, c3, f3
        )
    }

    /// Save all layers to .ant file.
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        ThermalWeightMatrix::save_multi(&[&self.w1, &self.w2, &self.w3], path)?;
        Ok(())
    }

    /// Load all layers from .ant file.
    pub fn load(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut layers = ThermalWeightMatrix::load_multi(path)?;
        if layers.len() != 3 { return Err("expected 3 layers".into()); }
        self.w3 = layers.pop().unwrap();
        self.w2 = layers.pop().unwrap();
        self.w1 = layers.pop().unwrap();
        Ok(())
    }
}
