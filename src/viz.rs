//! ANT-MNIST Control Room v2
//!
//! Architecture: 10 spatial correlation detectors → finalizer ANT
//! Each detector returns 784 confidence signals.
//! Finalizer takes 7840 signals → 10 class outputs.

mod mnist;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use serde::{Serialize, Deserialize};
use tiny_http::{Server, Response, Header, Method};

use atomic_neural_transistors::modules::ant_ml::{AntMlModule, AntRuntime};
use runes_core::engine::Engine;
use runes_core::value::Value;
use runes_parser::lexer::Lexer;
use runes_parser::parser::Parser;
use runes_eval::evaluator::Evaluator;

use mnist::MnistDataset;

const DETECTOR_SOURCE: &str = include_str!("../runes/digit_detector.rune");
const FINALIZER_SOURCE: &str = include_str!("../runes/finalizer.rune");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Copy, PartialEq)]
enum Phase { Cold, Imprinted, Training, Frozen }
impl Default for Phase { fn default() -> Self { Self::Cold } }

#[derive(Serialize, Clone, Default)]
struct FinalizerViz {
    class_magnitudes: Vec<i64>,     // 10 values: sum of |weight| per output class
    class_nonzero: Vec<u32>,        // 10 values: count of nonzero weights per class
    total_weights: usize,
    nonzero_weights: usize,
    steps: u64,
    transitions: u64,
}

/// Live signal flow for the last processed sample — shows the full pipeline.
#[derive(Serialize, Clone, Default)]
struct SignalFlow {
    label: u8,                          // true label
    predicted: u8,                      // finalizer prediction
    detector_winner: u8,                // which detector had highest raw confidence
    raw_confidences: Vec<i64>,          // 10 raw detector scalar confidences (pre-inhibition)
    inhibited_confidences: Vec<i64>,    // 10 post-inhibition confidences
    finalizer_scores: Vec<i64>,         // 10 finalizer output scores
}

#[derive(Serialize, Clone, Default)]
struct ControlState {
    detectors: Vec<DetViz>,
    finalizer: FinalizerViz,
    signal_flow: SignalFlow,
    samples: Vec<SampleViz>,
    confusion: [[u32; 10]; 10],
    history: Vec<Snap>,
    test_acc: f32,
    best_test_acc: f32,
    cycle: u32,
    log: Vec<String>,
}

#[derive(Serialize, Clone, Default)]
struct DetViz {
    digit: usize,
    phase: Phase,
    template: Vec<i32>,     // 784 imprint snapshot (frozen at imprint time)
    learned: Vec<i32>,      // 784 live weight currents (changes during training)
    correlation: Vec<i64>,  // 784 last correlation output (what this detector saw)
    eval_accuracy: f32,
    confidence_history: Vec<i64>,
    thermals: Vec<u8>,      // 784 temperature values (0-255, HOT→COLD)
    activations: Vec<u8>,   // 784 activation intensities (0-255, decays over time)
    mastery_steps: u64,
    mastery_transitions: u64,
}

#[derive(Serialize, Clone)]
struct SampleViz {
    pixels: Vec<u8>,
    label: u8,
    predicted: u8,
    correct: bool,
    confidences: Vec<i64>,  // 10 scalar confidences (one per detector)
}

#[derive(Serialize, Clone)]
struct Snap { cycle: u32, test_acc: f32, det_accs: Vec<f32> }

// ---------------------------------------------------------------------------
// Detector wrapper
// ---------------------------------------------------------------------------

struct Detector {
    evaluator: Evaluator,
    engine: Engine,
    digit: usize,
    runtime: Arc<Mutex<AntRuntime>>,
    phase: Phase,
    imprint_count: u32,
    eval_correct: u32,
    eval_total: u32,
    confidence_history: Vec<i64>,
    activations: Vec<u8>,   // 784 activation glow values, decayed each poll
    imprint_snapshot: Vec<i32>,  // frozen copy of template after imprinting
    last_correlation: Vec<i64>,  // 784 last correlation output
}

impl Detector {
    fn new(digit: usize) -> Self {
        let engine = Engine::builder()
            .namespace("default").module(AntMlModule::new()).build().unwrap();
        let tokens = Lexer::new(DETECTOR_SOURCE).tokenize().unwrap();
        let program = Parser::new(tokens).parse_program().unwrap();
        let runtime = Arc::new(Mutex::new(AntRuntime::new()));
        let mut evaluator = Evaluator::new();
        evaluator.set_host(runtime.clone());
        evaluator.define_variable("prefix", Value::String(format!("d{}", digit).into()));
        evaluator.eval_with_engine(&program, &engine).unwrap();
        Self { evaluator, engine, digit, runtime, phase: Phase::Cold,
               imprint_count: 0, eval_correct: 0, eval_total: 0,
               confidence_history: Vec::new(), activations: vec![0u8; 784],
               imprint_snapshot: vec![0; 784], last_correlation: vec![0; 784] }
    }

    fn imprint_sample(&mut self, input: &[Value]) {
        // Template is 1 row × 784 cols. All samples go to row 0 (accumulated).
        let input_val = Value::Array(Arc::new(input.to_vec()));
        let row_val = Value::Integer(0);
        self.evaluator.invoke("imprint", &[input_val, row_val], &self.engine).ok();
        self.imprint_count += 1;
    }

    fn finish_imprint(&mut self) {
        if self.imprint_count > 0 {
            let count = Value::Integer(self.imprint_count as i64);
            self.evaluator.invoke("finish_imprint", &[count], &self.engine).ok();
            // Snapshot the template at imprint time — this never changes
            self.imprint_snapshot = self.get_current_weights();
        }
    }

    /// Returns 784 correlation signals as Value::Array, stores for viz.
    fn correlate(&mut self, input: &[Value]) -> Value {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        let result = self.evaluator.invoke("correlate", &[input_val], &self.engine)
            .unwrap_or(Value::Array(Arc::new(vec![Value::Integer(0); 784])));
        // Store for visualization
        if let Value::Array(arr) = &result {
            self.last_correlation = arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect();
        }
        result
    }

    /// Returns scalar confidence
    fn confidence(&mut self, input: &[Value]) -> i64 {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        match self.evaluator.invoke("confidence", &[input_val], &self.engine) {
            Ok(Value::Integer(n)) => n,
            _ => 0,
        }
    }

    /// Train the detector template via thermal mastery.
    /// is_mine: true if this sample belongs to this detector's digit.
    fn train_sample(&mut self, input: &[Value], is_mine: bool) {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        let is_mine_val = Value::Integer(if is_mine { 1 } else { 0 });
        self.evaluator.invoke("train", &[input_val, is_mine_val], &self.engine).ok();
        if is_mine {
            self.phase = Phase::Training;
        }
    }

    /// Decay pressure on all template weights. Call once per cycle.
    fn decay_pressure(&mut self) {
        self.evaluator.invoke("decay", &[], &self.engine).ok();
    }

    /// Current live weight values (changes during training)
    fn get_current_weights(&self) -> Vec<i32> {
        let rt = self.runtime.lock().unwrap();
        let key = format!("d{}.template", self.digit);
        if let Some(twm) = rt.thermal_weights_by_key(&key) {
            twm.data.iter().take(784).map(|tw| tw.current()).collect()
        } else {
            vec![0; 784]
        }
    }

    fn get_thermals(&self) -> Vec<u8> {
        let rt = self.runtime.lock().unwrap();
        let key = format!("d{}.template", self.digit);
        if let Some(twm) = rt.thermal_weights_by_key(&key) {
            twm.data.iter().take(784).map(|tw| tw.temperature).collect()
        } else {
            vec![0; 784]
        }
    }

    /// Mark activation from correlation output — which weights actually spiked.
    /// Uses the correlation result (output signals), not the input.
    fn mark_activations_from_correlation(&mut self, correlation: &Value) {
        if let Value::Array(arr) = correlation {
            for (i, v) in arr.iter().enumerate().take(784) {
                if let Value::Integer(n) = v {
                    // Positive polarity + high magnitude = spike
                    if *n > 0 {
                        let intensity = ((*n).min(60) * 255 / 60) as u8;
                        self.activations[i] = self.activations[i].max(intensity);
                    }
                }
            }
        }
    }

    /// Decay all activations toward zero (phosphor fade).
    fn decay_activations(&mut self) {
        for a in &mut self.activations {
            *a = a.saturating_sub(40); // ~6 polls to fully decay from 255
        }
    }

    fn get_mastery_stats(&self) -> (u64, u64) {
        let rt = self.runtime.lock().unwrap();
        let key = format!("d{}.template", self.digit);
        if let Some(&handle) = rt.synaptic_key_handles().get(&key) {
            if let Some(ms) = rt.get_thermal_mastery(handle) {
                return (ms.steps, ms.transitions);
            }
        }
        (0, 0)
    }

    fn get_viz(&self) -> DetViz {
        let (steps, transitions) = self.get_mastery_stats();
        DetViz {
            digit: self.digit,
            phase: self.phase,
            template: self.imprint_snapshot.clone(),
            learned: self.get_current_weights(),
            correlation: self.last_correlation.clone(),
            eval_accuracy: if self.eval_total > 0 { self.eval_correct as f32 / self.eval_total as f32 } else { 0.0 },
            confidence_history: self.confidence_history.clone(),
            thermals: self.get_thermals(),
            activations: self.activations.clone(),
            mastery_steps: steps,
            mastery_transitions: transitions,
        }
    }

    fn flush(&mut self) { *self = Self::new(self.digit); }
}

// ---------------------------------------------------------------------------
// Finalizer wrapper
// ---------------------------------------------------------------------------

struct Finalizer {
    evaluator: Evaluator,
    engine: Engine,
    runtime: Arc<Mutex<AntRuntime>>,
}

impl Finalizer {
    fn new() -> Self {
        let engine = Engine::builder()
            .namespace("default").module(AntMlModule::new()).build().unwrap();
        let tokens = Lexer::new(FINALIZER_SOURCE).tokenize().unwrap();
        let program = Parser::new(tokens).parse_program().unwrap();
        let runtime = Arc::new(Mutex::new(AntRuntime::new()));
        let mut evaluator = Evaluator::new();
        evaluator.set_host(runtime.clone());
        evaluator.eval_with_engine(&program, &engine).unwrap();
        Self { evaluator, engine, runtime }
    }

    fn predict(&mut self, correlations: &Value) -> usize {
        match self.evaluator.invoke("predict", &[correlations.clone()], &self.engine) {
            Ok(Value::Integer(n)) => n as usize,
            _ => 0,
        }
    }

    /// Forward pass returning the 10 raw output scores.
    fn forward(&mut self, correlations: &Value) -> Vec<i64> {
        match self.evaluator.invoke("forward", &[correlations.clone()], &self.engine) {
            Ok(Value::Array(arr)) => arr.iter().map(|v| {
                if let Value::Integer(n) = v { *n } else { 0 }
            }).collect(),
            _ => vec![0; 10],
        }
    }

    fn train(&mut self, correlations: &Value, label: usize) -> usize {
        let label_val = Value::Integer(label as i64);
        match self.evaluator.invoke("train", &[correlations.clone(), label_val], &self.engine) {
            Ok(Value::Integer(n)) => n as usize,
            _ => 0,
        }
    }

    fn decay(&mut self) {
        self.evaluator.invoke("decay", &[], &self.engine).ok();
    }

    fn get_viz(&self) -> FinalizerViz {
        let rt = self.runtime.lock().unwrap();
        if let Some(wm) = rt.weights_by_key("finalizer.w") {
            let mut class_magnitudes = vec![0i64; 10];
            let mut class_nonzero = vec![0u32; 10];
            let mut total_nonzero = 0usize;

            for row in 0..wm.rows.min(10) {
                for col in 0..wm.cols {
                    let s = &wm.data[row * wm.cols + col];
                    let c = s.current();
                    class_magnitudes[row] += c.abs() as i64;
                    if c != 0 {
                        class_nonzero[row] += 1;
                        total_nonzero += 1;
                    }
                }
            }

            // Get mastery state if available
            let (steps, transitions) = if let Some(&handle) = rt.synaptic_key_handles().get("finalizer.w") {
                if let Some(ms) = rt.get_mastery(handle) {
                    (ms.steps, ms.transitions)
                } else { (0, 0) }
            } else { (0, 0) };

            FinalizerViz {
                class_magnitudes,
                class_nonzero,
                total_weights: wm.data.len(),
                nonzero_weights: total_nonzero,
                steps,
                transitions,
            }
        } else {
            FinalizerViz::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn encode_pixel(px: u8) -> i64 {
    if px <= 84 { -(((84 - px as i32) * 60) / 84) as i64 }
    else if px >= 171 { (((px as i32 - 171) * 60) / 84) as i64 }
    else { 0 }
}

fn encode_image(pixels: &[u8]) -> Vec<Value> {
    pixels.iter().map(|&px| Value::Integer(encode_pixel(px))).collect()
}

fn log_msg(state: &Arc<Mutex<ControlState>>, msg: String) {
    eprintln!("  {}", msg);
    let mut s = state.lock().unwrap();
    s.log.insert(0, msg);
    if s.log.len() > 50 { s.log.truncate(50); }
}

/// Compute scalar confidence per detector (sum of absolute correlation values).
fn scalar_confidences(correlations: &[Value]) -> Vec<i64> {
    correlations.iter().map(|c| {
        if let Value::Array(arr) = c {
            arr.iter().map(|v| if let Value::Integer(n) = v { n.abs() } else { 0 }).sum::<i64>()
        } else { 0 }
    }).collect()
}

/// Cross-ANT inhibition: suppress losing detectors' correlations.
/// Winner passes at full strength. Others are scaled down proportional
/// to how much weaker they are than the winner.
fn inhibit_correlations(correlations: &mut Vec<Value>, confs: &[i64]) {
    let max_conf = *confs.iter().max().unwrap_or(&1);
    if max_conf == 0 { return; }

    for (i, corr) in correlations.iter_mut().enumerate() {
        let scale = confs[i] as f64 / max_conf as f64;
        // Winner gets scale=1.0, others get proportionally less.
        // Square the scale to make suppression more aggressive.
        let scale = scale * scale;

        if scale < 0.99 {
            if let Value::Array(arr) = corr {
                let inhibited: Vec<Value> = arr.iter().map(|v| {
                    if let Value::Integer(n) = v {
                        Value::Integer((*n as f64 * scale) as i64)
                    } else {
                        v.clone()
                    }
                }).collect();
                *corr = Value::Array(Arc::new(inhibited));
            }
        }
    }
}

/// Normalize each detector's 784 correlation signals to a consistent scale.
/// Divides each signal by the detector's max absolute value, then scales to [-60, 60].
/// This ensures the finalizer sees signals on the same scale regardless of detector magnitude.
fn normalize_correlations(correlations: &mut Vec<Value>) {
    for corr in correlations.iter_mut() {
        if let Value::Array(arr) = corr {
            let max_abs = arr.iter().map(|v| {
                if let Value::Integer(n) = v { n.abs() } else { 0 }
            }).max().unwrap_or(1).max(1);

            let normalized: Vec<Value> = arr.iter().map(|v| {
                if let Value::Integer(n) = v {
                    Value::Integer(*n * 60 / max_abs)
                } else {
                    v.clone()
                }
            }).collect();
            *corr = Value::Array(Arc::new(normalized));
        }
    }
}

/// Concatenate 10 correlation arrays into one 7840-element array
fn concat_correlations(corrs: &[Value]) -> Value {
    let mut all = Vec::with_capacity(7840);
    for c in corrs {
        if let Value::Array(arr) = c {
            for v in arr.iter() { all.push(v.clone()); }
        }
    }
    Value::Array(Arc::new(all))
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Cmd { action: String, #[serde(default)] digit: Option<usize>, #[serde(default)] count: Option<usize> }

/// Three-phase training curriculum.
#[derive(Clone, Copy, PartialEq)]
enum CurriculumPhase {
    /// Phase 1: Each detector trains ONLY on its own digit. Isolation.
    Isolation,
    /// Phase 2: Each detector trains on what ISN'T theirs. Adversarial hardening.
    Adversarial,
    /// Phase 3: Detectors frozen. Only the finalizer trains on stable outputs.
    Finalizer,
}

struct Curriculum {
    phase: CurriculumPhase,
    remaining_in_phase: u32,
    isolation_per_phase: u32,
    adversarial_per_phase: u32,
    finalizer_per_phase: u32,
    step_in_phase: u32,
}

impl Curriculum {
    fn new(isolation: u32, adversarial: u32, finalizer: u32) -> Self {
        Self {
            phase: CurriculumPhase::Isolation,
            remaining_in_phase: isolation,
            isolation_per_phase: isolation,
            adversarial_per_phase: adversarial,
            finalizer_per_phase: finalizer,
            step_in_phase: 0,
        }
    }

    fn advance_phase(&mut self) -> bool {
        match self.phase {
            CurriculumPhase::Isolation => {
                self.phase = CurriculumPhase::Adversarial;
                self.remaining_in_phase = self.adversarial_per_phase;
                self.step_in_phase = 0;
                true
            }
            CurriculumPhase::Adversarial => {
                self.phase = CurriculumPhase::Finalizer;
                self.remaining_in_phase = self.finalizer_per_phase;
                self.step_in_phase = 0;
                true
            }
            CurriculumPhase::Finalizer => false, // done
        }
    }

    fn phase_name(&self) -> &'static str {
        match self.phase {
            CurriculumPhase::Isolation => "Phase 1: Isolation",
            CurriculumPhase::Adversarial => "Phase 2: Adversarial",
            CurriculumPhase::Finalizer => "Phase 3: Finalizer",
        }
    }
}

fn handle_cmd(
    cmd: &Cmd,
    detectors: &mut Vec<Detector>,
    finalizer: &mut Finalizer,
    train: &MnistDataset,
    test: &MnistDataset,
    state: &Arc<Mutex<ControlState>>,
) {
    match cmd.action.as_str() {
        "imprint" => {
            let digits: Vec<usize> = cmd.digit.map(|d| vec![d]).unwrap_or((0..10).collect());
            let count = cmd.count.unwrap_or(50);

            let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
            for (i, img) in train.images.iter().enumerate() {
                let d = img.label as usize;
                if d < 10 && by_digit[d].len() < count { by_digit[d].push(i); }
            }

            for &digit in &digits {
                if digit >= 10 { continue; }
                let det = &mut detectors[digit];
                det.phase = Phase::Imprinted;
                det.imprint_count = 0;

                for &img_idx in &by_digit[digit] {
                    let encoded = encode_image(&train.images[img_idx].pixels);
                    det.imprint_sample(&encoded);
                }
                det.finish_imprint();

                let template = det.get_current_weights();
                let nonzero = template.iter().filter(|&&v| v != 0).count();
                let min_v = template.iter().min().copied().unwrap_or(0);
                let max_v = template.iter().max().copied().unwrap_or(0);
                log_msg(state, format!("Imprinted d{}: {} samples → min={} max={} nonzero={}/784",
                    digit, by_digit[digit].len(), min_v, max_v, nonzero));
            }

            // Auto-eval
            let eval_cmd = Cmd { action: "evaluate".into(), digit: None, count: Some(500) };
            handle_cmd(&eval_cmd, detectors, finalizer, train, test, state);
        }

        "train_cycle" => {
            let t0 = Instant::now();
            let per_digit = cmd.count.unwrap_or(50);

            // Build balanced set
            let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
            for (i, img) in train.images.iter().enumerate() {
                let d = img.label as usize;
                if d < 10 && by_digit[d].len() < per_digit { by_digit[d].push(i); }
            }
            let mut train_set: Vec<(usize, u8)> = Vec::new();
            for d in 0..10 { for &idx in &by_digit[d] { train_set.push((idx, d as u8)); } }

            let mut correct = 0u32;
            for &(img_idx, label) in &train_set {
                let encoded = encode_image(&train.images[img_idx].pixels);

                // Train each detector: is this sample my digit?
                for d in detectors.iter_mut() {
                    let is_mine = d.digit == label as usize;
                    d.train_sample(&encoded, is_mine);
                }

                // Get 10 correlation maps + inhibit losers + normalize
                let mut correlations: Vec<Value> = detectors.iter_mut()
                    .map(|d| d.correlate(&encoded))
                    .collect();
                let confs = scalar_confidences(&correlations);
                inhibit_correlations(&mut correlations, &confs);
                normalize_correlations(&mut correlations);

                for (d, c) in detectors.iter_mut().zip(correlations.iter()) {
                    d.mark_activations_from_correlation(c);
                }
                let combined = concat_correlations(&correlations);

                // Train finalizer
                let pred = finalizer.train(&combined, label as usize);
                if pred == label as usize {
                    correct += 1;
                } else {
                    // Negative reinforcement: penalize the winning detector if it was wrong.
                    // The detector that had highest confidence led to a wrong prediction —
                    // train it with is_mine=false to weaken its response to this input.
                    let winner = confs.iter().enumerate()
                        .max_by_key(|(_, c)| *c)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if winner != label as usize {
                        detectors[winner].train_sample(&encoded, false);
                    }
                }
            }

            // Decay pressure on all detectors + finalizer
            for d in detectors.iter_mut() { d.decay_pressure(); }
            finalizer.decay();

            {
                let mut s = state.lock().unwrap();
                s.cycle += 1;
            }
            log_msg(state, format!("Trained: {}/{} ({:.1}%) {}/digit ({:.1}s)",
                correct, train_set.len(), correct as f64 / train_set.len() as f64 * 100.0,
                per_digit, t0.elapsed().as_secs_f64()));

            // Auto-eval
            let eval_cmd = Cmd { action: "evaluate".into(), digit: None, count: Some(500) };
            handle_cmd(&eval_cmd, detectors, finalizer, train, test, state);
        }

        "evaluate" => {
            let test_n = cmd.count.unwrap_or(1000).min(test.len());

            for d in detectors.iter_mut() { d.eval_correct = 0; d.eval_total = 0; }

            let mut test_correct = 0u32;
            let mut confusion = [[0u32; 10]; 10];
            let mut samples = Vec::new();

            for i in 0..test_n {
                let img = &test.images[i];
                let encoded = encode_image(&img.pixels);
                let label = img.label as usize;

                // Correlations + inhibit losers + normalize + mark activations
                let mut correlations: Vec<Value> = detectors.iter_mut()
                    .map(|d| d.correlate(&encoded))
                    .collect();
                let raw_confs = scalar_confidences(&correlations);
                inhibit_correlations(&mut correlations, &raw_confs);
                normalize_correlations(&mut correlations);
                let inhibited_confs = scalar_confidences(&correlations);

                for (d, c) in detectors.iter_mut().zip(correlations.iter()) {
                    d.mark_activations_from_correlation(c);
                }

                let combined = concat_correlations(&correlations);
                let finalizer_scores = finalizer.forward(&combined);
                let predicted = finalizer_scores.iter().enumerate()
                    .max_by_key(|(_, s)| *s).map(|(i, _)| i).unwrap_or(0);

                // Update signal flow for latest sample
                let detector_winner = raw_confs.iter().enumerate()
                    .max_by_key(|(_, c)| *c).map(|(i, _)| i).unwrap_or(0);
                state.lock().unwrap().signal_flow = SignalFlow {
                    label: img.label,
                    predicted: predicted as u8,
                    detector_winner: detector_winner as u8,
                    raw_confidences: raw_confs.clone(),
                    inhibited_confidences: inhibited_confs,
                    finalizer_scores,
                };
                let scalar_confs = raw_confs;

                confusion[label][predicted] += 1;
                if predicted == label { test_correct += 1; }

                // Per-detector tracking
                if label < 10 {
                    detectors[label].eval_total += 1;
                    // Detector's own confidence for its digit
                    if scalar_confs[label] > scalar_confs.iter().enumerate()
                        .filter(|(i,_)| *i != label).map(|(_,&c)| c).max().unwrap_or(0)
                    {
                        detectors[label].eval_correct += 1;
                    }
                    detectors[label].confidence_history.push(scalar_confs[label]);
                    if detectors[label].confidence_history.len() > 100 {
                        detectors[label].confidence_history.remove(0);
                    }
                }

                // Sample for viz
                if i % 25 == 0 {
                    samples.push(SampleViz {
                        pixels: img.pixels.clone(),
                        label: img.label,
                        predicted: predicted as u8,
                        correct: predicted == label,
                        confidences: scalar_confs,
                    });
                }
            }

            let test_acc = test_correct as f32 / test_n as f32;
            let det_accs: Vec<f32> = detectors.iter()
                .map(|d| if d.eval_total > 0 { d.eval_correct as f32 / d.eval_total as f32 } else { 0.0 })
                .collect();

            let per_det_str = det_accs.iter().enumerate()
                .map(|(i, a)| format!("d{}:{:.0}%", i, a * 100.0))
                .collect::<Vec<_>>().join(" ");
            log_msg(state, format!("Eval {}/{} ({:.1}%) | {}", test_correct, test_n, test_acc * 100.0, per_det_str));

            let mut s = state.lock().unwrap();
            s.test_acc = test_acc;
            if test_acc > s.best_test_acc { s.best_test_acc = test_acc; }
            s.confusion = confusion;
            s.samples = samples;
            s.detectors = detectors.iter().map(|d| d.get_viz()).collect();
            let cy = s.cycle;
            s.history.push(Snap { cycle: cy, test_acc, det_accs });
        }

        "freeze" => {
            if let Some(d) = cmd.digit { if d < 10 { detectors[d].phase = Phase::Frozen; } }
            else { for d in detectors.iter_mut() { d.phase = Phase::Frozen; } }
            state.lock().unwrap().detectors = detectors.iter().map(|d| d.get_viz()).collect();
        }

        "flush" => {
            if let Some(d) = cmd.digit { if d < 10 { detectors[d].flush(); log_msg(state, format!("Flushed d{}", d)); } }
            state.lock().unwrap().detectors = detectors.iter().map(|d| d.get_viz()).collect();
        }

        "reset" => {
            // Full system reset — flush all detectors, rebuild finalizer, zero all state
            for d in detectors.iter_mut() { d.flush(); }
            *finalizer = Finalizer::new();
            let mut s = state.lock().unwrap();
            *s = ControlState::default();
            s.detectors = detectors.iter().map(|d| d.get_viz()).collect();
            drop(s);
            log_msg(state, "Reset: all detectors flushed, finalizer rebuilt, counters zeroed".into());
        }

        "train_isolation" => {
            let t0 = Instant::now();
            let per_digit = cmd.count.unwrap_or(200);
            let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
            for (i, img) in train.images.iter().enumerate() {
                let d = img.label as usize;
                if d < 10 && by_digit[d].len() < per_digit { by_digit[d].push(i); }
            }

            for digit in 0..10 {
                for &img_idx in &by_digit[digit] {
                    let encoded = encode_image(&train.images[img_idx].pixels);
                    detectors[digit].train_sample(&encoded, true);
                }
                detectors[digit].decay_pressure();
            }

            { state.lock().unwrap().cycle += 1; }
            log_msg(state, format!("Isolation: {} samples/det ({:.1}s)", per_digit, t0.elapsed().as_secs_f64()));
        }

        "train_adversarial" => {
            let t0 = Instant::now();
            let per_digit = cmd.count.unwrap_or(50);
            let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
            for (i, img) in train.images.iter().enumerate() {
                let d = img.label as usize;
                if d < 10 && by_digit[d].len() < per_digit { by_digit[d].push(i); }
            }

            for digit in 0..10 {
                for other in 0..10 {
                    if other == digit { continue; }
                    for &img_idx in &by_digit[other] {
                        let encoded = encode_image(&train.images[img_idx].pixels);
                        detectors[digit].train_sample(&encoded, false);
                    }
                }
                detectors[digit].decay_pressure();
            }

            { state.lock().unwrap().cycle += 1; }
            log_msg(state, format!("Adversarial: {}×9 samples/det ({:.1}s)", per_digit, t0.elapsed().as_secs_f64()));
        }

        "train_finalizer_only" => {
            let t0 = Instant::now();
            let per_digit = cmd.count.unwrap_or(200);
            let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
            for (i, img) in train.images.iter().enumerate() {
                let d = img.label as usize;
                if d < 10 && by_digit[d].len() < per_digit { by_digit[d].push(i); }
            }
            let mut train_set: Vec<(usize, u8)> = Vec::new();
            for d in 0..10 { for &idx in &by_digit[d] { train_set.push((idx, d as u8)); } }

            let mut correct = 0u32;
            for &(img_idx, label) in &train_set {
                let encoded = encode_image(&train.images[img_idx].pixels);
                let mut correlations: Vec<Value> = detectors.iter_mut()
                    .map(|d| d.correlate(&encoded)).collect();
                let confs = scalar_confidences(&correlations);
                inhibit_correlations(&mut correlations, &confs);
                normalize_correlations(&mut correlations);
                let combined = concat_correlations(&correlations);
                let pred = finalizer.train(&combined, label as usize);
                if pred == label as usize { correct += 1; }
            }
            finalizer.decay();

            { state.lock().unwrap().cycle += 1; }
            log_msg(state, format!("Finalizer: {}/{} ({:.1}%) ({:.1}s)",
                correct, train_set.len(), correct as f64 / train_set.len() as f64 * 100.0,
                t0.elapsed().as_secs_f64()));

            let eval_cmd = Cmd { action: "evaluate".into(), digit: None, count: Some(1000) };
            handle_cmd(&eval_cmd, detectors, finalizer, train, test, state);
        }

        _ => log_msg(state, format!("Unknown: {}", cmd.action)),
    }
}

// ---------------------------------------------------------------------------
// WebSocket state push
// ---------------------------------------------------------------------------

use std::net::TcpListener;
use tungstenite::{accept, Message};

fn refresh_state(
    detectors: &mut Vec<Detector>,
    finalizer: &Finalizer,
    state: &Arc<Mutex<ControlState>>,
) {
    for d in detectors.iter_mut() { d.decay_activations(); }
    let mut s = state.lock().unwrap();
    s.detectors = detectors.iter().map(|d| d.get_viz()).collect();
    s.finalizer = finalizer.get_viz();
}

fn push_state(
    ws: &mut tungstenite::WebSocket<std::net::TcpStream>,
    state: &Arc<Mutex<ControlState>>,
) -> bool {
    let s = state.lock().unwrap().clone();
    let json = serde_json::to_string(&s).unwrap_or_else(|_| "{}".into());
    ws.send(Message::Text(json.into())).is_ok()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let ws_port = std::env::args().nth(2).unwrap_or("8091".into());
    let http_port = std::env::args().nth(3).unwrap_or("8090".into());

    eprintln!("=== ANT-MNIST Control Room v3 (WebSocket) ===");
    eprintln!("  Architecture: 10 correlation detectors → finalizer ANT");
    eprintln!("Loading MNIST...");

    let train = Arc::new(MnistDataset::load(
        &PathBuf::from(&data_dir).join("train-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("train-labels-idx1-ubyte"),
    ).expect("load train"));
    let test = Arc::new(MnistDataset::load(
        &PathBuf::from(&data_dir).join("t10k-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("t10k-labels-idx1-ubyte"),
    ).expect("load test"));
    eprintln!("  train={}, test={}", train.len(), test.len());

    let state = Arc::new(Mutex::new(ControlState::default()));
    let mut detectors: Vec<Detector> = (0..10).map(Detector::new).collect();
    let mut finalizer = Finalizer::new();

    state.lock().unwrap().detectors = detectors.iter().map(|d| d.get_viz()).collect();

    // HTTP server for legacy HTML (background thread)
    let http_addr = format!("0.0.0.0:{}", http_port);
    thread::spawn(move || {
        let server = Server::http(&http_addr).unwrap();
        for request in server.incoming_requests() {
            let html = include_str!("viz_specialist.html");
            let _ = request.respond(Response::from_string(html)
                .with_header(Header::from_bytes("Content-Type", "text/html").unwrap()));
        }
    });

    // WebSocket server
    let ws_addr = format!("0.0.0.0:{}", ws_port);
    let listener = TcpListener::bind(&ws_addr).expect("failed to bind WS port");
    eprintln!("  WebSocket: ws://localhost:{}", ws_port);
    eprintln!("  HTTP:      http://localhost:{}", http_port);
    eprintln!("\n  Ready. 10 detectors COLD. Finalizer COLD.\n");
    eprintln!("  Waiting for dashboard connection...\n");

    // Accept connections in a loop (one client at a time)
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Set non-blocking for command reads during auto-train
        stream.set_nonblocking(false).ok();
        let mut ws = match accept(stream) {
            Ok(ws) => ws,
            Err(_) => continue,
        };

        eprintln!("  Client connected.");

        // Send initial state
        refresh_state(&mut detectors, &finalizer, &state);
        if !push_state(&mut ws, &state) { continue; }

        let mut curriculum: Option<Curriculum> = None;

        loop {
            // Read commands (non-blocking when curriculum is active)
            if curriculum.is_some() {
                ws.get_mut().set_nonblocking(true).ok();
            } else {
                ws.get_mut().set_nonblocking(false).ok();
            }

            match ws.read() {
                Ok(Message::Text(text)) => {
                    if let Ok(cmd) = serde_json::from_str::<Cmd>(&text) {
                        match cmd.action.as_str() {
                            "auto_train" => {
                                let total = cmd.count.unwrap_or(100) as u32;
                                if detectors[0].phase == Phase::Cold {
                                    let imprint_cmd = Cmd { action: "imprint".into(), digit: None, count: Some(50) };
                                    handle_cmd(&imprint_cmd, &mut detectors, &mut finalizer, &train, &test, &state);
                                }
                                let iso = (total * 40 / 100).max(3);
                                let adv = (total * 30 / 100).max(3);
                                let fin = total.saturating_sub(iso + adv).max(3);
                                curriculum = Some(Curriculum::new(iso, adv, fin));
                                log_msg(&state, format!("Auto-train: {} isolation → {} adversarial → {} finalizer",
                                    iso, adv, fin));
                            }
                            "stop" => {
                                curriculum = None;
                                log_msg(&state, "Stopped auto-train".into());
                            }
                            _ => {
                                handle_cmd(&cmd, &mut detectors, &mut finalizer, &train, &test, &state);
                            }
                        }
                        // Push state after every command
                        refresh_state(&mut detectors, &finalizer, &state);
                        if !push_state(&mut ws, &state) { break; }
                    }
                }
                Ok(Message::Close(_)) => {
                    eprintln!("  Client disconnected.");
                    break;
                }
                Err(tungstenite::Error::Io(ref e))
                    if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No command waiting — continue to curriculum step
                }
                Err(_) => {
                    eprintln!("  Client disconnected (error).");
                    break;
                }
                _ => {}
            }

            // Curriculum step
            if let Some(ref mut cur) = curriculum {
                if cur.remaining_in_phase == 0 {
                    if !cur.advance_phase() {
                        let eval_cmd = Cmd { action: "evaluate".into(), digit: None, count: Some(1000) };
                        handle_cmd(&eval_cmd, &mut detectors, &mut finalizer, &train, &test, &state);
                        log_msg(&state, "Auto-train: complete".into());
                        curriculum = None;
                    } else {
                        log_msg(&state, format!("→ {}", cur.phase_name()));
                    }
                }

                if let Some(ref mut cur) = curriculum {
                    let action = match cur.phase {
                        CurriculumPhase::Isolation => "train_isolation",
                        CurriculumPhase::Adversarial => "train_adversarial",
                        CurriculumPhase::Finalizer => "train_finalizer_only",
                    };
                    let count = match cur.phase {
                        CurriculumPhase::Isolation => 200,
                        CurriculumPhase::Adversarial => 50,
                        CurriculumPhase::Finalizer => 200,
                    };
                    let cmd = Cmd { action: action.into(), digit: None, count: Some(count) };
                    handle_cmd(&cmd, &mut detectors, &mut finalizer, &train, &test, &state);

                    cur.remaining_in_phase -= 1;
                    cur.step_in_phase += 1;

                    // Eval every 5 steps
                    if cur.step_in_phase % 5 == 0 {
                        let eval_cmd = Cmd { action: "evaluate".into(), digit: None, count: Some(1000) };
                        handle_cmd(&eval_cmd, &mut detectors, &mut finalizer, &train, &test, &state);
                    }

                    // Push state after every training step — real-time updates
                    refresh_state(&mut detectors, &finalizer, &state);
                    if !push_state(&mut ws, &state) { break; }
                }
            }
        }
        // Client disconnected — reset curriculum, wait for next connection
        curriculum = None;
        eprintln!("  Waiting for dashboard connection...\n");
    }
}
