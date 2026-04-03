//! ANT-MNIST Control Room v2
//!
//! Architecture: 10 spatial correlation detectors → finalizer ANT
//! Each detector returns 784 confidence signals.
//! Finalizer takes 7840 signals → 10 class outputs.

mod mnist;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

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
struct ControlState {
    detectors: Vec<DetViz>,
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
    template: Vec<i32>,     // 784 template current values (the imprint)
    eval_accuracy: f32,
    confidence_history: Vec<i64>,
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
               confidence_history: Vec::new() }
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
        }
    }

    /// Returns 784 correlation signals as Value::Array
    fn correlate(&mut self, input: &[Value]) -> Value {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        self.evaluator.invoke("correlate", &[input_val], &self.engine)
            .unwrap_or(Value::Array(Arc::new(vec![Value::Integer(0); 784])))
    }

    /// Returns scalar confidence
    fn confidence(&mut self, input: &[Value]) -> i64 {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        match self.evaluator.invoke("confidence", &[input_val], &self.engine) {
            Ok(Value::Integer(n)) => n,
            _ => 0,
        }
    }

    fn get_template(&self) -> Vec<i32> {
        let rt = self.runtime.lock().unwrap();
        let key = format!("d{}.template", self.digit);
        if let Some(twm) = rt.thermal_weights_by_key(&key) {
            twm.data.iter().take(784).map(|tw| tw.current()).collect()
        } else {
            vec![0; 784]
        }
    }

    fn get_viz(&self) -> DetViz {
        DetViz {
            digit: self.digit,
            phase: self.phase,
            template: self.get_template(),
            eval_accuracy: if self.eval_total > 0 { self.eval_correct as f32 / self.eval_total as f32 } else { 0.0 },
            confidence_history: self.confidence_history.clone(),
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

                let template = det.get_template();
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

                // Get 10 correlation maps
                let correlations: Vec<Value> = detectors.iter_mut()
                    .map(|d| d.correlate(&encoded))
                    .collect();
                let combined = concat_correlations(&correlations);

                // Train finalizer
                let pred = finalizer.train(&combined, label as usize);
                if pred == label as usize { correct += 1; }
            }
            finalizer.decay();

            {
                let mut s = state.lock().unwrap();
                s.cycle += 1;
            }
            log_msg(state, format!("Trained finalizer: {}/{} ({:.1}%) on {} samples ({}/digit)",
                correct, train_set.len(), correct as f64 / train_set.len() as f64 * 100.0,
                train_set.len(), per_digit));

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

                // Correlations
                let correlations: Vec<Value> = detectors.iter_mut()
                    .map(|d| d.correlate(&encoded))
                    .collect();

                // Scalar confidences for viz
                let scalar_confs: Vec<i64> = correlations.iter().map(|c| {
                    if let Value::Array(arr) = c {
                        arr.iter().map(|v| if let Value::Integer(n) = v { n.abs() } else { 0 }).sum::<i64>()
                    } else { 0 }
                }).collect();

                let combined = concat_correlations(&correlations);
                let predicted = finalizer.predict(&combined);

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

        _ => log_msg(state, format!("Unknown: {}", cmd.action)),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let port = std::env::args().nth(2).unwrap_or("8090".into());

    eprintln!("=== ANT-MNIST Control Room v2 ===");
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

    let addr = format!("0.0.0.0:{}", port);
    let server = Server::http(&addr).unwrap();
    eprintln!("  Dashboard: http://localhost:{}", port);
    eprintln!("\n  Ready. 10 detectors COLD. Finalizer COLD.\n");

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();
        let cors = Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap();
        let json_ct = Header::from_bytes("Content-Type", "application/json").unwrap();

        match (method, url.as_str()) {
            (_, "/") => {
                let html = include_str!("viz_specialist.html");
                let _ = request.respond(Response::from_string(html)
                    .with_header(Header::from_bytes("Content-Type", "text/html").unwrap()));
            }
            (_, "/api/state") => {
                let s = state.lock().unwrap().clone();
                let json = serde_json::to_string(&s).unwrap_or_else(|_| "{}".into());
                let _ = request.respond(Response::from_string(json).with_header(json_ct).with_header(cors));
            }
            (Method::Post, "/api/cmd") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).ok();
                let resp = if let Ok(cmd) = serde_json::from_str::<Cmd>(&body) {
                    handle_cmd(&cmd, &mut detectors, &mut finalizer, &train, &test, &state);
                    Response::from_string(r#"{"ok":true}"#).with_header(json_ct).with_header(cors)
                } else {
                    Response::from_string(r#"{"error":"bad json"}"#)
                        .with_header(json_ct).with_header(cors).with_status_code(400)
                };
                let _ = request.respond(resp);
            }
            _ => { let _ = request.respond(Response::from_string("404").with_status_code(404).with_header(cors)); }
        }
    }
}
