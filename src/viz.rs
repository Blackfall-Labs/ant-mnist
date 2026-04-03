//! ANT-MNIST Specialist Visualizer
//!
//! 10 binary digit detectors with live web dashboard.
//! http://localhost:8090
//!
//! Shows per-detector: weights, pressure, thermals, activations, accuracy.
//! Shows consensus: all 10 outputs for each test image.

mod mnist;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use serde::Serialize;
use tiny_http::{Server, Response, Header};

use atomic_neural_transistors::modules::ant_ml::{AntMlModule, AntRuntime};
use runes_core::engine::Engine;
use runes_core::value::Value;
use runes_parser::lexer::Lexer;
use runes_parser::parser::Parser;
use runes_eval::evaluator::Evaluator;
use ternary_signal::Signal;

use mnist::MnistDataset;

const DETECTOR_SOURCE: &str = include_str!("../runes/digit_detector.rune");

// ---------------------------------------------------------------------------
// Viz state structures
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Default)]
struct VizState {
    cycle: u32,
    step: usize,
    total_steps: usize,
    running: bool,
    // Per-detector stats
    detectors: Vec<DetectorState>,
    // Consensus predictions on test samples
    samples: Vec<SamplePrediction>,
    // Training history
    history: Vec<CycleSnapshot>,
    // Overall accuracy
    test_acc: f32,
    best_test_acc: f32,
}

#[derive(Serialize, Clone, Default)]
struct DetectorState {
    digit: usize,
    // Output weights: 48 values (1×48 matrix)
    weight_currents: Vec<i32>,
    weight_temps: Vec<u8>,
    weight_pressures: Vec<i16>,
    weight_streaks: Vec<u16>,
    weight_confidences: Vec<u16>,
    // Hidden activations for last sample (48 values)
    hidden_activations: Vec<i32>,
    // Rolling accuracy (last 100 samples)
    rolling_correct: u32,
    rolling_total: u32,
    // Output confidence for last sample
    last_confidence: i64,
    last_was_match: bool,
}

#[derive(Serialize, Clone)]
struct SamplePrediction {
    pixels: Vec<u8>,
    label: u8,
    predicted: u8,
    correct: bool,
    confidences: Vec<i64>, // all 10 detector outputs
}

#[derive(Serialize, Clone)]
struct CycleSnapshot {
    cycle: u32,
    test_acc: f32,
    per_detector_acc: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Detector wrapper with introspection
// ---------------------------------------------------------------------------

struct Detector {
    evaluator: Evaluator,
    engine: Engine,
    digit: usize,
    runtime: Arc<Mutex<AntRuntime>>,
}

impl Detector {
    fn new(digit: usize) -> Self {
        let engine = Engine::builder()
            .namespace("default")
                .module(AntMlModule::new())
            .build()
            .expect("engine build failed");

        let tokens = Lexer::new(DETECTOR_SOURCE).tokenize().expect("lex failed");
        let program = Parser::new(tokens).parse_program().expect("parse failed");

        let runtime = Arc::new(Mutex::new(AntRuntime::new()));
        let mut evaluator = Evaluator::new();
        evaluator.set_host(runtime.clone());
        evaluator.define_variable("prefix", Value::String(format!("d{}", digit).into()));
        evaluator.eval_with_engine(&program, &engine).expect("eval failed");

        Self { evaluator, engine, digit, runtime }
    }

    fn forward(&mut self, input: &[Value]) -> i64 {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        let result = self.evaluator.invoke("forward", &[input_val], &self.engine)
            .unwrap_or(Value::Integer(0));
        match result {
            Value::Array(arr) => {
                if let Some(Value::Integer(n)) = arr.first() { *n } else { 0 }
            }
            Value::Integer(n) => n,
            _ => 0,
        }
    }

    fn train(&mut self, input: &[Value], is_match: bool) -> i64 {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        let match_val = Value::Integer(if is_match { 1 } else { 0 });
        let result = self.evaluator.invoke("train", &[input_val, match_val], &self.engine)
            .unwrap_or(Value::Integer(0));
        match result {
            Value::Array(arr) => {
                if let Some(Value::Integer(n)) = arr.first() { *n } else { 0 }
            }
            Value::Integer(n) => n,
            _ => 0,
        }
    }

    fn decay(&mut self) {
        self.evaluator.invoke("decay", &[], &self.engine).ok();
    }

    fn get_state(&self) -> DetectorState {
        let rt = self.runtime.lock().unwrap();
        let prefix = format!("d{}", self.digit);
        let out_key = format!("{}.w_out", prefix);

        let mut ds = DetectorState {
            digit: self.digit,
            ..Default::default()
        };

        if let Some(wm) = rt.weights_by_key(&out_key) {
            for s in &wm.data {
                ds.weight_currents.push(s.current());
            }
        }

        // Get mastery state if available
        if let Some(&handle) = rt.synaptic_key_handles().get(&out_key) {
            if let Some(ms) = rt.get_mastery(handle) {
                ds.weight_pressures = ms.pressure.iter().map(|&p| p as i16).collect();
            }
        }

        ds
    }
}

fn encode_pixel(px: u8) -> i64 {
    if px <= 84 {
        -(((84 - px as i32) * 60) / 84) as i64
    } else if px >= 171 {
        (((px as i32 - 171) * 60) / 84) as i64
    } else {
        0
    }
}

fn encode_image(pixels: &[u8]) -> Vec<Value> {
    pixels.iter().map(|&px| Value::Integer(encode_pixel(px))).collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let port = std::env::args().nth(2).unwrap_or("8090".into());

    eprintln!("=== ANT-MNIST Specialist Visualizer ===");
    eprintln!("Loading MNIST...");

    let train = MnistDataset::load(
        &PathBuf::from(&data_dir).join("train-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("train-labels-idx1-ubyte"),
    ).expect("failed to load training set");
    let test = MnistDataset::load(
        &PathBuf::from(&data_dir).join("t10k-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("t10k-labels-idx1-ubyte"),
    ).expect("failed to load test set");
    eprintln!("  train={}, test={}", train.len(), test.len());

    let state = Arc::new(Mutex::new(VizState::default()));

    // HTTP server
    let addr = format!("0.0.0.0:{}", port);
    let server = Server::http(&addr).unwrap();
    eprintln!("  Dashboard: http://localhost:{}", port);

    let state_http = state.clone();

    // Serve HTTP
    let _http = thread::spawn(move || {
        for request in server.incoming_requests() {
            let url = request.url().to_string();
            let cors = Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap();
            let json_ct = Header::from_bytes("Content-Type", "application/json").unwrap();

            match url.as_str() {
                "/" => {
                    let html = include_str!("viz_specialist.html");
                    let resp = Response::from_string(html)
                        .with_header(Header::from_bytes("Content-Type", "text/html").unwrap());
                    let _ = request.respond(resp);
                }
                "/api/state" => {
                    let s = state_http.lock().unwrap().clone();
                    let json = serde_json::to_string(&s).unwrap_or_else(|_| "{}".into());
                    let resp = Response::from_string(json).with_header(json_ct).with_header(cors);
                    let _ = request.respond(resp);
                }
                "/api/start" => {
                    state_http.lock().unwrap().running = true;
                    let resp = Response::from_string(r#"{"ok":true}"#).with_header(json_ct).with_header(cors);
                    let _ = request.respond(resp);
                }
                "/api/stop" => {
                    state_http.lock().unwrap().running = false;
                    let resp = Response::from_string(r#"{"ok":true}"#).with_header(json_ct).with_header(cors);
                    let _ = request.respond(resp);
                }
                _ => {
                    let resp = Response::from_string("404").with_status_code(404);
                    let _ = request.respond(resp);
                }
            }
        }
    });

    // Create detectors
    let mut detectors: Vec<Detector> = (0..10).map(|d| Detector::new(d)).collect();
    eprintln!("  10 detectors initialized\n");
    eprintln!("  Press Start in the dashboard to begin training.");

    // Initialize state with detector info
    {
        let mut s = state.lock().unwrap();
        s.detectors = detectors.iter().map(|d| d.get_state()).collect();
        s.total_steps = train.len();
    }

    // Generate initial sample predictions
    {
        let mut samples = Vec::new();
        for i in (0..100).step_by(5) {
            let img = &test.images[i];
            let encoded = encode_image(&img.pixels);
            let confidences: Vec<i64> = detectors.iter_mut().map(|d| d.forward(&encoded)).collect();
            let predicted = confidences.iter().enumerate()
                .max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
            samples.push(SamplePrediction {
                pixels: img.pixels.clone(),
                label: img.label,
                predicted: predicted as u8,
                correct: predicted == img.label as usize,
                confidences,
            });
        }
        state.lock().unwrap().samples = samples;
    }

    // Training loop
    let test_subset = 1000.min(test.len());
    loop {
        // Wait for start signal
        {
            let s = state.lock().unwrap();
            if !s.running {
                drop(s);
                thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        }

        let cycle = {
            let mut s = state.lock().unwrap();
            s.cycle += 1;
            s.cycle
        };

        let cycle_start = Instant::now();
        let mut train_correct = 0;

        // Train
        for i in 0..train.len() {
            let encoded = encode_image(&train.images[i].pixels);
            let label = train.images[i].label as usize;

            let mut confidences = Vec::with_capacity(10);
            for d in 0..10 {
                let c = detectors[d].train(&encoded, label == d);
                confidences.push(c);
            }

            let predicted = confidences.iter().enumerate()
                .max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
            if predicted == label {
                train_correct += 1;
            }

            // Update viz state every 1000 steps
            if i % 1000 == 0 {
                let mut s = state.lock().unwrap();
                s.step = i;
                if !s.running { break; }
            }
        }

        // Decay
        for d in &mut detectors {
            d.decay();
        }

        // Evaluate
        let mut test_correct = 0;
        for i in 0..test_subset {
            let encoded = encode_image(&test.images[i].pixels);
            let confidences: Vec<i64> = detectors.iter_mut().map(|d| d.forward(&encoded)).collect();
            let predicted = confidences.iter().enumerate()
                .max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
            if predicted == test.images[i].label as usize {
                test_correct += 1;
            }
        }

        let test_acc = test_correct as f32 / test_subset as f32;
        let elapsed = cycle_start.elapsed();

        eprintln!("Cycle {:2}: train {}/{} ({:.1}%), test {}/{} ({:.1}%) [{:.1}s]",
            cycle,
            train_correct, train.len(),
            train_correct as f64 / train.len() as f64 * 100.0,
            test_correct, test_subset,
            test_acc * 100.0,
            elapsed.as_secs_f64());

        // Update viz state
        {
            let mut s = state.lock().unwrap();
            s.test_acc = test_acc;
            if test_acc > s.best_test_acc { s.best_test_acc = test_acc; }
            s.step = train.len();

            // Update detector states
            s.detectors = detectors.iter().map(|d| d.get_state()).collect();

            // Update sample predictions
            let mut samples = Vec::new();
            for i in (0..100).step_by(5) {
                let img = &test.images[i];
                let encoded = encode_image(&img.pixels);
                let confidences: Vec<i64> = detectors.iter_mut().map(|d| d.forward(&encoded)).collect();
                let predicted = confidences.iter().enumerate()
                    .max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0);
                samples.push(SamplePrediction {
                    pixels: img.pixels.clone(),
                    label: img.label,
                    predicted: predicted as u8,
                    correct: predicted == img.label as usize,
                    confidences,
                });
            }
            s.samples = samples;

            // History
            s.history.push(CycleSnapshot {
                cycle,
                test_acc,
                per_detector_acc: vec![], // TODO: per-detector accuracy
            });
        }
    }
}
