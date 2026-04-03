//! ant-mnist runtime — 10 specialist digit detectors running in parallel.
//!
//! Each detector is a separate AtomicNeuralTransistor with its own synaptic
//! state. Classification = argmax over 10 detector confidences.
//!
//! Usage: ant-mnist [data_dir] [cycles]

mod mnist;
mod mnist_module;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use atomic_neural_transistors::modules::ant_ml::{AntMlModule, AntRuntime};
use runes_core::engine::Engine;
use runes_core::value::Value;
use runes_parser::lexer::Lexer;
use runes_parser::parser::Parser;
use runes_eval::evaluator::Evaluator;
use ternary_signal::Signal;

use mnist::MnistDataset;
use mnist_module::{MnistModule, MnistRuntime};

const DETECTOR_SOURCE: &str = include_str!("../runes/digit_detector.rune");

struct Detector {
    evaluator: Evaluator,
    engine: Engine,
    digit: usize,
}

impl Detector {
    fn new(digit: usize, seed: u64) -> Self {
        let engine = Engine::builder()
            .namespace("default")
                .module(AntMlModule::new())
            .build()
            .expect("engine build failed");

        let tokens = Lexer::new(DETECTOR_SOURCE).tokenize().expect("lex failed");
        let program = Parser::new(tokens).parse_program().expect("parse failed");

        let ant_runtime = Arc::new(Mutex::new(AntRuntime::new()));
        let mut evaluator = Evaluator::new();
        evaluator.set_host(ant_runtime.clone());

        // Set the prefix variable so the rune knows its synaptic key namespace
        let prefix = format!("d{}", digit);
        evaluator.define_variable("prefix", Value::String(prefix.into()));

        // Set the random seed for the frozen projection — unique per detector
        // The load_synaptic_frozen verb uses a default seed, but the prefix
        // ensures each detector gets different keys → different matrices.

        evaluator.eval_with_engine(&program, &engine).expect("eval failed");

        Self { evaluator, engine, digit }
    }

    fn forward(&mut self, input: &[Value]) -> i64 {
        let input_val = Value::Array(Arc::new(input.to_vec()));
        let result = self.evaluator.invoke("forward", &[input_val], &self.engine)
            .expect("forward failed");
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
            .expect("train failed");
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

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or_else(|| "data".into());
    let n_cycles: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    eprintln!("=== ANT-MNIST: 10 Specialist Detectors ===\n");

    // Load data
    let train = MnistDataset::load(
        &PathBuf::from(&data_dir).join("train-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("train-labels-idx1-ubyte"),
    ).expect("failed to load training set");
    let test = MnistDataset::load(
        &PathBuf::from(&data_dir).join("t10k-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("t10k-labels-idx1-ubyte"),
    ).expect("failed to load test set");
    eprintln!("Loaded: {} train, {} test\n", train.len(), test.len());

    // Create 10 detectors with different seeds
    let mut detectors: Vec<Detector> = (0..10)
        .map(|d| Detector::new(d, 42 + d as u64 * 7919))
        .collect();
    eprintln!("10 detectors initialized\n");

    // Baseline
    let mut baseline_correct = 0;
    let test_subset = 1000.min(test.len());
    for i in 0..test_subset {
        let encoded = encode_image(&test.images[i].pixels);
        let confidences: Vec<i64> = detectors.iter_mut()
            .map(|d| d.forward(&encoded))
            .collect();
        let predicted = confidences.iter().enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        if predicted == test.images[i].label as usize {
            baseline_correct += 1;
        }
    }
    eprintln!("Baseline: {}/{} ({:.1}%)\n",
        baseline_correct, test_subset,
        baseline_correct as f64 / test_subset as f64 * 100.0);

    // Training
    for cycle in 1..=n_cycles {
        let cycle_start = Instant::now();
        let mut train_correct = 0;

        for i in 0..train.len() {
            let encoded = encode_image(&train.images[i].pixels);
            let label = train.images[i].label as usize;

            // Each detector trains on whether this IS its digit
            let mut confidences = Vec::with_capacity(10);
            for d in 0..10 {
                let confidence = detectors[d].train(&encoded, label == d);
                confidences.push(confidence);
            }

            let predicted = confidences.iter().enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == label {
                train_correct += 1;
            }
        }

        // Decay all detectors
        for d in &mut detectors {
            d.decay();
        }

        // Evaluate
        let mut test_correct = 0;
        for i in 0..test_subset {
            let encoded = encode_image(&test.images[i].pixels);
            let confidences: Vec<i64> = detectors.iter_mut()
                .map(|d| d.forward(&encoded))
                .collect();
            let predicted = confidences.iter().enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == test.images[i].label as usize {
                test_correct += 1;
            }
        }

        let elapsed = cycle_start.elapsed();
        eprintln!("Cycle {:2}: train {}/{} ({:.1}%), test {}/{} ({:.1}%) [{:.1}s]",
            cycle,
            train_correct, train.len(),
            train_correct as f64 / train.len() as f64 * 100.0,
            test_correct, test_subset,
            test_correct as f64 / test_subset as f64 * 100.0,
            elapsed.as_secs_f64());
    }

    eprintln!("\n=== Done ===");
}
