//! ant-mnist runtime — loads and executes a .rune program for MNIST classification.
//!
//! Usage: ant-mnist [rune_file] [data_dir]
//!   rune_file: path to .rune program (default: runes/mnist_classifier.rune)
//!   data_dir:  path to MNIST IDX files (default: data)
//!
//! Provides two module namespaces:
//!   :ant_ml  — matmul, relu, mastery, thermal, salience, neuromod, etc.
//!   :mnist   — load_train, load_test, get_sample, dataset_len
//!
//! The .rune file IS the program. Hot-reloadable without recompilation.

mod mnist;
mod mnist_module;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use atomic_neural_transistors::modules::ant_ml::{AntMlModule, AntRuntime};
use runes_core::engine::Engine;
use runes_parser::lexer::Lexer;
use runes_parser::parser::Parser;
use runes_eval::evaluator::Evaluator;

use mnist_module::{MnistModule, MnistRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rune_file = std::env::args().nth(1)
        .unwrap_or_else(|| "runes/mnist_classifier.rune".into());
    let data_dir = std::env::args().nth(2)
        .unwrap_or_else(|| "data".into());

    eprintln!("=== ANT-MNIST Runtime ===");
    eprintln!("  Rune: {}", rune_file);
    eprintln!("  Data: {}", data_dir);
    eprintln!();

    // Read the .rune source
    let source = std::fs::read_to_string(&rune_file)
        .map_err(|e| format!("Failed to read {}: {}", rune_file, e))?;

    // Build the engine with both modules
    let engine = Engine::builder()
        .namespace("default")
            .module(AntMlModule::new())
            .module(MnistModule::new())
        .build()
        .map_err(|e| format!("Engine build failed: {:?}", e))?;

    // Create and register both runtimes
    let ant_runtime = Arc::new(Mutex::new(AntRuntime::new()));
    let mnist_runtime = Arc::new(Mutex::new(MnistRuntime::new(PathBuf::from(&data_dir))));

    // Parse
    let tokens = Lexer::new(&source).tokenize()
        .map_err(|e| format!("Lex error: {:?}", e))?;
    let program = Parser::new(tokens).parse_program()
        .map_err(|e| format!("Parse error: {:?}", e))?;

    // Create evaluator and register both runtimes in the host registry
    let mut evaluator = Evaluator::new();
    evaluator.register_runtime(ant_runtime.clone());
    evaluator.register_runtime(mnist_runtime.clone());

    // Execute the program
    eprintln!("Running {}...\n", rune_file);
    match evaluator.eval_with_engine(&program, &engine) {
        Ok(_) => {
            eprintln!("\n=== Program completed ===");
        }
        Err(e) => {
            eprintln!("\n=== Runtime error ===");
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
