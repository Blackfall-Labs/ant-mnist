//! MNIST Runes module — provides dataset loading and encoding verbs.
//!
//! Verbs:
//!   load_train(data_dir)      → Handle (training dataset)
//!   load_test(data_dir)       → Handle (test dataset)
//!   dataset_len(handle)       → Integer (number of images)
//!   get_sample(handle, index) → Array [encoded_signal(784)..., label]
//!   encode_pixel(byte)        → Integer (signal current value)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use runes_core::error::RuneError;
use runes_core::traits::{EvalContext, Module, ModuleVersion, Verb, VerbResult};
use runes_core::value::Value;
use ternary_signal::Signal;

use crate::mnist::{MnistDataset, MnistImage};

const HANDLE_DATASET: u32 = 10;

/// MNIST runtime state — holds loaded datasets.
pub struct MnistRuntime {
    datasets: HashMap<u64, MnistDataset>,
    next_handle: u64,
    pub data_dir: PathBuf,
}

impl MnistRuntime {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            datasets: HashMap::new(),
            next_handle: 0,
            data_dir,
        }
    }

    fn alloc_handle(&mut self) -> u64 {
        let id = self.next_handle;
        self.next_handle += 1;
        id
    }
}

/// Encode a single pixel (0-255) to a Signal.
///
/// 3-band ternary encoding:
///   0-84:   Inhibitory (dark/background)
///   85-170: Silent (ambiguous mid-grey)
///   171-255: Excitatory (bright/stroke)
fn encode_pixel(px: u8) -> Signal {
    if px <= 84 {
        let strength = ((84 - px as i32) * 60) / 84;
        Signal::from_current(-strength)
    } else if px >= 171 {
        let strength = ((px as i32 - 171) * 60) / 84;
        Signal::from_current(strength)
    } else {
        Signal::ZERO
    }
}

/// Encode a full 784-pixel image to signal current values.
fn encode_image_to_values(pixels: &[u8]) -> Vec<Value> {
    pixels.iter()
        .map(|&px| Value::Integer(encode_pixel(px).current() as i64))
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: extract MNIST runtime from EvalContext
// ---------------------------------------------------------------------------

fn with_mnist<F, R>(ctx: &mut EvalContext, f: F) -> Result<R, RuneError>
where
    F: FnOnce(&mut MnistRuntime) -> Result<R, RuneError>,
{
    let span = ctx.span;
    let arc = ctx.runtime_mut::<Arc<std::sync::Mutex<MnistRuntime>>>()?;
    let mut guard = arc.lock()
        .map_err(|_| RuneError::type_error("MNIST runtime lock poisoned", Some(span)))?;
    f(&mut guard)
}

fn require_int(args: &[Value], idx: usize, name: &str, span: runes_core::Span) -> Result<i64, RuneError> {
    match args.get(idx) {
        Some(Value::Integer(n)) => Ok(*n),
        _ => Err(RuneError::argument(format!("{name} requires integer at position {idx}"), Some(span))),
    }
}

fn require_str(args: &[Value], idx: usize, name: &str, span: runes_core::Span) -> Result<String, RuneError> {
    match args.get(idx) {
        Some(Value::String(s)) => Ok(s.to_string()),
        _ => Err(RuneError::argument(format!("{name} requires string at position {idx}"), Some(span))),
    }
}

fn require_handle(args: &[Value], idx: usize, name: &str, expected_type: u32, span: runes_core::Span) -> Result<u64, RuneError> {
    match args.get(idx) {
        Some(Value::Handle(ht, id)) if ht.0 == expected_type => Ok(*id),
        _ => Err(RuneError::argument(format!("{name} requires dataset handle at position {idx}"), Some(span))),
    }
}

// ---------------------------------------------------------------------------
// Verb implementations
// ---------------------------------------------------------------------------

/// load_train() → Handle
/// Loads MNIST training dataset from the configured data directory.
struct LoadTrainVerb;
impl Verb for LoadTrainVerb {
    fn name(&self) -> &str { "load_train" }
    fn call(&self, _args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        with_mnist(ctx, |rt| {
            let img_path = rt.data_dir.join("train-images-idx3-ubyte");
            let lbl_path = rt.data_dir.join("train-labels-idx1-ubyte");
            let dataset = MnistDataset::load(&img_path, &lbl_path)
                .map_err(|e| RuneError::argument(format!("load_train: {e}"), Some(span)))?;
            let id = rt.alloc_handle();
            eprintln!("Loaded training set: {} images", dataset.len());
            rt.datasets.insert(id, dataset);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_DATASET), id))
        })
    }
}

/// load_test() → Handle
/// Loads MNIST test dataset from the configured data directory.
struct LoadTestVerb;
impl Verb for LoadTestVerb {
    fn name(&self) -> &str { "load_test" }
    fn call(&self, _args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        with_mnist(ctx, |rt| {
            let img_path = rt.data_dir.join("t10k-images-idx3-ubyte");
            let lbl_path = rt.data_dir.join("t10k-labels-idx1-ubyte");
            let dataset = MnistDataset::load(&img_path, &lbl_path)
                .map_err(|e| RuneError::argument(format!("load_test: {e}"), Some(span)))?;
            let id = rt.alloc_handle();
            eprintln!("Loaded test set: {} images", dataset.len());
            rt.datasets.insert(id, dataset);
            Ok(Value::Handle(runes_core::value::HandleType(HANDLE_DATASET), id))
        })
    }
}

/// dataset_len(handle) → Integer
struct DatasetLenVerb;
impl Verb for DatasetLenVerb {
    fn name(&self) -> &str { "dataset_len" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "dataset_len", HANDLE_DATASET, span)?;
        with_mnist(ctx, |rt| {
            let ds = rt.datasets.get(&handle)
                .ok_or_else(|| RuneError::argument("dataset_len: invalid handle", Some(span)))?;
            Ok(Value::Integer(ds.len() as i64))
        })
    }
}

/// get_sample(handle, index) → Array [signal_values(784)..., label]
/// Returns a flat array of 785 integers: 784 encoded signal current values + 1 label.
struct GetSampleVerb;
impl Verb for GetSampleVerb {
    fn name(&self) -> &str { "get_sample" }
    fn call(&self, args: &[Value], ctx: &mut EvalContext) -> VerbResult {
        let span = ctx.span;
        let handle = require_handle(args, 0, "get_sample", HANDLE_DATASET, span)?;
        let index = require_int(args, 1, "get_sample", span)? as usize;
        with_mnist(ctx, |rt| {
            let ds = rt.datasets.get(&handle)
                .ok_or_else(|| RuneError::argument("get_sample: invalid handle", Some(span)))?;
            if index >= ds.len() {
                return Err(RuneError::argument(
                    format!("get_sample: index {} out of bounds (len={})", index, ds.len()),
                    Some(span),
                ));
            }
            let image = &ds.images[index];
            let mut values = encode_image_to_values(&image.pixels);
            values.push(Value::Integer(image.label as i64));
            Ok(Value::Array(Arc::new(values)))
        })
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

/// print(value) → Nil
/// Print a value to stderr (eprintln).
struct PrintVerb;
impl Verb for PrintVerb {
    fn name(&self) -> &str { "print" }
    fn call(&self, args: &[Value], _ctx: &mut EvalContext) -> VerbResult {
        let mut parts: Vec<String> = Vec::new();
        for arg in args {
            match arg {
                Value::String(s) => parts.push(s.to_string()),
                Value::Integer(n) => parts.push(n.to_string()),
                Value::Float(f) => parts.push(f.to_string()),
                Value::Bool(b) => parts.push(b.to_string()),
                Value::Nil => parts.push("nil".to_string()),
                Value::Array(arr) => {
                    let items: Vec<String> = arr.iter().map(|v| format!("{:?}", v)).collect();
                    parts.push(format!("[{}]", items.join(", ")));
                }
                other => parts.push(format!("{:?}", other)),
            }
        }
        eprintln!("{}", parts.join(""));
        Ok(Value::Nil)
    }
}

pub struct MnistModule {
    verbs: Vec<Box<dyn Verb>>,
}

impl MnistModule {
    pub fn new() -> Self {
        Self {
            verbs: vec![
                Box::new(LoadTrainVerb),
                Box::new(LoadTestVerb),
                Box::new(DatasetLenVerb),
                Box::new(GetSampleVerb),
                Box::new(PrintVerb),
            ],
        }
    }
}

impl Module for MnistModule {
    fn name(&self) -> &str { "mnist" }
    fn version(&self) -> ModuleVersion { ModuleVersion::new(0, 1, 0) }
    fn verbs(&self) -> &[Box<dyn Verb>] { &self.verbs }
}
