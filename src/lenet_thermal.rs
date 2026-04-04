//! LeNet-5 with ThermalWeight output layer from atomic-neural-transistors.
//!
//! Same frozen random conv/FC architecture as lenet_mastery.rs,
//! but the output layer uses full ThermalWeight (P×M×K + temperature +
//! pressure + hits + streak + confidence) instead of simple MasteryWeight.

mod mnist;

use std::path::PathBuf;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use mnist::MnistDataset;

use atomic_neural_transistors::core::thermal::{ThermalWeight, ThermalWeightMatrix, ThermalMasteryConfig};
use atomic_neural_transistors::thermal_mastery::ThermalMasteryState;
use ternary_signal::Signal;

// ---------------------------------------------------------------------------
// Reuse frozen layer types from lenet_mastery
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range_i16(&mut self, lo: i16, hi: i16) -> i16 {
        let range = (hi - lo + 1) as u64;
        (lo as i64 + (self.next_u64() % range) as i64) as i16
    }
}

#[derive(Clone)]
struct T1 { data: Vec<i16> }
#[derive(Clone)]
struct T2 { data: Vec<i16>, rows: usize, cols: usize }
#[derive(Clone)]
struct T3 { data: Vec<i16>, c: usize, h: usize, w: usize }
#[derive(Clone)]
struct T4 { data: Vec<i16>, n: usize, c: usize, h: usize, w: usize }

impl T1 {
    fn random(len: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..len).map(|_| rng.range_i16(-range, range)).collect() }
    }
}
impl T2 {
    fn random(rows: usize, cols: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..rows * cols).map(|_| rng.range_i16(-range, range)).collect(), rows, cols }
    }
    fn at(&self, r: usize, c: usize) -> i16 { self.data[r * self.cols + c] }
}
impl T3 {
    fn zeros(c: usize, h: usize, w: usize) -> Self { Self { data: vec![0; c * h * w], c, h, w } }
    fn at(&self, c: usize, y: usize, x: usize) -> i16 { self.data[c * self.h * self.w + y * self.w + x] }
    fn at_mut(&mut self, c: usize, y: usize, x: usize) -> &mut i16 { &mut self.data[c * self.h * self.w + y * self.w + x] }
}
impl T4 {
    fn random(n: usize, c: usize, h: usize, w: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..n * c * h * w).map(|_| rng.range_i16(-range, range)).collect(), n, c, h, w }
    }
    fn at(&self, n: usize, c: usize, y: usize, x: usize) -> i16 {
        self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x]
    }
}

const CONV_SHIFT: i32 = 5;
const FC_SHIFT: i32 = 6;

fn conv2d(input: &T3, kernel: &T4, bias: &T1) -> T3 {
    let oh = input.h - kernel.h + 1;
    let ow = input.w - kernel.w + 1;
    let mut out = T3::zeros(kernel.n, oh, ow);
    for f in 0..kernel.n {
        for y in 0..oh {
            for x in 0..ow {
                let mut s: i32 = (bias.data[f] as i32) << CONV_SHIFT;
                for c in 0..kernel.c { for ky in 0..kernel.h { for kx in 0..kernel.w {
                    s += input.at(c, y+ky, x+kx) as i32 * kernel.at(f, c, ky, kx) as i32;
                }}}
                *out.at_mut(f, y, x) = (s >> CONV_SHIFT).clamp(-32000, 32000) as i16;
            }
        }
    }
    out
}

fn relu(input: &T3) -> T3 {
    T3 { data: input.data.iter().map(|&v| v.max(0)).collect(), c: input.c, h: input.h, w: input.w }
}

fn avgpool(input: &T3) -> T3 {
    let oh = input.h / 2; let ow = input.w / 2;
    let mut out = T3::zeros(input.c, oh, ow);
    for c in 0..input.c { for y in 0..oh { for x in 0..ow {
        let s = input.at(c, y*2, x*2) as i32 + input.at(c, y*2+1, x*2) as i32
            + input.at(c, y*2, x*2+1) as i32 + input.at(c, y*2+1, x*2+1) as i32;
        *out.at_mut(c, y, x) = (s / 4) as i16;
    }}}
    out
}

fn fc(input: &[i16], w: &T2, bias: &T1) -> Vec<i16> {
    let mut out = vec![0i16; w.rows];
    for i in 0..w.rows {
        let mut s: i32 = (bias.data[i] as i32) << FC_SHIFT;
        for j in 0..w.cols { s += input[j] as i32 * w.at(i, j) as i32; }
        out[i] = (s >> FC_SHIFT).clamp(-32000, 32000) as i16;
    }
    out
}

fn relu_vec(v: &[i16]) -> Vec<i16> { v.iter().map(|&x| x.max(0)).collect() }

// ---------------------------------------------------------------------------
// Thermal LeNet
// ---------------------------------------------------------------------------

pub struct ThermalLeNet {
    conv1_k: T4, conv1_b: T1,
    conv2_k: T4, conv2_b: T1,
    fc1_w: T2, fc1_b: T1,
    fc2_w: T2, fc2_b: T1,
    // Thermal output layer
    output_weights: ThermalWeightMatrix,
    output_mastery: ThermalMasteryState,
}

impl ThermalLeNet {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            conv1_k: T4::random(6, 1, 5, 5, &mut rng, 25),
            conv1_b: T1::random(6, &mut rng, 5),
            conv2_k: T4::random(16, 6, 5, 5, &mut rng, 10),
            conv2_b: T1::random(16, &mut rng, 5),
            fc1_w: T2::random(120, 256, &mut rng, 5),
            fc1_b: T1::random(120, &mut rng, 5),
            fc2_w: T2::random(84, 120, &mut rng, 8),
            fc2_b: T1::random(84, &mut rng, 5),
            output_weights: ThermalWeightMatrix::zeros(10, 84),
            output_mastery: ThermalMasteryState::new(ThermalMasteryConfig {
                pressure_threshold: 8,
                decay_rate: 1,
                participation_gate: 5,
                cooling_rate: 50,
                warming_step: 10,
                warming_threshold: 20,
            }),
        }
    }

    fn frozen_features(&self, image: &[u8]) -> Vec<i16> {
        let mut input = T3::zeros(1, 28, 28);
        for y in 0..28 { for x in 0..28 {
            *input.at_mut(0, y, x) = (image[y * 28 + x] as i16) / 2;
        }}
        let x = relu(&conv2d(&input, &self.conv1_k, &self.conv1_b));
        let x = avgpool(&x);
        let x = relu(&conv2d(&x, &self.conv2_k, &self.conv2_b));
        let x = avgpool(&x);
        let x = relu_vec(&fc(&x.data, &self.fc1_w, &self.fc1_b));
        relu_vec(&fc(&x, &self.fc2_w, &self.fc2_b))
    }

    fn forward(&self, features: &[i16]) -> Vec<i16> {
        // Convert features to Signal for thermal matmul
        let signals: Vec<Signal> = features.iter()
            .map(|&v| Signal::from_current(v as i32))
            .collect();
        let output = self.output_weights.matmul(&signals);
        output.iter().map(|s| s.current().clamp(-32000, 32000) as i16).collect()
    }

    fn predict(&self, image: &[u8]) -> usize {
        let feats = self.frozen_features(image);
        let out = self.forward(&feats);
        out.iter().enumerate().max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap_or(0)
    }

    fn train_one(&mut self, image: &[u8], label: usize) -> (usize, bool) {
        let feats = self.frozen_features(image);
        let out = self.forward(&feats);
        let predicted = out.iter().enumerate().max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap_or(0);
        let correct = predicted == label;

        // Convert features to Signal for thermal mastery
        let input_signals: Vec<Signal> = feats.iter()
            .map(|&v| Signal::from_current(v as i32))
            .collect();

        // Selective update: only touch correct class + predicted class
        self.output_mastery.update_selective(
            &mut self.output_weights,
            &input_signals,
            label,
            predicted,
            correct,
        );

        (predicted, correct)
    }

    fn decay(&mut self) {
        self.output_mastery.decay(&mut self.output_weights);
    }

    fn thermal_stats(&self) -> [u32; 4] {
        let mut dist = [0u32; 4];
        for tw in &self.output_weights.data {
            let t = tw.temperature;
            if t >= 192 { dist[0] += 1; }
            else if t >= 128 { dist[1] += 1; }
            else if t >= 64 { dist[2] += 1; }
            else { dist[3] += 1; }
        }
        dist
    }

    fn nonzero(&self) -> usize {
        self.output_weights.data.iter().filter(|tw| tw.current() != 0).count()
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn peak_rss_mb() -> f64 {
    #[cfg(windows)]
    {
        use std::mem;
        #[repr(C)]
        struct Pmc { cb: u32, pf: u32, pws: usize, ws: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize }
        extern "system" { fn GetCurrentProcess() -> isize; fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32; }
        unsafe { let mut p: Pmc = mem::zeroed(); p.cb = mem::size_of::<Pmc>() as u32; K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb); p.pws as f64 / (1024.0 * 1024.0) }
    }
    #[cfg(not(windows))]
    { 0.0 }
}

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let n_voters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let epochs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    eprintln!("=== Thermal Ensemble LeNet-5: {} voters, {} epochs ===", n_voters, epochs);
    eprintln!("  ThermalWeight size: {} bytes", std::mem::size_of::<ThermalWeight>());
    eprintln!("Loading MNIST...");

    let train = MnistDataset::load(
        &PathBuf::from(&data_dir).join("train-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("train-labels-idx1-ubyte"),
    ).expect("load train");
    let test = MnistDataset::load(
        &PathBuf::from(&data_dir).join("t10k-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("t10k-labels-idx1-ubyte"),
    ).expect("load test");
    eprintln!("  train={}, test={}", train.len(), test.len());

    let seeds: Vec<u64> = (0..n_voters as u64).map(|i| 42 + i * 97).collect();
    let mut nets: Vec<ThermalLeNet> = seeds.iter().map(|&s| ThermalLeNet::new(s)).collect();

    let thermal_bytes = nets[0].output_weights.data.len() * std::mem::size_of::<ThermalWeight>();
    eprintln!("  thermal output: {} weights, {} bytes ({:.1} KB)",
        nets[0].output_weights.data.len(), thermal_bytes, thermal_bytes as f64 / 1024.0);

    println!("voters,epoch,test_acc,train_acc,epoch_s,rss_mb,nz,steps,transitions,hot,warm,cool,cold");

    for epoch in 1..=epochs {
        let t0 = Instant::now();

        // Train in parallel
        let results: Vec<u32> = nets.par_iter_mut().map(|net| {
            let mut correct = 0u32;
            for i in 0..train.len() {
                let img = &train.images[i];
                let (_, c) = net.train_one(&img.pixels, img.label as usize);
                if c { correct += 1; }
            }
            net.decay();
            correct
        }).collect();

        let train_acc = results.iter().sum::<u32>() as f64 / (train.len() * n_voters) as f64;
        let train_s = t0.elapsed().as_secs_f64();

        // Eval via majority vote
        let mut test_correct = 0u32;
        let mut pc = [0u32; 10]; let mut pt = [0u32; 10];
        for i in 0..test.len() {
            let img = &test.images[i];
            let label = img.label as usize;
            let mut votes = [0u32; 10];
            for net in &nets {
                let pred = net.predict(&img.pixels);
                if pred < 10 { votes[pred] += 1; }
            }
            let winner = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
            if label < 10 { pt[label] += 1; if winner == label { test_correct += 1; pc[label] += 1; } }
        }
        let test_acc = test_correct as f64 / test.len() as f64;
        let total_s = t0.elapsed().as_secs_f64();
        let rss = peak_rss_mb();

        // Aggregate thermal stats across all voters
        let mut total_nz = 0;
        let mut total_steps = 0u64;
        let mut total_trans = 0u64;
        let mut total_dist = [0u32; 4];
        for net in &nets {
            total_nz += net.nonzero();
            total_steps += net.output_mastery.steps;
            total_trans += net.output_mastery.transitions;
            let d = net.thermal_stats();
            for i in 0..4 { total_dist[i] += d[i]; }
        }

        println!("{},{},{:.4},{:.4},{:.1},{:.1},{},{},{},{},{},{},{}",
            n_voters, epoch, test_acc, train_acc, total_s, rss,
            total_nz, total_steps, total_trans,
            total_dist[0], total_dist[1], total_dist[2], total_dist[3]);

        let pca_short = (0..10).map(|d| {
            format!("d{}:{:.0}%", d, if pt[d] > 0 { pc[d] as f64 / pt[d] as f64 * 100.0 } else { 0.0 })
        }).collect::<Vec<_>>().join(" ");

        eprintln!("  E{:2} | train {:.1}% | test {:.1}% | {:.0}s | nz {} | s:{} t:{} | H/W/C/F {}/{}/{}/{} | {:.1}MB",
            epoch, train_acc * 100.0, test_acc * 100.0, total_s,
            total_nz, total_steps, total_trans,
            total_dist[0], total_dist[1], total_dist[2], total_dist[3], rss);
        eprintln!("      | {}", pca_short);
    }

    eprintln!("\n=== Thermal Ensemble complete ===");
}
