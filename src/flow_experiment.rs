//! Flow weight experiment runner — Experiment A: single-layer 784→10.
//!
//! Outputs CSV metrics to stdout, human-readable progress to stderr.
//! Usage: cargo run --release --bin flow-experiment [data_dir]

mod mnist;
mod flow;

use std::path::PathBuf;
use std::time::Instant;
use mnist::MnistDataset;
use flow::{FlowClassifier, FlowWeight};

fn encode_pixel(px: u8) -> i16 {
    if px <= 84 { -(((84 - px as i32) * 60) / 84) as i16 }
    else if px >= 171 { (((px as i32 - 171) * 60) / 84) as i16 }
    else { 0 }
}

fn encode_image(pixels: &[u8]) -> Vec<i16> {
    pixels.iter().map(|&px| encode_pixel(px)).collect()
}

struct EvalMetrics {
    test_acc: f64,
    per_class_acc: [f64; 10],
    confusion: [[u32; 10]; 10],
    collapse_score: f64,
}

fn evaluate(clf: &flow::FlowClassifier, test: &MnistDataset, count: usize) -> EvalMetrics {
    let n = count.min(test.len());
    let mut correct = 0u32;
    let mut per_class_correct = [0u32; 10];
    let mut per_class_total = [0u32; 10];
    let mut confusion = [[0u32; 10]; 10];

    for i in 0..n {
        let img = &test.images[i];
        let encoded = encode_image(&img.pixels);
        let label = img.label as usize;
        let predicted = clf.predict(&encoded);

        if label < 10 {
            per_class_total[label] += 1;
            if predicted == label {
                correct += 1;
                per_class_correct[label] += 1;
            }
            if predicted < 10 {
                confusion[label][predicted] += 1;
            }
        }
    }

    let test_acc = correct as f64 / n as f64;
    let mut per_class_acc = [0.0f64; 10];
    for d in 0..10 {
        per_class_acc[d] = if per_class_total[d] > 0 {
            per_class_correct[d] as f64 / per_class_total[d] as f64
        } else { 0.0 };
    }

    let max_acc = per_class_acc.iter().cloned().fold(0.0f64, f64::max);
    let mean_acc = per_class_acc.iter().sum::<f64>() / 10.0;
    let collapse_score = max_acc - mean_acc;

    EvalMetrics { test_acc, per_class_acc, confusion, collapse_score }
}

fn weight_stats(clf: &flow::FlowClassifier) -> (f64, f64, f64, [u32; 4], usize) {
    let weights = &clf.weights.data;
    let n = weights.len() as f64;

    // Magnitude stats
    let mag_sum: f64 = weights.iter().map(|w| w.value.unsigned_abs() as f64).sum();
    let mag_mean = mag_sum / n;
    let mag_var: f64 = weights.iter()
        .map(|w| { let d = w.value.unsigned_abs() as f64 - mag_mean; d * d })
        .sum::<f64>() / n;
    let mag_std = mag_var.sqrt();

    // Momentum mean
    let mom_mean: f64 = weights.iter()
        .map(|w| w.momentum.unsigned_abs() as f64).sum::<f64>() / n;

    // Temperature distribution
    let mut temp_dist = [0u32; 4]; // HOT, WARM, COOL, COLD
    for w in weights {
        let t = w.temperature();
        if t >= 192 { temp_dist[0] += 1; }
        else if t >= 128 { temp_dist[1] += 1; }
        else if t >= 64 { temp_dist[2] += 1; }
        else { temp_dist[3] += 1; }
    }

    let nonzero = weights.iter().filter(|w| w.value != 0).count();

    (mag_mean, mag_std, mom_mean, temp_dist, nonzero)
}

fn peak_rss_mb() -> f64 {
    // Windows: use GetProcessMemoryInfo
    #[cfg(windows)]
    {
        use std::mem;
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(process: isize, ppmc: *mut ProcessMemoryCounters, cb: u32) -> i32;
        }
        unsafe {
            let mut pmc: ProcessMemoryCounters = mem::zeroed();
            pmc.cb = mem::size_of::<ProcessMemoryCounters>() as u32;
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb);
            pmc.peak_working_set_size as f64 / (1024.0 * 1024.0)
        }
    }
    #[cfg(not(windows))]
    { 0.0 }
}

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());

    eprintln!("=== Flow Weight Experiment A: Single-Layer 784→10 ===");
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

    let mut clf = FlowClassifier::new();
    let weight_count = clf.weights.data.len();
    let weight_bytes = weight_count * std::mem::size_of::<FlowWeight>();

    eprintln!("  weights: {} ({} bytes, {:.1} KB)", weight_count, weight_bytes, weight_bytes as f64 / 1024.0);
    eprintln!("  FlowWeight size: {} bytes", std::mem::size_of::<FlowWeight>());
    eprintln!();

    // CSV header
    println!("experiment,cycle,test_acc,per_class_acc,collapse_score,weight_mag_mean,weight_mag_std,momentum_mean,temp_hot,temp_warm,temp_cool,temp_cold,nonzero,weight_count,weight_bytes,peak_rss_mb,cycle_ms");

    let total_cycles = 100;
    let per_digit = 200;
    let eval_every = 5;

    for cycle in 1..=total_cycles {
        let t0 = Instant::now();

        // Build balanced training set
        let mut by_digit: Vec<Vec<usize>> = vec![Vec::new(); 10];
        // Use different samples each cycle by offsetting
        let offset = (cycle - 1) * per_digit;
        for (i, img) in train.images.iter().enumerate() {
            let d = img.label as usize;
            if d < 10 && by_digit[d].len() < per_digit {
                // Skip first `offset` samples of this digit for variety
                let digit_count = train.images.iter().take(i + 1)
                    .filter(|im| im.label as usize == d).count();
                if digit_count > offset % 5000 {
                    by_digit[d].push(i);
                }
            }
        }

        let mut train_set: Vec<(usize, u8)> = Vec::new();
        for d in 0..10 {
            for &idx in &by_digit[d] {
                train_set.push((idx, d as u8));
            }
        }

        // Train
        let mut correct = 0u32;
        for &(img_idx, label) in &train_set {
            let encoded = encode_image(&train.images[img_idx].pixels);
            let (_, c) = clf.train(&encoded, label as usize);
            if c { correct += 1; }
        }
        clf.coast();

        let cycle_ms = t0.elapsed().as_millis();
        let train_acc = correct as f64 / train_set.len() as f64;

        // Eval every N cycles
        if cycle % eval_every == 0 || cycle == 1 {
            let metrics = evaluate(&clf, &test, 1000);
            let (mag_mean, mag_std, mom_mean, temp_dist, nonzero) = weight_stats(&clf);
            let rss = peak_rss_mb();

            let pca = metrics.per_class_acc.iter()
                .map(|a| format!("{:.3}", a))
                .collect::<Vec<_>>().join("|");

            println!("A,{},{:.4},{},{:.3},{:.1},{:.1},{:.1},{},{},{},{},{},{},{},{:.1},{}",
                cycle, metrics.test_acc, pca, metrics.collapse_score,
                mag_mean, mag_std, mom_mean,
                temp_dist[0], temp_dist[1], temp_dist[2], temp_dist[3],
                nonzero, weight_count, weight_bytes, rss, cycle_ms);

            eprintln!("  Cycle {:3} | train {:.1}% | test {:.1}% | collapse {:.2} | mag {:.0}±{:.0} | mom {:.0} | H/W/C/F {}/{}/{}/{} | nz {} | {:.0}ms | {:.1}MB",
                cycle, train_acc * 100.0, metrics.test_acc * 100.0,
                metrics.collapse_score, mag_mean, mag_std, mom_mean,
                temp_dist[0], temp_dist[1], temp_dist[2], temp_dist[3],
                nonzero, cycle_ms, rss);

            // Print per-class accuracy
            let pca_str = metrics.per_class_acc.iter().enumerate()
                .map(|(i, a)| format!("d{}:{:.0}%", i, a * 100.0))
                .collect::<Vec<_>>().join(" ");
            eprintln!("          | {}", pca_str);

            // Print confusion matrix summary (diagonal)
            let diag: Vec<String> = (0..10).map(|i| format!("{}", metrics.confusion[i][i])).collect();
            eprintln!("          | diag: [{}]", diag.join(", "));
        } else {
            eprintln!("  Cycle {:3} | train {:.1}% | {:.0}ms", cycle, train_acc * 100.0, cycle_ms);
        }
    }

    eprintln!();
    eprintln!("=== Experiment A complete ===");
}
