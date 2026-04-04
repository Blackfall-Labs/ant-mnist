//! Dual hemisphere LeNet-5: two competing pathways with lateral inhibition.
//!
//! Each hemisphere has its own frozen random projections and plastic output.
//! Their outputs inhibit each other — when A says "4" strongly, B's "4" is
//! suppressed, forcing B to find different evidence. Agreement = high confidence.
//! Disagreement = the stronger conviction wins.

mod mnist;
mod lenet_mastery;

use std::path::PathBuf;
use std::time::Instant;
use rayon::prelude::*;
use mnist::MnistDataset;
use lenet_mastery::LeNetMastery;

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

/// Two hemispheres — near identical, slight variations, full information preserved.
struct DualHemisphere {
    left: LeNetMastery,
    right: LeNetMastery,
}

impl DualHemisphere {
    fn new(seed_left: u64, seed_right: u64) -> Self {
        // Same architecture, different random projections.
        // Right hemisphere gets a slightly different pressure threshold —
        // making it more cautious (needs more evidence to commit).
        let mut right = LeNetMastery::new(seed_right);
        right.output.pressure_threshold = 12; // vs left's default 8

        Self {
            left: LeNetMastery::new(seed_left),
            right,
        }
    }

    /// Both hemispheres process the same image.
    /// Combined output = full sum of both (no suppression, no information loss).
    /// Competition is by magnitude: the hemisphere with stronger evidence for
    /// a class contributes more to that class's combined score naturally.
    fn predict_combined(&self, image: &[u8]) -> (usize, Vec<i32>) {
        let scores_l = self.left.scores(image);
        let scores_r = self.right.scores(image);

        let mut combined = [0i32; 10];
        for c in 0..10 {
            combined[c] = scores_l[c] as i32 + scores_r[c] as i32;
        }

        let winner = combined.iter().enumerate()
            .max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap_or(0);
        (winner, combined.to_vec())
    }

    fn train_one(&mut self, image: &[u8], label: usize) -> (usize, bool) {
        // Both hemispheres learn from the same sample independently
        self.left.train_one(image, label);
        self.right.train_one(image, label);

        let (pred, _) = self.predict_combined(image);
        (pred, pred == label)
    }

    fn decay(&mut self) {
        self.left.decay();
        self.right.decay();
    }
}

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let n_pairs: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let epochs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    eprintln!("=== Dual Hemisphere LeNet-5: {} pairs ({} total voters), {} epochs ===",
        n_pairs, n_pairs * 2, epochs);
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

    // Create pairs with different seeds
    let mut pairs: Vec<DualHemisphere> = (0..n_pairs)
        .map(|i| DualHemisphere::new(42 + i as u64 * 97, 1000 + i as u64 * 131))
        .collect();

    println!("pairs,epoch,test_inhibited,test_ensemble,train_acc,epoch_s,rss_mb");

    for epoch in 1..=epochs {
        let t0 = Instant::now();

        // Train all pairs in parallel
        let train_results: Vec<u32> = pairs.par_iter_mut().map(|pair| {
            let mut correct = 0u32;
            for i in 0..train.len() {
                let img = &train.images[i];
                let (_, c) = pair.train_one(&img.pixels, img.label as usize);
                if c { correct += 1; }
            }
            pair.decay();
            correct
        }).collect();

        let train_acc = train_results.iter().sum::<u32>() as f64 / (train.len() * n_pairs) as f64;
        let train_s = t0.elapsed().as_secs_f64();

        // Evaluate: inhibited pairs + cross-pair ensemble
        let mut inhibited_correct = 0u32;
        let mut ensemble_correct = 0u32;
        let mut pc_inh = [0u32; 10];
        let mut pc_ens = [0u32; 10];
        let mut pt = [0u32; 10];

        for i in 0..test.len() {
            let img = &test.images[i];
            let label = img.label as usize;

            // Each pair combines its two hemispheres
            let pair_preds: Vec<(usize, Vec<i32>)> = pairs.iter()
                .map(|p| p.predict_combined(&img.pixels))
                .collect();

            // Cross-pair: sum combined scores across all pairs
            let mut ensemble_scores = [0i64; 10];
            for (_, scores) in &pair_preds {
                for (c, &s) in scores.iter().enumerate() {
                    ensemble_scores[c] += s as i64;
                }
            }
            let ensemble_winner = ensemble_scores.iter().enumerate()
                .max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap_or(0);

            let best_pair = pair_preds[0].0;

            if label < 10 {
                pt[label] += 1;
                if best_pair == label { inhibited_correct += 1; pc_inh[label] += 1; }
                if ensemble_winner == label { ensemble_correct += 1; pc_ens[label] += 1; }
            }
        }

        let inh_acc = inhibited_correct as f64 / test.len() as f64;
        let ens_acc = ensemble_correct as f64 / test.len() as f64;
        let total_s = t0.elapsed().as_secs_f64();
        let rss = peak_rss_mb();

        let pca = (0..10).map(|d| {
            format!("d{}:{:.0}%", d, if pt[d] > 0 { pc_ens[d] as f64 / pt[d] as f64 * 100.0 } else { 0.0 })
        }).collect::<Vec<_>>().join(" ");

        println!("{},{},{:.4},{:.4},{:.4},{:.1},{:.1}",
            n_pairs, epoch, inh_acc, ens_acc, train_acc, total_s, rss);
        eprintln!("  E{:2} | pair0 {:.1}% | ensemble {:.1}% | train {:.1}% | {:.0}s | {:.1}MB",
            epoch, inh_acc * 100.0, ens_acc * 100.0, train_acc * 100.0, total_s, rss);
        eprintln!("      | {}", pca);
    }

    eprintln!("\n=== Dual Hemisphere complete ===");
}
