//! Ensemble LeNet-5: shared frozen conv layers, N parallel FC classifiers.
//!
//! Conv1 + Conv2 are computed ONCE per image (shared across all voters).
//! Each voter has its own random FC1 → frozen FC2 → mastery FC3.
//! Training runs in parallel via rayon.
//! Prediction via majority vote.

mod mnist;
mod lenet_mastery;

use std::path::PathBuf;
use std::time::Instant;
use rayon::prelude::*;
use mnist::MnistDataset;
use lenet_mastery::{LeNetMastery, MasteryLayer};

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
    let n_voters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let epochs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    eprintln!("=== Ensemble LeNet-5: {} voters, {} epochs ===", n_voters, epochs);
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

    // Create N voters with different seeds
    let seeds: Vec<u64> = (0..n_voters as u64).map(|i| 42 + i * 97).collect();
    let mut nets: Vec<LeNetMastery> = seeds.iter().map(|&s| LeNetMastery::new(s)).collect();

    let plastic_per = nets[0].plastic_weights();
    let total_per = nets[0].total_weights();
    eprintln!("  per voter: {} total, {} plastic", total_per, plastic_per);
    eprintln!("  total: {} weights across {} voters", total_per * n_voters, n_voters);
    eprintln!("  rayon threads: {}", rayon::current_num_threads());

    println!("voters,epoch,test_acc,train_acc,epoch_s,rss_mb,per_class");

    for epoch in 1..=epochs {
        let t0 = Instant::now();

        // Train all voters in parallel
        let train_results: Vec<u32> = nets.par_iter_mut().map(|net| {
            let mut correct = 0u32;
            for i in 0..train.len() {
                let img = &train.images[i];
                let (_, c) = net.train_one(&img.pixels, img.label as usize);
                if c { correct += 1; }
            }
            net.decay();
            correct
        }).collect();

        let total_correct: u32 = train_results.iter().sum();
        let train_acc = total_correct as f64 / (train.len() * n_voters) as f64;
        let train_s = t0.elapsed().as_secs_f64();

        // Evaluate via majority vote
        let t1 = Instant::now();
        let mut test_correct = 0u32;
        let mut pc = [0u32; 10];
        let mut pt = [0u32; 10];

        for i in 0..test.len() {
            let img = &test.images[i];
            let label = img.label as usize;
            let mut votes = [0u32; 10];
            for net in &nets {
                let pred = net.predict(&img.pixels);
                if pred < 10 { votes[pred] += 1; }
            }
            let winner = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
            if label < 10 {
                pt[label] += 1;
                if winner == label { test_correct += 1; pc[label] += 1; }
            }
        }

        let test_acc = test_correct as f64 / test.len() as f64;
        let eval_s = t1.elapsed().as_secs_f64();
        let total_s = t0.elapsed().as_secs_f64();
        let rss = peak_rss_mb();

        let mut pca = [0.0f64; 10];
        for d in 0..10 { pca[d] = if pt[d] > 0 { pc[d] as f64 / pt[d] as f64 } else { 0.0 }; }

        let pca_str = pca.iter().map(|a| format!("{:.3}", a)).collect::<Vec<_>>().join("|");
        let pca_short = pca.iter().enumerate().map(|(i, a)| format!("d{}:{:.0}%", i, a * 100.0)).collect::<Vec<_>>().join(" ");

        println!("{},{},{:.4},{:.4},{:.1},{:.1},{}", n_voters, epoch, test_acc, train_acc, total_s, rss, pca_str);
        eprintln!("  E{:2} | train {:.1}% | test {:.1}% | {:.0}s (train {:.0}s + eval {:.0}s) | {:.1}MB",
            epoch, train_acc * 100.0, test_acc * 100.0, total_s, train_s, eval_s, rss);
        eprintln!("      | {}", pca_short);

        // Evaluate individual voters
        if epoch == 1 || epoch == epochs {
            let mut voter_accs: Vec<(usize, f64)> = Vec::new();
            for (v, net) in nets.iter().enumerate() {
                let mut vc = 0u32;
                let n = test.len().min(10000);
                for i in 0..n {
                    let img = &test.images[i];
                    if net.predict(&img.pixels) == img.label as usize { vc += 1; }
                }
                let acc = vc as f64 / n as f64;
                voter_accs.push((v, acc));
                eprintln!("      | voter {} alone: {:.1}%", v, acc * 100.0);
            }

            // After epoch 1: prune weak voters (below 50%)
            if epoch == 1 {
                let before = nets.len();
                let mut keep = Vec::new();
                for (v, acc) in &voter_accs {
                    if *acc >= 0.50 {
                        keep.push(*v);
                    } else {
                        eprintln!("      | PRUNED voter {} ({:.1}% < 50%)", v, acc * 100.0);
                    }
                }
                if keep.len() < before {
                    // Measure pruned ensemble accuracy (without actually removing)
                    let mut pruned_correct = 0u32;
                    for i in 0..test.len() {
                        let img = &test.images[i];
                        let label = img.label as usize;
                        let mut votes = [0u32; 10];
                        for &k in &keep {
                            let pred = nets[k].predict(&img.pixels);
                            if pred < 10 { votes[pred] += 1; }
                        }
                        let winner = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
                        if label < 10 && winner == label { pruned_correct += 1; }
                    }
                    let pruned_acc = pruned_correct as f64 / test.len() as f64;
                    eprintln!("      | PRUNED ensemble ({} voters): {:.1}%", keep.len(), pruned_acc * 100.0);
                }
            }
        }
    }

    eprintln!("\n=== Ensemble complete ===");
}
