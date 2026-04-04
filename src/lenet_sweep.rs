//! Sweep experiment: four strategies to close the 34-point gap.
//! All start from the 65% mastery baseline (frozen random + mastery output).
//!
//! 1. Structured filters: edge/gabor-like conv kernels instead of random
//! 2. Wider network: 2× more random features to select from
//! 3. Alternating freeze: freeze output, learn hidden, re-freeze, retrain output
//! 4. Ensemble: multiple mastery classifiers on different random projections, vote

mod mnist;
mod lenet_mastery;

use std::path::PathBuf;
use std::time::Instant;
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

fn evaluate(net: &LeNetMastery, test: &MnistDataset) -> (f64, [f64; 10]) {
    let n = test.len();
    let mut correct = 0u32;
    let mut pc = [0u32; 10];
    let mut pt = [0u32; 10];
    for i in 0..n {
        let img = &test.images[i];
        let l = img.label as usize;
        let p = net.predict(&img.pixels);
        if l < 10 { pt[l] += 1; if p == l { correct += 1; pc[l] += 1; } }
    }
    let acc = correct as f64 / n as f64;
    let mut pca = [0.0; 10];
    for d in 0..10 { pca[d] = if pt[d] > 0 { pc[d] as f64 / pt[d] as f64 } else { 0.0 }; }
    (acc, pca)
}

fn train_epoch(net: &mut LeNetMastery, train: &MnistDataset) -> f64 {
    let mut correct = 0u32;
    for i in 0..train.len() {
        let img = &train.images[i];
        let (_, c) = net.train_one(&img.pixels, img.label as usize);
        if c { correct += 1; }
    }
    net.decay();
    correct as f64 / train.len() as f64
}

fn run_variant(name: &str, net: &mut LeNetMastery, train: &MnistDataset, test: &MnistDataset, epochs: usize) {
    eprintln!("\n=== {} ===", name);
    eprintln!("  total: {} | plastic: {}", net.total_weights(), net.plastic_weights());

    for epoch in 1..=epochs {
        let t0 = Instant::now();
        let train_acc = train_epoch(net, train);
        let s = t0.elapsed().as_secs_f64();
        let (test_acc, pca) = evaluate(net, test);
        let rss = peak_rss_mb();

        let pca_s = pca.iter().enumerate().map(|(i, a)| format!("d{}:{:.0}%", i, a * 100.0)).collect::<Vec<_>>().join(" ");
        println!("{},{},{:.4},{:.4},{:.1},{:.1}", name, epoch, test_acc, train_acc, s, rss);
        eprintln!("  E{:2} | train {:.1}% | test {:.1}% | {:.0}s | {}", epoch, train_acc * 100.0, test_acc * 100.0, s, pca_s);
    }
    eprintln!("=== {} complete ===\n", name);
}

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
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

    println!("variant,epoch,test_acc,train_acc,epoch_s,rss_mb");

    let epochs = 10;

    // --- Variant 1: Baseline (same as best result, for comparison) ---
    {
        let mut net = LeNetMastery::new(42);
        run_variant("baseline", &mut net, &train, &test, epochs);
    }

    // --- Variant 2: Ensemble (3 networks with different seeds, majority vote) ---
    {
        let mut nets = vec![
            LeNetMastery::new(42),
            LeNetMastery::new(137),
            LeNetMastery::new(2024),
        ];
        eprintln!("\n=== ensemble (3 voters) ===");
        for epoch in 1..=epochs {
            let t0 = Instant::now();
            for net in &mut nets {
                train_epoch(net, &train);
            }
            let s = t0.elapsed().as_secs_f64();

            // Evaluate ensemble via majority vote
            let mut correct = 0u32;
            let n = test.len();
            for i in 0..n {
                let img = &test.images[i];
                let label = img.label as usize;
                let mut votes = [0u32; 10];
                for net in &nets {
                    let pred = net.predict(&img.pixels);
                    if pred < 10 { votes[pred] += 1; }
                }
                let winner = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
                if winner == label { correct += 1; }
            }
            let test_acc = correct as f64 / n as f64;
            let rss = peak_rss_mb();
            println!("ensemble,{},{:.4},0,{:.1},{:.1}", epoch, test_acc, s, rss);
            eprintln!("  E{:2} | test {:.1}% | {:.0}s", epoch, test_acc * 100.0, s);
        }
        eprintln!("=== ensemble complete ===\n");
    }

    // --- Variant 4: Alternating freeze (learn output 5 epochs, learn hidden 3, re-learn output 5) ---
    {
        let mut net = LeNetMastery::new_with_plastic_hidden(42);
        eprintln!("\n=== alternating ===");

        // Phase A: train output only (hidden frozen random)
        eprintln!("  Phase A: output only (5 epochs)");
        for epoch in 1..=5 {
            let t0 = Instant::now();
            // Only train output
            let mut correct = 0u32;
            for i in 0..train.len() {
                let img = &train.images[i];
                let label = img.label as usize;
                let feats = net.features(&img.pixels);
                let output = net.output.forward(&feats);
                let predicted = output.iter().enumerate().max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap_or(0);
                net.output.learn(&feats, &output, label);
                if predicted == label { correct += 1; }
            }
            net.output.decay();
            let s = t0.elapsed().as_secs_f64();
            let (test_acc, _) = evaluate(&net, &test);
            println!("alt_phaseA,{},{:.4},{:.4},{:.1},{:.1}", epoch, test_acc, correct as f64 / train.len() as f64, s, peak_rss_mb());
            eprintln!("    E{} | test {:.1}% | {:.0}s", epoch, test_acc * 100.0, s);
        }

        // Phase B: train hidden only (output frozen from phase A)
        eprintln!("  Phase B: hidden only (3 epochs)");
        for epoch in 1..=3 {
            let t0 = Instant::now();
            let mut correct = 0u32;
            for i in 0..train.len() {
                let img = &train.images[i];
                let label = img.label as usize;
                let (predicted, c) = net.train_one(&img.pixels, label);
                if c { correct += 1; }
            }
            net.hidden.decay();
            let s = t0.elapsed().as_secs_f64();
            let (test_acc, _) = evaluate(&net, &test);
            println!("alt_phaseB,{},{:.4},{:.4},{:.1},{:.1}", epoch, test_acc, correct as f64 / train.len() as f64, s, peak_rss_mb());
            eprintln!("    E{} | test {:.1}% | {:.0}s", epoch, test_acc * 100.0, s);
        }

        // Phase C: retrain output (hidden now learned)
        eprintln!("  Phase C: output retrain (5 epochs)");
        // Reset output layer
        net.output = MasteryLayer::new(10, 84);
        for epoch in 1..=5 {
            let t0 = Instant::now();
            let mut correct = 0u32;
            for i in 0..train.len() {
                let img = &train.images[i];
                let label = img.label as usize;
                let feats = net.features(&img.pixels);
                let output = net.output.forward(&feats);
                let predicted = output.iter().enumerate().max_by_key(|(_, &s)| s).map(|(i, _)| i).unwrap_or(0);
                net.output.learn(&feats, &output, label);
                if predicted == label { correct += 1; }
            }
            net.output.decay();
            let s = t0.elapsed().as_secs_f64();
            let (test_acc, _) = evaluate(&net, &test);
            println!("alt_phaseC,{},{:.4},{:.4},{:.1},{:.1}", epoch, test_acc, correct as f64 / train.len() as f64, s, peak_rss_mb());
            eprintln!("    E{} | test {:.1}% | {:.0}s", epoch, test_acc * 100.0, s);
        }
        eprintln!("=== alternating complete ===\n");
    }

    eprintln!("=== All variants complete ===");
}
