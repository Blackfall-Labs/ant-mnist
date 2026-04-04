//! Dual LeNet-5 experiment runner.
//! Usage: cargo run --release --bin lenet-experiment [data_dir]

mod mnist;
mod lenet_f32;
mod lenet_i16;
mod lenet_mastery;

use std::path::PathBuf;
use std::time::Instant;
use mnist::MnistDataset;
use lenet_f32::LeNetF32;
use lenet_i16::LeNetI16;
use lenet_mastery::LeNetMastery;

fn peak_rss_mb() -> f64 {
    #[cfg(windows)]
    {
        use std::mem;
        #[repr(C)]
        struct Pmc { cb: u32, pf: u32, peak_ws: usize, ws: usize, qppp: usize, qpp: usize, qpnpp: usize, qnpp: usize, pfu: usize, ppfu: usize }
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
        }
        unsafe {
            let mut p: Pmc = mem::zeroed();
            p.cb = mem::size_of::<Pmc>() as u32;
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb);
            p.peak_ws as f64 / (1024.0 * 1024.0)
        }
    }
    #[cfg(not(windows))]
    { 0.0 }
}

fn evaluate(net: &LeNetF32, test: &MnistDataset, count: usize) -> (f64, [f64; 10]) {
    let n = count.min(test.len());
    let mut correct = 0u32;
    let mut per_class_correct = [0u32; 10];
    let mut per_class_total = [0u32; 10];

    for i in 0..n {
        let img = &test.images[i];
        let label = img.label as usize;
        let pred = net.predict(&img.pixels);

        if label < 10 {
            per_class_total[label] += 1;
            if pred == label {
                correct += 1;
                per_class_correct[label] += 1;
            }
        }
    }

    let acc = correct as f64 / n as f64;
    let mut pca = [0.0f64; 10];
    for d in 0..10 {
        pca[d] = if per_class_total[d] > 0 {
            per_class_correct[d] as f64 / per_class_total[d] as f64
        } else { 0.0 };
    }
    (acc, pca)
}

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());

    eprintln!("=== Dual LeNet-5 Experiment ===");
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

    // CSV header
    println!("version,epoch,test_acc,per_class,train_acc,epoch_s,weight_count,weight_kb,rss_mb,w_mean,w_std");

    let epochs = 10;

    // f32 baseline skipped — results already recorded in experiments/lenet_f32.csv
    // f32 epoch 10: 98.8% test accuracy, 44426 weights, 173.5 KB, 32s/epoch

    // --- P×M×K signal-native ---
    eprintln!("\n=== Phase 2: P×M×K Signal LeNet-5 (frozen conv + basin learning) ===");
    let mut net16 = LeNetI16::new(42);
    let ws16 = net16.weight_stats();
    eprintln!("  total weights: {} ({:.1} KB)", ws16.count, ws16.bytes as f64 / 1024.0);
    eprintln!("  plastic weights: {} (FC3 output layer)", ws16.plastic_count);
    eprintln!("  SignalWeight size: {} bytes", std::mem::size_of::<lenet_i16::SignalWeight>());

    for epoch in 1..=epochs {
        let t0 = Instant::now();
        let mut correct = 0u32;
        let mut total = 0u32;

        for i in 0..train.len() {
            let img = &train.images[i];
            let label = img.label as usize;
            let (_, c) = net16.train_one(&img.pixels, label);
            if c { correct += 1; }
            total += 1;

            if (i + 1) % 10000 == 0 {
                eprintln!("    {}/{} ({:.1}%)", i + 1, train.len(),
                    correct as f64 / total as f64 * 100.0);
            }
        }
        net16.decay();

        let epoch_s = t0.elapsed().as_secs_f64();
        let train_acc = correct as f64 / total as f64;

        let (test_acc, pca) = evaluate_i16(&net16, &test, 10000);
        let ws16 = net16.weight_stats();
        let rss = peak_rss_mb();

        let pca_str = pca.iter().map(|a| format!("{:.3}", a)).collect::<Vec<_>>().join("|");

        println!("pmk,{},{:.4},{},{:.4},{:.1},{},{:.1},{:.1},{:.4},{:.4}",
            epoch, test_acc, pca_str, train_acc, epoch_s,
            ws16.count, ws16.bytes as f64 / 1024.0, rss, ws16.mean, ws16.std);

        let pca_short = pca.iter().enumerate()
            .map(|(i, a)| format!("d{}:{:.0}%", i, a * 100.0))
            .collect::<Vec<_>>().join(" ");

        eprintln!("  Epoch {:2} | train {:.1}% | test {:.1}% | {:.0}s | nz {}/{} | H/W/C/F {}/{}/{}/{} | s:{} t:{} | {:.1}MB",
            epoch, train_acc * 100.0, test_acc * 100.0, epoch_s,
            ws16.nonzero, ws16.plastic_count,
            ws16.temp_dist[0], ws16.temp_dist[1], ws16.temp_dist[2], ws16.temp_dist[3],
            net16.steps, net16.transitions, rss);
        eprintln!("         | {}", pca_short);
    }

    eprintln!("\n=== P×M×K Signal LeNet-5 complete ===");

    // --- Mastery learning ---
    eprintln!("\n=== Phase 3: Mastery LeNet-5 (frozen random + mastery output) ===");
    let mut netm = LeNetMastery::new(42);
    eprintln!("  total weights: {} ({:.1} KB)", netm.total_weights(), netm.total_weights() as f64 * 2.0 / 1024.0);
    eprintln!("  plastic weights: {} (output layer only)", netm.plastic_weights());

    for epoch in 1..=epochs {
        let t0 = Instant::now();
        let mut correct = 0u32;
        let mut total = 0u32;

        for i in 0..train.len() {
            let img = &train.images[i];
            let label = img.label as usize;
            let (_, c) = netm.train_one(&img.pixels, label);
            if c { correct += 1; }
            total += 1;

            if (i + 1) % 10000 == 0 {
                eprintln!("    {}/{} ({:.1}%)", i + 1, train.len(),
                    correct as f64 / total as f64 * 100.0);
            }
        }
        netm.decay();

        let epoch_s = t0.elapsed().as_secs_f64();
        let train_acc = correct as f64 / total as f64;

        // Eval
        let (test_acc, pca) = evaluate_mastery(&netm, &test, 10000);
        let (w_mean, w_std) = netm.output.weight_stats();
        let rss = peak_rss_mb();

        let pca_str = pca.iter().map(|a| format!("{:.3}", a)).collect::<Vec<_>>().join("|");

        println!("mastery,{},{:.4},{},{:.4},{:.1},{},{:.1},{:.1},{:.4},{:.4}",
            epoch, test_acc, pca_str, train_acc, epoch_s,
            netm.total_weights(), netm.total_weights() as f64 * 2.0 / 1024.0, rss,
            w_mean, w_std);

        let pca_short = pca.iter().enumerate()
            .map(|(i, a)| format!("d{}:{:.0}%", i, a * 100.0))
            .collect::<Vec<_>>().join(" ");

        eprintln!("  Epoch {:2} | train {:.1}% | test {:.1}% | {:.0}s | hidden nz {}/{} s:{} t:{} | output nz {}/{} s:{} t:{} | {:.1}MB",
            epoch, train_acc * 100.0, test_acc * 100.0, epoch_s,
            netm.hidden.nonzero_count(), netm.hidden.weights.len(),
            netm.hidden.steps, netm.hidden.transitions,
            netm.output.nonzero_count(), netm.output.weights.len(),
            netm.output.steps, netm.output.transitions, rss);
        eprintln!("         | {}", pca_short);
    }

    eprintln!("\n=== Mastery LeNet-5 complete ===");
    eprintln!("\n=== All experiments complete ===");
}

fn evaluate_mastery(net: &LeNetMastery, test: &MnistDataset, count: usize) -> (f64, [f64; 10]) {
    let n = count.min(test.len());
    let mut correct = 0u32;
    let mut per_class_correct = [0u32; 10];
    let mut per_class_total = [0u32; 10];

    for i in 0..n {
        let img = &test.images[i];
        let label = img.label as usize;
        let pred = net.predict(&img.pixels);

        if label < 10 {
            per_class_total[label] += 1;
            if pred == label {
                correct += 1;
                per_class_correct[label] += 1;
            }
        }
    }

    let acc = correct as f64 / n as f64;
    let mut pca = [0.0f64; 10];
    for d in 0..10 {
        pca[d] = if per_class_total[d] > 0 {
            per_class_correct[d] as f64 / per_class_total[d] as f64
        } else { 0.0 };
    }
    (acc, pca)
}

fn evaluate_i16(net: &LeNetI16, test: &MnistDataset, count: usize) -> (f64, [f64; 10]) {
    let n = count.min(test.len());
    let mut correct = 0u32;
    let mut per_class_correct = [0u32; 10];
    let mut per_class_total = [0u32; 10];

    for i in 0..n {
        let img = &test.images[i];
        let label = img.label as usize;
        let pred = net.predict(&img.pixels);

        if label < 10 {
            per_class_total[label] += 1;
            if pred == label {
                correct += 1;
                per_class_correct[label] += 1;
            }
        }
    }

    let acc = correct as f64 / n as f64;
    let mut pca = [0.0f64; 10];
    for d in 0..10 {
        pca[d] = if per_class_total[d] > 0 {
            per_class_correct[d] as f64 / per_class_total[d] as f64
        } else { 0.0 };
    }
    (acc, pca)
}
