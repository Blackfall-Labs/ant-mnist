//! Voter analysis: what makes voters agree, disagree, and where do they fail?
//! Runs 10 voters, then analyzes their predictions on the test set.

mod mnist;
mod lenet_mastery;

use std::path::PathBuf;
use rayon::prelude::*;
use mnist::MnistDataset;
use lenet_mastery::LeNetMastery;

fn main() {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    eprintln!("=== Voter Disagreement Analysis ===");
    eprintln!("Loading MNIST...");

    let train = MnistDataset::load(
        &PathBuf::from(&data_dir).join("train-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("train-labels-idx1-ubyte"),
    ).expect("load train");
    let test = MnistDataset::load(
        &PathBuf::from(&data_dir).join("t10k-images-idx3-ubyte"),
        &PathBuf::from(&data_dir).join("t10k-labels-idx1-ubyte"),
    ).expect("load test");

    let n_voters = 10;
    let seeds: Vec<u64> = (0..n_voters as u64).map(|i| 42 + i * 97).collect();
    let mut nets: Vec<LeNetMastery> = seeds.iter().map(|&s| LeNetMastery::new(s)).collect();

    // Train 1 epoch
    eprintln!("Training {} voters (1 epoch)...", n_voters);
    nets.par_iter_mut().for_each(|net| {
        for i in 0..train.len() {
            let img = &train.images[i];
            net.train_one(&img.pixels, img.label as usize);
        }
        net.decay();
    });

    // Collect all predictions
    eprintln!("Analyzing predictions on test set...");
    let n_test = test.len();

    // predictions[voter][sample] = predicted class
    let predictions: Vec<Vec<u8>> = nets.iter().map(|net| {
        (0..n_test).map(|i| net.predict(&test.images[i].pixels) as u8).collect()
    }).collect();

    let labels: Vec<u8> = test.images.iter().map(|img| img.label).collect();

    // --- Analysis 1: Per-voter accuracy ---
    eprintln!("\n--- Per-Voter Accuracy ---");
    let mut voter_accs = vec![0.0f64; n_voters];
    for v in 0..n_voters {
        let correct: u32 = (0..n_test).map(|i| if predictions[v][i] == labels[i] { 1 } else { 0 }).sum();
        voter_accs[v] = correct as f64 / n_test as f64;
        eprintln!("  Voter {:2}: {:.1}%", v, voter_accs[v] * 100.0);
    }

    // --- Analysis 2: Agreement rate ---
    eprintln!("\n--- Agreement Analysis ---");
    let mut all_agree = 0u32;
    let mut all_agree_correct = 0u32;
    let mut all_agree_wrong = 0u32;
    let mut majority_correct = 0u32;
    let mut minority_correct = 0u32; // majority wrong but some voters right
    let mut all_wrong = 0u32;

    for i in 0..n_test {
        let mut votes = [0u32; 10];
        for v in 0..n_voters {
            votes[predictions[v][i] as usize] += 1;
        }
        let winner = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
        let max_votes = votes[winner];
        let any_correct = (0..n_voters).any(|v| predictions[v][i] == labels[i]);
        let majority_right = winner == labels[i] as usize;

        if max_votes == n_voters as u32 {
            all_agree += 1;
            if majority_right { all_agree_correct += 1; } else { all_agree_wrong += 1; }
        }

        if majority_right {
            majority_correct += 1;
        } else if any_correct {
            minority_correct += 1;
        } else {
            all_wrong += 1;
        }
    }

    eprintln!("  All voters agree:        {:5} ({:.1}%)", all_agree, all_agree as f64 / n_test as f64 * 100.0);
    eprintln!("    ...and correct:        {:5} ({:.1}%)", all_agree_correct, all_agree_correct as f64 / n_test as f64 * 100.0);
    eprintln!("    ...but wrong:          {:5} ({:.1}%)", all_agree_wrong, all_agree_wrong as f64 / n_test as f64 * 100.0);
    eprintln!("  Majority correct:        {:5} ({:.1}%)", majority_correct, majority_correct as f64 / n_test as f64 * 100.0);
    eprintln!("  Majority wrong, some ok: {:5} ({:.1}%)", minority_correct, minority_correct as f64 / n_test as f64 * 100.0);
    eprintln!("  ALL voters wrong:        {:5} ({:.1}%)", all_wrong, all_wrong as f64 / n_test as f64 * 100.0);

    // --- Analysis 3: Per-digit confusion patterns ---
    eprintln!("\n--- Ensemble Confusion (what does the ensemble confuse?) ---");
    let mut confusion = [[0u32; 10]; 10];
    for i in 0..n_test {
        let mut votes = [0u32; 10];
        for v in 0..n_voters { votes[predictions[v][i] as usize] += 1; }
        let winner = votes.iter().enumerate().max_by_key(|(_, &v)| v).map(|(i, _)| i).unwrap_or(0);
        let label = labels[i] as usize;
        if label < 10 && winner < 10 { confusion[label][winner] += 1; }
    }
    eprintln!("     0    1    2    3    4    5    6    7    8    9");
    for r in 0..10 {
        let row: Vec<String> = confusion[r].iter().map(|&v| {
            if v == 0 { "   .".into() } else { format!("{:4}", v) }
        }).collect();
        eprintln!("  {}: {}", r, row.join(""));
    }

    // --- Analysis 4: What do voters disagree on most? ---
    eprintln!("\n--- Most Confused Pairs (true→predicted, ensemble errors) ---");
    let mut pair_counts: Vec<(usize, usize, u32)> = Vec::new();
    for r in 0..10 {
        for c in 0..10 {
            if r != c && confusion[r][c] > 0 {
                pair_counts.push((r, c, confusion[r][c]));
            }
        }
    }
    pair_counts.sort_by(|a, b| b.2.cmp(&a.2));
    for &(true_d, pred_d, count) in pair_counts.iter().take(15) {
        eprintln!("  {} → {} : {:4} errors", true_d, pred_d, count);
    }

    // --- Analysis 5: Per-voter agreement matrix ---
    eprintln!("\n--- Voter Agreement Matrix (how often do pairs agree?) ---");
    eprintln!("        {:>4}{:>4}{:>4}{:>4}{:>4}{:>4}{:>4}{:>4}{:>4}{:>4}",
        "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9");
    for a in 0..n_voters {
        let mut row = Vec::new();
        for b in 0..n_voters {
            let agree: u32 = (0..n_test).map(|i| {
                if predictions[a][i] == predictions[b][i] { 1 } else { 0 }
            }).sum();
            row.push(format!("{:4}", (agree as f64 / n_test as f64 * 100.0) as u32));
        }
        eprintln!("  v{}: {}", a, row.join(""));
    }

    // --- Analysis 6: Which samples are ALWAYS wrong? ---
    eprintln!("\n--- Hardest Samples (all voters wrong, first 20) ---");
    let mut hard_count = 0;
    for i in 0..n_test {
        if hard_count >= 20 { break; }
        let any_correct = (0..n_voters).any(|v| predictions[v][i] == labels[i]);
        if !any_correct {
            let vote_dist: Vec<String> = (0..n_voters).map(|v| format!("{}", predictions[v][i])).collect();
            eprintln!("  sample {:5} | label {} | votes: [{}]", i, labels[i], vote_dist.join(","));
            hard_count += 1;
        }
    }

    eprintln!("\n=== Analysis complete ===");
}
