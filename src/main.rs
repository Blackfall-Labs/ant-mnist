mod mnist;
mod encode;
mod network;
mod eval;

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let checkpoint_path = std::env::args().nth(2).unwrap_or("checkpoints/ant_mnist.ant".into());

    println!("=== ANT-MNIST: LeNet-5 with Atomic Neural Transistors ===\n");

    // Load datasets
    print!("Loading MNIST... ");
    let train = mnist::MnistDataset::load(
        Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)),
        Path::new(&format!("{}/train-labels-idx1-ubyte", data_dir)),
    )?;
    let test = mnist::MnistDataset::load(
        Path::new(&format!("{}/t10k-images-idx3-ubyte", data_dir)),
        Path::new(&format!("{}/t10k-labels-idx1-ubyte", data_dir)),
    )?;
    println!("train={}, test={}", train.len(), test.len());

    // Show a few samples
    println!("\nSample images (ternary-encoded):");
    for i in [0, 1, 7, 3] {
        let img = &train.images[i];
        let encoded = encode::encode_image(&img.pixels);
        println!("--- Digit {} ---", img.label);
        encode::print_encoded(&encoded);
    }

    // Create network
    let mut net = network::AntNetwork::new();

    // Try loading checkpoint
    let cp = Path::new(&checkpoint_path);
    if cp.exists() {
        match net.load(cp) {
            Ok(_) => println!("\nLoaded checkpoint: {}", checkpoint_path),
            Err(e) => println!("\nCheckpoint load failed ({}), starting fresh", e),
        }
    }

    // Baseline: random weights
    println!("\n--- Baseline (random weights) ---");
    let baseline = eval::evaluate(&net, &test);
    baseline.print();

    // Training cycles
    let max_cycles = 5;
    let mut best_test_acc = 0.0f32;

    for cycle in 1..=max_cycles {
        println!("\n=== Training Cycle {} ===", cycle);
        let cycle_start = Instant::now();
        net.reset_counters();

        // Shuffle training data (deterministic pseudo-shuffle via cycle seed)
        let mut indices: Vec<usize> = (0..train.len()).collect();
        // Simple Fisher-Yates with xorshift
        let mut rng = (cycle as u64) * 6364136223846793005 + 1;
        for i in (1..indices.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut batch_correct = 0u64;
        let mut batch_total = 0u64;

        for (step, &idx) in indices.iter().enumerate() {
            let image = &train.images[idx];
            let input = encode::encode_image(&image.pixels);
            let predicted = net.learn(&input, image.label);

            batch_total += 1;
            if predicted == image.label as usize { batch_correct += 1; }

            // Progress every 10K
            if (step + 1) % 10000 == 0 {
                let batch_acc = batch_correct as f32 / batch_total as f32 * 100.0;
                print!("  {:>5}/{}: train={:.1}%  {}\r",
                    step + 1, train.len(), batch_acc, net.thermal_summary());
                batch_correct = 0;
                batch_total = 0;
            }
        }

        let train_acc = net.accuracy() * 100.0;
        let cycle_time = cycle_start.elapsed().as_secs_f32();
        println!("  Train: {:.1}%  ({:.1}s)                              ", train_acc, cycle_time);
        println!("  Thermal: {}", net.thermal_summary());
        println!("  Transitions: L1:{} L2:{} L3:{}", net.m1.transitions, net.m2.transitions, net.m3.transitions);

        // Evaluate on test set
        println!("\n  --- Test Evaluation ---");
        let result = eval::evaluate(&net, &test);
        result.print();

        if result.accuracy > best_test_acc {
            best_test_acc = result.accuracy;
            // Save checkpoint
            std::fs::create_dir_all(Path::new(&checkpoint_path).parent().unwrap_or(Path::new(".")))?;
            net.save(cp)?;
            println!("  Saved checkpoint ({:.1}%)", best_test_acc * 100.0);
        }

        if result.accuracy >= 0.95 {
            println!("\n>>> TARGET REACHED: {:.1}% <<<", result.accuracy * 100.0);
            break;
        }
    }

    println!("\n=== Done. Best test accuracy: {:.1}% ===", best_test_acc * 100.0);
    Ok(())
}
