//! Evaluation: accuracy, confusion matrix, thermal stats.

use crate::mnist::MnistDataset;
use crate::encode::encode_image;
use crate::network::AntNetwork;

pub struct EvalResult {
    pub accuracy: f32,
    pub per_class: [f32; 10],
    pub confusion: [[u32; 10]; 10], // confusion[true_label][predicted]
    pub total: usize,
    pub correct: usize,
}

pub fn evaluate(net: &AntNetwork, dataset: &MnistDataset) -> EvalResult {
    let mut confusion = [[0u32; 10]; 10];
    let mut correct = 0usize;

    for image in &dataset.images {
        let input = encode_image(&image.pixels);
        let predicted = net.predict(&input);
        let true_label = image.label as usize;

        confusion[true_label][predicted] += 1;
        if predicted == true_label { correct += 1; }
    }

    let total = dataset.len();
    let accuracy = correct as f32 / total as f32;

    let mut per_class = [0.0f32; 10];
    for c in 0..10 {
        let class_total: u32 = confusion[c].iter().sum();
        per_class[c] = if class_total > 0 {
            confusion[c][c] as f32 / class_total as f32
        } else { 0.0 };
    }

    EvalResult { accuracy, per_class, confusion, total, correct }
}

impl EvalResult {
    pub fn print(&self) {
        println!("  Accuracy: {:.1}% ({}/{})", self.accuracy * 100.0, self.correct, self.total);
        print!("  Per-class: ");
        for c in 0..10 {
            print!("{}:{:.0}% ", c, self.per_class[c] * 100.0);
        }
        println!();

        println!("  Confusion matrix (rows=true, cols=predicted):");
        print!("      ");
        for c in 0..10 { print!("{:>5}", c); }
        println!();
        for r in 0..10 {
            print!("  {:>2}: ", r);
            for c in 0..10 {
                print!("{:>5}", self.confusion[r][c]);
            }
            println!();
        }
    }
}
