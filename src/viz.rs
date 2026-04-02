mod mnist;
mod encode;
mod network;
mod eval;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use serde::Serialize;
use tiny_http::{Server, Response, Header};

#[derive(Serialize, Clone, Default)]
struct TrainingState {
    cycle: u32,
    step: usize,
    total_steps: usize,
    train_acc: f32,
    test_acc: f32,
    best_test_acc: f32,
    per_class_acc: [f32; 10],
    confusion: [[u32; 10]; 10],
    thermal: LayerThermal,
    transitions: [u64; 3],
    running: bool,
    history: Vec<CycleSnapshot>,
    // Sample predictions for visualization
    samples: Vec<SamplePrediction>,
}

#[derive(Serialize, Clone, Default)]
struct LayerThermal {
    l1: [usize; 4], // hot, warm, cool, frozen
    l2: [usize; 4],
    l3: [usize; 4],
}

#[derive(Serialize, Clone)]
struct CycleSnapshot {
    cycle: u32,
    train_acc: f32,
    test_acc: f32,
    thermal_frozen_pct: [f32; 3],
}

#[derive(Serialize, Clone)]
struct SamplePrediction {
    pixels: Vec<u8>,  // 784 pixels
    label: u8,
    predicted: u8,
    correct: bool,
    output_scores: Vec<i32>, // 10 output currents
}

#[derive(Serialize)]
struct WeightLayerViz {
    rows: usize,
    cols: usize,
    // Sample of weight currents (not all — too large)
    sample_currents: Vec<i32>,
    sample_temps: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args().nth(1).unwrap_or("data".into());
    let port = std::env::args().nth(2).unwrap_or("8090".into());

    println!("=== ANT-MNIST Visualizer ===");
    println!("Loading MNIST...");

    let train = mnist::MnistDataset::load(
        Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)),
        Path::new(&format!("{}/train-labels-idx1-ubyte", data_dir)),
    )?;
    let test = mnist::MnistDataset::load(
        Path::new(&format!("{}/t10k-images-idx3-ubyte", data_dir)),
        Path::new(&format!("{}/t10k-labels-idx1-ubyte", data_dir)),
    )?;
    println!("  train={}, test={}", train.len(), test.len());

    let state = Arc::new(Mutex::new(TrainingState::default()));
    let net = Arc::new(Mutex::new(network::AntNetwork::new()));

    // HTTP server
    let addr = format!("0.0.0.0:{}", port);
    let server = Server::http(&addr).unwrap();
    println!("  Dashboard: http://localhost:{}", port);

    let state_http = state.clone();
    let net_http = net.clone();
    let test_arc = Arc::new(test);
    let train_arc = Arc::new(train);
    let test_for_http = test_arc.clone();

    // Serve HTTP in background
    let http_handle = thread::spawn(move || {
        for request in server.incoming_requests() {
            let url = request.url().to_string();
            let cors = Header::from_bytes("Access-Control-Allow-Origin", "*").unwrap();

            match url.as_str() {
                "/" => {
                    let html = include_str!("viz_index.html");
                    let mut resp = Response::from_string(html);
                    resp = resp.with_header(Header::from_bytes("Content-Type", "text/html").unwrap());
                    let _ = request.respond(resp);
                }
                "/api/state" => {
                    let s = state_http.lock().unwrap().clone();
                    let json = serde_json::to_string(&s).unwrap();
                    let resp = Response::from_string(json)
                        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                        .with_header(cors);
                    let _ = request.respond(resp);
                }
                "/api/start" => {
                    {
                        let mut s = state_http.lock().unwrap();
                        s.running = true;
                    }
                    let resp = Response::from_string("{\"ok\":true}")
                        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                        .with_header(cors);
                    let _ = request.respond(resp);
                }
                "/api/stop" => {
                    {
                        let mut s = state_http.lock().unwrap();
                        s.running = false;
                    }
                    let resp = Response::from_string("{\"ok\":true}")
                        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                        .with_header(cors);
                    let _ = request.respond(resp);
                }
                "/api/weights" => {
                    let n = net_http.lock().unwrap();
                    // Sample first 100 weights from each layer
                    let sample = |w: &atomic_neural_transistors::core::ThermalWeightMatrix, n: usize| -> (Vec<i32>, Vec<u8>) {
                        let mut currents = Vec::new();
                        let mut temps = Vec::new();
                        for i in 0..n.min(w.rows * w.cols) {
                            let r = i / w.cols;
                            let c = i % w.cols;
                            let tw = w.get(r, c);
                            currents.push(tw.signal.current());
                            temps.push(tw.temperature);
                        }
                        (currents, temps)
                    };
                    let (c1, t1) = sample(&n.w1, 1000);
                    let (c2, t2) = sample(&n.w2, 1000);
                    let (c3, t3) = sample(&n.w3, 1000);

                    let layers = vec![
                        WeightLayerViz { rows: n.w1.rows, cols: n.w1.cols, sample_currents: c1, sample_temps: t1 },
                        WeightLayerViz { rows: n.w2.rows, cols: n.w2.cols, sample_currents: c2, sample_temps: t2 },
                        WeightLayerViz { rows: n.w3.rows, cols: n.w3.cols, sample_currents: c3, sample_temps: t3 },
                    ];
                    let json = serde_json::to_string(&layers).unwrap();
                    let resp = Response::from_string(json)
                        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                        .with_header(cors);
                    let _ = request.respond(resp);
                }
                "/api/samples" => {
                    // Run 20 test predictions and return them
                    let n = net_http.lock().unwrap();
                    let mut samples = Vec::new();
                    for i in (0..200).step_by(10) {
                        let img = &test_for_http.images[i];
                        let input = encode::encode_image(&img.pixels);
                        let (_, _, out, predicted) = n.forward(&input);
                        let scores: Vec<i32> = out.iter().map(|s| s.current()).collect();
                        samples.push(SamplePrediction {
                            pixels: img.pixels.clone(),
                            label: img.label,
                            predicted: predicted as u8,
                            correct: predicted == img.label as usize,
                            output_scores: scores,
                        });
                    }
                    let json = serde_json::to_string(&samples).unwrap();
                    let resp = Response::from_string(json)
                        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
                        .with_header(cors);
                    let _ = request.respond(resp);
                }
                _ => {
                    let resp = Response::from_string("404").with_status_code(404);
                    let _ = request.respond(resp);
                }
            }
        }
    });

    // Training loop in main thread, controlled by state.running
    let max_cycles = 20;
    loop {
        let running = state.lock().unwrap().running;
        if !running {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }

        let cycle = {
            let s = state.lock().unwrap();
            s.cycle + 1
        };
        if cycle > max_cycles { break; }

        println!("\n=== Cycle {} ===", cycle);
        let cycle_start = Instant::now();

        // Shuffle
        let mut indices: Vec<usize> = (0..train_arc.len()).collect();
        let mut rng = (cycle as u64) * 6364136223846793005 + 1;
        for i in (1..indices.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut batch_correct = 0u64;
        let mut batch_total = 0u64;

        {
            let mut n = net.lock().unwrap();
            n.reset_counters();
        }

        for (step, &idx) in indices.iter().enumerate() {
            // Check if stopped
            if !state.lock().unwrap().running { break; }

            let image = &train_arc.images[idx];
            let input = encode::encode_image(&image.pixels);

            let predicted = {
                let mut n = net.lock().unwrap();
                n.learn(&input, image.label)
            };

            batch_total += 1;
            if predicted == image.label as usize { batch_correct += 1; }

            // Update state every 1000 steps
            if (step + 1) % 1000 == 0 {
                let n = net.lock().unwrap();
                let batch_acc = if batch_total > 0 { batch_correct as f32 / batch_total as f32 } else { 0.0 };
                let (h1, w1, c1, f1) = n.w1.temp_summary();
                let (h2, w2, c2, f2) = n.w2.temp_summary();
                let (h3, w3, c3, f3) = n.w3.temp_summary();

                let mut s = state.lock().unwrap();
                s.cycle = cycle;
                s.step = step + 1;
                s.total_steps = train_arc.len();
                s.train_acc = batch_acc;
                s.thermal = LayerThermal {
                    l1: [h1, w1, c1, f1],
                    l2: [h2, w2, c2, f2],
                    l3: [h3, w3, c3, f3],
                };
                s.transitions = [n.m1.transitions, n.m2.transitions, n.m3.transitions];

                if (step + 1) % 5000 == 0 {
                    batch_correct = 0;
                    batch_total = 0;
                }
            }
        }

        // Evaluate
        println!("  Evaluating...");
        let result = {
            let n = net.lock().unwrap();
            eval::evaluate(&n, &test_arc)
        };
        result.print();

        let cycle_time = cycle_start.elapsed().as_secs_f32();

        // Build sample predictions
        let mut samples = Vec::new();
        {
            let n = net.lock().unwrap();
            for i in (0..100).step_by(5) {
                let img = &test_arc.images[i];
                let input = encode::encode_image(&img.pixels);
                let (_, _, out, predicted) = n.forward(&input);
                let scores: Vec<i32> = out.iter().map(|s| s.current()).collect();
                samples.push(SamplePrediction {
                    pixels: img.pixels.clone(),
                    label: img.label,
                    predicted: predicted as u8,
                    correct: predicted == img.label as usize,
                    output_scores: scores,
                });
            }
        }

        // Update state
        {
            let n = net.lock().unwrap();
            let (h1, w1, c1, f1) = n.w1.temp_summary();
            let (h2, w2, c2, f2) = n.w2.temp_summary();
            let (h3, w3, c3, f3) = n.w3.temp_summary();

            let mut s = state.lock().unwrap();
            s.cycle = cycle;
            s.step = train_arc.len();
            s.total_steps = train_arc.len();
            s.train_acc = n.accuracy();
            s.test_acc = result.accuracy;
            if result.accuracy > s.best_test_acc { s.best_test_acc = result.accuracy; }
            s.per_class_acc = result.per_class;
            s.confusion = result.confusion;
            s.thermal = LayerThermal {
                l1: [h1, w1, c1, f1],
                l2: [h2, w2, c2, f2],
                l3: [h3, w3, c3, f3],
            };
            s.transitions = [n.m1.transitions, n.m2.transitions, n.m3.transitions];
            s.samples = samples;

            let frozen_pct = |t: &[usize; 4]| -> f32 {
                let total = (t[0] + t[1] + t[2] + t[3]) as f32;
                if total > 0.0 { t[3] as f32 / total } else { 0.0 }
            };
            let fp = [
                frozen_pct(&s.thermal.l1),
                frozen_pct(&s.thermal.l2),
                frozen_pct(&s.thermal.l3),
            ];
            let ta = n.accuracy();

            s.history.push(CycleSnapshot {
                cycle,
                train_acc: ta,
                test_acc: result.accuracy,
                thermal_frozen_pct: fp,
            });
        }

        println!("  Cycle {} done in {:.1}s", cycle, cycle_time);
    }

    println!("\nTraining complete. Dashboard still running. Ctrl+C to exit.");
    http_handle.join().unwrap();
    Ok(())
}
