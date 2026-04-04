//! LeNet-5 in f32 — the baseline implementation.
//!
//! Pure Rust, no dependencies, no SIMD. Clear, readable, correct.
//! Architecture: Conv1(6×5×5) → ReLU → Pool → Conv2(16×6×5×5) → ReLU → Pool → FC120 → FC84 → FC10

/// Simple PRNG for weight initialization (xorshift64)
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform random in [-range, range]
    fn uniform(&mut self, range: f32) -> f32 {
        let u = (self.next_u64() & 0xFFFFFFFF) as f64 / 0xFFFFFFFF_u64 as f64;
        (u as f32 * 2.0 - 1.0) * range
    }
}

// ---------------------------------------------------------------------------
// Tensor types — simple owned 1D/2D/3D/4D arrays
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Tensor1 { pub data: Vec<f32> }
#[derive(Clone)]
pub struct Tensor2 { pub data: Vec<f32>, pub rows: usize, pub cols: usize }
#[derive(Clone)]
pub struct Tensor3 { pub data: Vec<f32>, pub c: usize, pub h: usize, pub w: usize }
#[derive(Clone)]
pub struct Tensor4 { pub data: Vec<f32>, pub n: usize, pub c: usize, pub h: usize, pub w: usize }

impl Tensor1 {
    fn zeros(len: usize) -> Self { Self { data: vec![0.0; len] } }
    fn random(len: usize, rng: &mut Rng, range: f32) -> Self {
        Self { data: (0..len).map(|_| rng.uniform(range)).collect() }
    }
    fn len(&self) -> usize { self.data.len() }
}

impl Tensor2 {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }
    fn random(rows: usize, cols: usize, rng: &mut Rng, range: f32) -> Self {
        Self { data: (0..rows * cols).map(|_| rng.uniform(range)).collect(), rows, cols }
    }
    fn at(&self, r: usize, c: usize) -> f32 { self.data[r * self.cols + c] }
    fn at_mut(&mut self, r: usize, c: usize) -> &mut f32 { &mut self.data[r * self.cols + c] }
}

impl Tensor3 {
    fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0.0; c * h * w], c, h, w }
    }
    fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[c * self.h * self.w + y * self.w + x]
    }
    fn at_mut(&mut self, c: usize, y: usize, x: usize) -> &mut f32 {
        &mut self.data[c * self.h * self.w + y * self.w + x]
    }
    fn flat_len(&self) -> usize { self.c * self.h * self.w }
}

impl Tensor4 {
    fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0.0; n * c * h * w], n, c, h, w }
    }
    fn random(n: usize, c: usize, h: usize, w: usize, rng: &mut Rng, range: f32) -> Self {
        Self { data: (0..n * c * h * w).map(|_| rng.uniform(range)).collect(), n, c, h, w }
    }
    fn at(&self, n: usize, c: usize, y: usize, x: usize) -> f32 {
        self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x]
    }
    fn at_mut(&mut self, n: usize, c: usize, y: usize, x: usize) -> &mut f32 {
        &mut self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x]
    }
}

// ---------------------------------------------------------------------------
// Layer operations — forward and backward
// ---------------------------------------------------------------------------

/// Conv2D forward: input(C_in, H, W) × kernel(C_out, C_in, KH, KW) + bias → output(C_out, H-KH+1, W-KW+1)
fn conv2d_forward(input: &Tensor3, kernel: &Tensor4, bias: &Tensor1) -> Tensor3 {
    let out_h = input.h - kernel.h + 1;
    let out_w = input.w - kernel.w + 1;
    let mut output = Tensor3::zeros(kernel.n, out_h, out_w);

    for f in 0..kernel.n {
        for y in 0..out_h {
            for x in 0..out_w {
                let mut sum = bias.data[f];
                for c in 0..kernel.c {
                    for ky in 0..kernel.h {
                        for kx in 0..kernel.w {
                            sum += input.at(c, y + ky, x + kx) * kernel.at(f, c, ky, kx);
                        }
                    }
                }
                *output.at_mut(f, y, x) = sum;
            }
        }
    }
    output
}

/// Conv2D backward: compute gradients for kernel, bias, and input
fn conv2d_backward(
    input: &Tensor3, kernel: &Tensor4, d_output: &Tensor3,
) -> (Tensor4, Tensor1, Tensor3) {
    let out_h = d_output.h;
    let out_w = d_output.w;
    let mut d_kernel = Tensor4::zeros(kernel.n, kernel.c, kernel.h, kernel.w);
    let mut d_bias = Tensor1::zeros(kernel.n);
    let mut d_input = Tensor3::zeros(input.c, input.h, input.w);

    for f in 0..kernel.n {
        for y in 0..out_h {
            for x in 0..out_w {
                let d_out = d_output.at(f, y, x);
                d_bias.data[f] += d_out;
                for c in 0..kernel.c {
                    for ky in 0..kernel.h {
                        for kx in 0..kernel.w {
                            *d_kernel.at_mut(f, c, ky, kx) += d_out * input.at(c, y + ky, x + kx);
                            *d_input.at_mut(c, y + ky, x + kx) += d_out * kernel.at(f, c, ky, kx);
                        }
                    }
                }
            }
        }
    }
    (d_kernel, d_bias, d_input)
}

/// ReLU forward (in-place semantics, returns new tensor)
fn relu_forward(input: &Tensor3) -> Tensor3 {
    Tensor3 {
        data: input.data.iter().map(|&v| v.max(0.0)).collect(),
        c: input.c, h: input.h, w: input.w,
    }
}

/// ReLU backward: zero gradient where input was <= 0
fn relu_backward(input: &Tensor3, d_output: &Tensor3) -> Tensor3 {
    Tensor3 {
        data: input.data.iter().zip(d_output.data.iter())
            .map(|(&inp, &d)| if inp > 0.0 { d } else { 0.0 }).collect(),
        c: input.c, h: input.h, w: input.w,
    }
}

/// Average pool 2×2, stride 2
fn avgpool_forward(input: &Tensor3) -> Tensor3 {
    let out_h = input.h / 2;
    let out_w = input.w / 2;
    let mut output = Tensor3::zeros(input.c, out_h, out_w);

    for c in 0..input.c {
        for y in 0..out_h {
            for x in 0..out_w {
                let sum = input.at(c, y * 2, x * 2)
                    + input.at(c, y * 2 + 1, x * 2)
                    + input.at(c, y * 2, x * 2 + 1)
                    + input.at(c, y * 2 + 1, x * 2 + 1);
                *output.at_mut(c, y, x) = sum / 4.0;
            }
        }
    }
    output
}

/// Average pool backward
fn avgpool_backward(d_output: &Tensor3, in_h: usize, in_w: usize) -> Tensor3 {
    let mut d_input = Tensor3::zeros(d_output.c, in_h, in_w);
    for c in 0..d_output.c {
        for y in 0..d_output.h {
            for x in 0..d_output.w {
                let d = d_output.at(c, y, x) / 4.0;
                *d_input.at_mut(c, y * 2, x * 2) = d;
                *d_input.at_mut(c, y * 2 + 1, x * 2) = d;
                *d_input.at_mut(c, y * 2, x * 2 + 1) = d;
                *d_input.at_mut(c, y * 2 + 1, x * 2 + 1) = d;
            }
        }
    }
    d_input
}

/// FC forward: input(flat) × weights(out, in) + bias → output
fn fc_forward(input: &[f32], weights: &Tensor2, bias: &Tensor1) -> Vec<f32> {
    let mut output = vec![0.0f32; weights.rows];
    for i in 0..weights.rows {
        let mut sum = bias.data[i];
        for j in 0..weights.cols {
            sum += input[j] * weights.at(i, j);
        }
        output[i] = sum;
    }
    output
}

/// FC backward: compute gradients for weights, bias, and input
fn fc_backward(input: &[f32], weights: &Tensor2, d_output: &[f32]) -> (Tensor2, Tensor1, Vec<f32>) {
    let mut d_weights = Tensor2::zeros(weights.rows, weights.cols);
    let mut d_bias = Tensor1::zeros(weights.rows);
    let mut d_input = vec![0.0f32; weights.cols];

    for i in 0..weights.rows {
        d_bias.data[i] = d_output[i];
        for j in 0..weights.cols {
            *d_weights.at_mut(i, j) = d_output[i] * input[j];
            d_input[j] += d_output[i] * weights.at(i, j);
        }
    }
    (d_weights, d_bias, d_input)
}

/// ReLU forward on flat vector
fn relu_vec(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&v| v.max(0.0)).collect()
}

/// ReLU backward on flat vector
fn relu_vec_backward(input: &[f32], d_output: &[f32]) -> Vec<f32> {
    input.iter().zip(d_output.iter())
        .map(|(&inp, &d)| if inp > 0.0 { d } else { 0.0 })
        .collect()
}

/// Softmax
fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Cross-entropy loss gradient (softmax output - one-hot target)
fn cross_entropy_gradient(softmax_output: &[f32], label: usize) -> Vec<f32> {
    let mut grad = softmax_output.to_vec();
    grad[label] -= 1.0;
    grad
}

// ---------------------------------------------------------------------------
// LeNet-5 network
// ---------------------------------------------------------------------------

pub struct LeNetF32 {
    // Conv1: 6 filters × 1 channel × 5×5
    conv1_k: Tensor4,
    conv1_b: Tensor1,
    // Conv2: 16 filters × 6 channels × 5×5
    conv2_k: Tensor4,
    conv2_b: Tensor1,
    // FC layers
    fc1_w: Tensor2,
    fc1_b: Tensor1,
    fc2_w: Tensor2,
    fc2_b: Tensor1,
    fc3_w: Tensor2,
    fc3_b: Tensor1,
}

/// Cached activations for backward pass
struct ForwardCache {
    // Input to each layer (pre-activation)
    input: Tensor3,
    conv1_out: Tensor3,
    relu1_out: Tensor3,
    pool1_out: Tensor3,
    conv2_out: Tensor3,
    relu2_out: Tensor3,
    pool2_out: Tensor3,
    fc1_in: Vec<f32>,
    fc1_out: Vec<f32>,
    relu3_out: Vec<f32>,
    fc2_out: Vec<f32>,
    relu4_out: Vec<f32>,
    fc3_out: Vec<f32>,
    softmax_out: Vec<f32>,
}

impl LeNetF32 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        // Xavier-like initialization: range = sqrt(6 / (fan_in + fan_out))
        let conv1_fan = 1 * 25 + 6 * 25; // fan_in=25, fan_out=150
        let conv1_range = (6.0 / conv1_fan as f32).sqrt();
        let conv2_fan = 6 * 25 + 16 * 25;
        let conv2_range = (6.0 / conv2_fan as f32).sqrt();

        Self {
            conv1_k: Tensor4::random(6, 1, 5, 5, &mut rng, conv1_range),
            conv1_b: Tensor1::zeros(6),
            conv2_k: Tensor4::random(16, 6, 5, 5, &mut rng, conv2_range),
            conv2_b: Tensor1::zeros(16),
            fc1_w: Tensor2::random(120, 256, &mut rng, (6.0 / 376.0_f32).sqrt()),
            fc1_b: Tensor1::zeros(120),
            fc2_w: Tensor2::random(84, 120, &mut rng, (6.0 / 204.0_f32).sqrt()),
            fc2_b: Tensor1::zeros(84),
            fc3_w: Tensor2::random(10, 84, &mut rng, (6.0 / 94.0_f32).sqrt()),
            fc3_b: Tensor1::zeros(10),
        }
    }

    /// Forward pass, returning prediction and cached activations for backward.
    fn forward_cached(&self, image: &[u8]) -> (usize, ForwardCache) {
        // Input: 1×28×28, pixels normalized to [0, 1]
        let mut input = Tensor3::zeros(1, 28, 28);
        for y in 0..28 {
            for x in 0..28 {
                *input.at_mut(0, y, x) = image[y * 28 + x] as f32 / 255.0;
            }
        }

        // Conv1 → ReLU → Pool
        let conv1_out = conv2d_forward(&input, &self.conv1_k, &self.conv1_b); // 6×24×24
        let relu1_out = relu_forward(&conv1_out);
        let pool1_out = avgpool_forward(&relu1_out); // 6×12×12

        // Conv2 → ReLU → Pool
        let conv2_out = conv2d_forward(&pool1_out, &self.conv2_k, &self.conv2_b); // 16×8×8
        let relu2_out = relu_forward(&conv2_out);
        let pool2_out = avgpool_forward(&relu2_out); // 16×4×4

        // Flatten
        let fc1_in = pool2_out.data.clone(); // 256 elements (16×4×4)

        // FC1 → ReLU
        let fc1_out = fc_forward(&fc1_in, &self.fc1_w, &self.fc1_b);
        let relu3_out = relu_vec(&fc1_out);

        // FC2 → ReLU
        let fc2_out = fc_forward(&relu3_out, &self.fc2_w, &self.fc2_b);
        let relu4_out = relu_vec(&fc2_out);

        // FC3
        let fc3_out = fc_forward(&relu4_out, &self.fc3_w, &self.fc3_b);
        let softmax_out = softmax(&fc3_out);

        let prediction = softmax_out.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        let cache = ForwardCache {
            input, conv1_out, relu1_out, pool1_out,
            conv2_out, relu2_out, pool2_out,
            fc1_in, fc1_out, relu3_out, fc2_out, relu4_out,
            fc3_out, softmax_out,
        };

        (prediction, cache)
    }

    /// Predict without caching (for eval).
    pub fn predict(&self, image: &[u8]) -> usize {
        self.forward_cached(image).0
    }

    /// Train on one sample. Returns (predicted, correct).
    pub fn train_one(&mut self, image: &[u8], label: usize, lr: f32) -> (usize, bool) {
        let (prediction, cache) = self.forward_cached(image);

        // Loss gradient
        let d_fc3_out = cross_entropy_gradient(&cache.softmax_out, label);

        // FC3 backward
        let (d_fc3_w, d_fc3_b, d_relu4) = fc_backward(&cache.relu4_out, &self.fc3_w, &d_fc3_out);
        let d_fc2_out = relu_vec_backward(&cache.fc2_out, &d_relu4);

        // FC2 backward
        let (d_fc2_w, d_fc2_b, d_relu3) = fc_backward(&cache.relu3_out, &self.fc2_w, &d_fc2_out);
        let d_fc1_out = relu_vec_backward(&cache.fc1_out, &d_relu3);

        // FC1 backward
        let (d_fc1_w, d_fc1_b, d_flat) = fc_backward(&cache.fc1_in, &self.fc1_w, &d_fc1_out);

        // Unflatten gradient back to 16×4×4
        let d_pool2 = Tensor3 { data: d_flat, c: cache.pool2_out.c, h: cache.pool2_out.h, w: cache.pool2_out.w };

        // Pool2 backward
        let d_relu2 = avgpool_backward(&d_pool2, cache.relu2_out.h, cache.relu2_out.w);
        let d_conv2_out = relu_backward(&cache.conv2_out, &d_relu2);

        // Conv2 backward
        let (d_conv2_k, d_conv2_b, d_pool1) = conv2d_backward(&cache.pool1_out, &self.conv2_k, &d_conv2_out);

        // Pool1 backward
        let d_relu1 = avgpool_backward(&d_pool1, cache.relu1_out.h, cache.relu1_out.w);
        let d_conv1_out = relu_backward(&cache.conv1_out, &d_relu1);

        // Conv1 backward
        let (d_conv1_k, d_conv1_b, _) = conv2d_backward(&cache.input, &self.conv1_k, &d_conv1_out);

        // SGD update
        sgd_update(&mut self.conv1_k.data, &d_conv1_k.data, lr);
        sgd_update(&mut self.conv1_b.data, &d_conv1_b.data, lr);
        sgd_update(&mut self.conv2_k.data, &d_conv2_k.data, lr);
        sgd_update(&mut self.conv2_b.data, &d_conv2_b.data, lr);
        sgd_update(&mut self.fc1_w.data, &d_fc1_w.data, lr);
        sgd_update(&mut self.fc1_b.data, &d_fc1_b.data, lr);
        sgd_update(&mut self.fc2_w.data, &d_fc2_w.data, lr);
        sgd_update(&mut self.fc2_b.data, &d_fc2_b.data, lr);
        sgd_update(&mut self.fc3_w.data, &d_fc3_w.data, lr);
        sgd_update(&mut self.fc3_b.data, &d_fc3_b.data, lr);

        (prediction, prediction == label)
    }

    /// Weight statistics for a layer.
    pub fn weight_stats(&self) -> WeightStats {
        let all: Vec<f32> = [
            &self.conv1_k.data[..], &self.conv1_b.data[..],
            &self.conv2_k.data[..], &self.conv2_b.data[..],
            &self.fc1_w.data[..], &self.fc1_b.data[..],
            &self.fc2_w.data[..], &self.fc2_b.data[..],
            &self.fc3_w.data[..], &self.fc3_b.data[..],
        ].iter().flat_map(|s| s.iter().copied()).collect();

        let n = all.len() as f32;
        let mean = all.iter().sum::<f32>() / n;
        let std = (all.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();

        WeightStats {
            count: all.len(),
            bytes: all.len() * 4,
            mean, std,
        }
    }
}

fn sgd_update(weights: &mut [f32], gradients: &[f32], lr: f32) {
    for (w, g) in weights.iter_mut().zip(gradients.iter()) {
        *w -= lr * g;
    }
}

pub struct WeightStats {
    pub count: usize,
    pub bytes: usize,
    pub mean: f32,
    pub std: f32,
}
