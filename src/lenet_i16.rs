//! LeNet-5 in i16 — signal-native integer implementation.
//!
//! Same architecture as lenet_f32.rs. All arithmetic is integer.
//! Weights are i16. Activations are i16. Multiply-accumulate in i32.
//! Gradients computed in i32, scaled back to i16 for weight updates.
//!
//! The question: can 16-bit integer precision learn spatial features
//! with standard backpropagation?

/// Simple PRNG (same as f32 version)
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range_i16(&mut self, lo: i16, hi: i16) -> i16 {
        let range = (hi - lo + 1) as u64;
        (lo as i64 + (self.next_u64() % range) as i64) as i16
    }
}

// ---------------------------------------------------------------------------
// Integer tensor types
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ITensor1 { pub data: Vec<i16> }
#[derive(Clone)]
pub struct ITensor2 { pub data: Vec<i16>, pub rows: usize, pub cols: usize }
#[derive(Clone)]
pub struct ITensor3 { pub data: Vec<i16>, pub c: usize, pub h: usize, pub w: usize }
#[derive(Clone)]
pub struct ITensor4 { pub data: Vec<i16>, pub n: usize, pub c: usize, pub h: usize, pub w: usize }

// Gradient tensors use i32 for precision during accumulation
#[derive(Clone)]
pub struct GTensor1 { pub data: Vec<i32> }
#[derive(Clone)]
pub struct GTensor2 { pub data: Vec<i32>, pub rows: usize, pub cols: usize }
#[derive(Clone)]
pub struct GTensor3 { pub data: Vec<i32>, pub c: usize, pub h: usize, pub w: usize }
#[derive(Clone)]
pub struct GTensor4 { pub data: Vec<i32>, pub n: usize, pub c: usize, pub h: usize, pub w: usize }

impl ITensor1 {
    fn zeros(len: usize) -> Self { Self { data: vec![0; len] } }
    fn random(len: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..len).map(|_| rng.range_i16(-range, range)).collect() }
    }
}

impl ITensor2 {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0; rows * cols], rows, cols }
    }
    fn random(rows: usize, cols: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..rows * cols).map(|_| rng.range_i16(-range, range)).collect(), rows, cols }
    }
    fn at(&self, r: usize, c: usize) -> i16 { self.data[r * self.cols + c] }
}

impl ITensor3 {
    fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0; c * h * w], c, h, w }
    }
    fn at(&self, c: usize, y: usize, x: usize) -> i16 {
        self.data[c * self.h * self.w + y * self.w + x]
    }
    fn at_mut(&mut self, c: usize, y: usize, x: usize) -> &mut i16 {
        &mut self.data[c * self.h * self.w + y * self.w + x]
    }
}

impl ITensor4 {
    fn random(n: usize, c: usize, h: usize, w: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..n * c * h * w).map(|_| rng.range_i16(-range, range)).collect(), n, c, h, w }
    }
    fn at(&self, n: usize, c: usize, y: usize, x: usize) -> i16 {
        self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x]
    }
}

impl GTensor1 {
    fn zeros(len: usize) -> Self { Self { data: vec![0; len] } }
}
impl GTensor2 {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0; rows * cols], rows, cols }
    }
    fn at_mut(&mut self, r: usize, c: usize) -> &mut i32 { &mut self.data[r * self.cols + c] }
}
impl GTensor3 {
    fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0; c * h * w], c, h, w }
    }
    fn at_mut(&mut self, c: usize, y: usize, x: usize) -> &mut i32 {
        &mut self.data[c * self.h * self.w + y * self.w + x]
    }
}
impl GTensor4 {
    fn zeros(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { data: vec![0; n * c * h * w], n, c, h, w }
    }
    fn at_mut(&mut self, n: usize, c: usize, y: usize, x: usize) -> &mut i32 {
        &mut self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x]
    }
}

// ---------------------------------------------------------------------------
// Fixed-point constants
// ---------------------------------------------------------------------------

/// Scale factor for conv accumulation.
/// Conv1: 25 muls of i16×i16, max product 127×25=3175, sum up to ~80K → fits i32 fine.
/// Shift to keep output in i16 range.
const CONV_SHIFT: i32 = 5;  // divide by 32

/// FC accumulation shift
const FC_SHIFT: i32 = 6;    // divide by 64

/// Learning rate numerator and denominator (integer fraction).
/// Effective lr = LR_NUM / LR_DEN
const LR_NUM: i32 = 1;
const LR_DEN: i32 = 64;     // ~0.016

// ---------------------------------------------------------------------------
// Forward operations
// ---------------------------------------------------------------------------

fn conv2d_forward(input: &ITensor3, kernel: &ITensor4, bias: &ITensor1) -> ITensor3 {
    let out_h = input.h - kernel.h + 1;
    let out_w = input.w - kernel.w + 1;
    let mut output = ITensor3::zeros(kernel.n, out_h, out_w);

    for f in 0..kernel.n {
        for y in 0..out_h {
            for x in 0..out_w {
                let mut sum: i32 = (bias.data[f] as i32) << CONV_SHIFT;
                for c in 0..kernel.c {
                    for ky in 0..kernel.h {
                        for kx in 0..kernel.w {
                            sum += input.at(c, y + ky, x + kx) as i32
                                * kernel.at(f, c, ky, kx) as i32;
                        }
                    }
                }
                *output.at_mut(f, y, x) = (sum >> CONV_SHIFT).clamp(-32000, 32000) as i16;
            }
        }
    }
    output
}

fn relu_forward(input: &ITensor3) -> ITensor3 {
    ITensor3 {
        data: input.data.iter().map(|&v| v.max(0)).collect(),
        c: input.c, h: input.h, w: input.w,
    }
}

fn avgpool_forward(input: &ITensor3) -> ITensor3 {
    let out_h = input.h / 2;
    let out_w = input.w / 2;
    let mut output = ITensor3::zeros(input.c, out_h, out_w);

    for c in 0..input.c {
        for y in 0..out_h {
            for x in 0..out_w {
                let sum = input.at(c, y * 2, x * 2) as i32
                    + input.at(c, y * 2 + 1, x * 2) as i32
                    + input.at(c, y * 2, x * 2 + 1) as i32
                    + input.at(c, y * 2 + 1, x * 2 + 1) as i32;
                *output.at_mut(c, y, x) = (sum / 4) as i16;
            }
        }
    }
    output
}

fn fc_forward(input: &[i16], weights: &ITensor2, bias: &ITensor1) -> Vec<i16> {
    let mut output = vec![0i16; weights.rows];
    for i in 0..weights.rows {
        let mut sum: i32 = (bias.data[i] as i32) << FC_SHIFT;
        for j in 0..weights.cols {
            sum += input[j] as i32 * weights.at(i, j) as i32;
        }
        output[i] = (sum >> FC_SHIFT).clamp(-32000, 32000) as i16;
    }
    output
}

fn relu_vec(input: &[i16]) -> Vec<i16> {
    input.iter().map(|&v| v.max(0)).collect()
}

/// Integer softmax approximation: find max, subtract, use lookup for exp.
/// Returns probabilities as i16 scaled to [0, 1000].
fn softmax_i16(input: &[i16]) -> Vec<i16> {
    let max = *input.iter().max().unwrap_or(&0);
    // Approximate exp with linear segments for small differences
    let mut exps: Vec<i32> = input.iter().map(|&x| {
        let diff = (x as i32 - max as i32).max(-500);
        // exp approximation: max(1, 1000 + diff * 4)
        // This is crude but preserves relative ordering
        (1000 + diff * 4).max(1)
    }).collect();
    let sum: i32 = exps.iter().sum();
    exps.iter().map(|&e| ((e * 1000) / sum).clamp(0, 1000) as i16).collect()
}

// ---------------------------------------------------------------------------
// Backward operations
// ---------------------------------------------------------------------------

fn conv2d_backward(
    input: &ITensor3, kernel: &ITensor4, d_output: &GTensor3,
) -> (GTensor4, GTensor1, GTensor3) {
    let out_h = d_output.h;
    let out_w = d_output.w;
    let mut d_kernel = GTensor4::zeros(kernel.n, kernel.c, kernel.h, kernel.w);
    let mut d_bias = GTensor1::zeros(kernel.n);
    let mut d_input = GTensor3::zeros(input.c, input.h, input.w);

    for f in 0..kernel.n {
        for y in 0..out_h {
            for x in 0..out_w {
                let d_out = d_output.data[f * out_h * out_w + y * out_w + x];
                d_bias.data[f] += d_out;
                for c in 0..kernel.c {
                    for ky in 0..kernel.h {
                        for kx in 0..kernel.w {
                            *d_kernel.at_mut(f, c, ky, kx) += d_out * input.at(c, y + ky, x + kx) as i32;
                            *d_input.at_mut(c, y + ky, x + kx) += d_out * kernel.at(f, c, ky, kx) as i32;
                        }
                    }
                }
            }
        }
    }
    (d_kernel, d_bias, d_input)
}

fn relu_backward_3d(input: &ITensor3, d_output: &GTensor3) -> GTensor3 {
    GTensor3 {
        data: input.data.iter().zip(d_output.data.iter())
            .map(|(&inp, &d)| if inp > 0 { d } else { 0 }).collect(),
        c: input.c, h: input.h, w: input.w,
    }
}

fn avgpool_backward(d_output: &GTensor3, in_h: usize, in_w: usize) -> GTensor3 {
    let mut d_input = GTensor3::zeros(d_output.c, in_h, in_w);
    for c in 0..d_output.c {
        for y in 0..d_output.h {
            for x in 0..d_output.w {
                let d = d_output.data[c * d_output.h * d_output.w + y * d_output.w + x] / 4;
                *d_input.at_mut(c, y * 2, x * 2) = d;
                *d_input.at_mut(c, y * 2 + 1, x * 2) = d;
                *d_input.at_mut(c, y * 2, x * 2 + 1) = d;
                *d_input.at_mut(c, y * 2 + 1, x * 2 + 1) = d;
            }
        }
    }
    d_input
}

fn fc_backward(input: &[i16], weights: &ITensor2, d_output: &[i32]) -> (GTensor2, GTensor1, Vec<i32>) {
    let mut d_weights = GTensor2::zeros(weights.rows, weights.cols);
    let mut d_bias = GTensor1::zeros(weights.rows);
    let mut d_input = vec![0i32; weights.cols];

    for i in 0..weights.rows {
        d_bias.data[i] = d_output[i];
        for j in 0..weights.cols {
            *d_weights.at_mut(i, j) = d_output[i] * input[j] as i32;
            d_input[j] += d_output[i] * weights.at(i, j) as i32;
        }
    }
    (d_weights, d_bias, d_input)
}

fn relu_vec_backward(input: &[i16], d_output: &[i32]) -> Vec<i32> {
    input.iter().zip(d_output.iter())
        .map(|(&inp, &d)| if inp > 0 { d } else { 0 }).collect()
}

/// Cross-entropy gradient in integer: output - target (scaled)
fn cross_entropy_gradient_i16(softmax_out: &[i16], label: usize) -> Vec<i32> {
    // softmax_out is [0, 1000]. Target is 1000 for correct, 0 for others.
    // Gradient = softmax - target, scaled to reasonable range
    softmax_out.iter().enumerate().map(|(i, &s)| {
        let target = if i == label { 1000i32 } else { 0 };
        (s as i32 - target) // range: [-1000, 1000]
    }).collect()
}

// ---------------------------------------------------------------------------
// SGD update: gradient in i32, weights in i16
// ---------------------------------------------------------------------------

fn sgd_update_conv(weights: &mut ITensor4, grads: &GTensor4, bias: &mut ITensor1, d_bias: &GTensor1) {
    for (w, g) in weights.data.iter_mut().zip(grads.data.iter()) {
        // Shift gradient right for learning rate + accumulation scale
        let update = *g >> (LR_SHIFT + CONV_SHIFT);
        *w = (*w as i32 - update).clamp(-500, 500) as i16;
    }
    for (b, g) in bias.data.iter_mut().zip(d_bias.data.iter()) {
        let update = *g >> (LR_SHIFT + CONV_SHIFT);
        *b = (*b as i32 - update).clamp(-500, 500) as i16;
    }
}

fn sgd_update_fc(weights: &mut ITensor2, grads: &GTensor2, bias: &mut ITensor1, d_bias: &GTensor1) {
    for (w, g) in weights.data.iter_mut().zip(grads.data.iter()) {
        let update = *g >> (LR_SHIFT + FC_SHIFT + 2); // extra shift for larger fan-in
        *w = (*w as i32 - update).clamp(-500, 500) as i16;
    }
    for (b, g) in bias.data.iter_mut().zip(d_bias.data.iter()) {
        let update = *g >> (LR_SHIFT + FC_SHIFT);
        *b = (*b as i32 - update).clamp(-500, 500) as i16;
    }
}

// ---------------------------------------------------------------------------
// LeNet-5 i16
// ---------------------------------------------------------------------------

pub struct LeNetI16 {
    conv1_k: ITensor4,
    conv1_b: ITensor1,
    conv2_k: ITensor4,
    conv2_b: ITensor1,
    fc1_w: ITensor2,
    fc1_b: ITensor1,
    fc2_w: ITensor2,
    fc2_b: ITensor1,
    fc3_w: ITensor2,
    fc3_b: ITensor1,
}

struct IForwardCache {
    input: ITensor3,
    conv1_out: ITensor3,
    relu1_out: ITensor3,
    pool1_out: ITensor3,
    conv2_out: ITensor3,
    relu2_out: ITensor3,
    pool2_out: ITensor3,
    fc1_in: Vec<i16>,
    fc1_out: Vec<i16>,
    relu3_out: Vec<i16>,
    fc2_out: Vec<i16>,
    relu4_out: Vec<i16>,
    softmax_out: Vec<i16>,
}

impl LeNetI16 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        // Small integer init: range proportional to 1/sqrt(fan_in)
        // Conv1: fan_in=25, sqrt=5, range=127/5=25
        // Conv2: fan_in=150, sqrt=12, range=127/12=10
        // FC: range scales down with fan_in
        Self {
            conv1_k: ITensor4::random(6, 1, 5, 5, &mut rng, 25),
            conv1_b: ITensor1::zeros(6),
            conv2_k: ITensor4::random(16, 6, 5, 5, &mut rng, 10),
            conv2_b: ITensor1::zeros(16),
            fc1_w: ITensor2::random(120, 256, &mut rng, 5),
            fc1_b: ITensor1::zeros(120),
            fc2_w: ITensor2::random(84, 120, &mut rng, 8),
            fc2_b: ITensor1::zeros(84),
            fc3_w: ITensor2::random(10, 84, &mut rng, 10),
            fc3_b: ITensor1::zeros(10),
        }
    }

    fn forward_cached(&self, image: &[u8]) -> (usize, IForwardCache) {
        // Input: pixel values scaled to [0, 127] (half i16 range)
        let mut input = ITensor3::zeros(1, 28, 28);
        for y in 0..28 {
            for x in 0..28 {
                *input.at_mut(0, y, x) = (image[y * 28 + x] as i16) / 2;
            }
        }

        let conv1_out = conv2d_forward(&input, &self.conv1_k, &self.conv1_b);
        let relu1_out = relu_forward(&conv1_out);
        let pool1_out = avgpool_forward(&relu1_out);

        let conv2_out = conv2d_forward(&pool1_out, &self.conv2_k, &self.conv2_b);
        let relu2_out = relu_forward(&conv2_out);
        let pool2_out = avgpool_forward(&relu2_out);

        let fc1_in = pool2_out.data.clone();
        let fc1_out = fc_forward(&fc1_in, &self.fc1_w, &self.fc1_b);
        let relu3_out = relu_vec(&fc1_out);
        let fc2_out = fc_forward(&relu3_out, &self.fc2_w, &self.fc2_b);
        let relu4_out = relu_vec(&fc2_out);
        let fc3_out = fc_forward(&relu4_out, &self.fc3_w, &self.fc3_b);
        let softmax_out = softmax_i16(&fc3_out);

        let prediction = softmax_out.iter().enumerate()
            .max_by_key(|(_, s)| *s)
            .map(|(i, _)| i).unwrap_or(0);

        let cache = IForwardCache {
            input, conv1_out, relu1_out, pool1_out,
            conv2_out, relu2_out, pool2_out,
            fc1_in, fc1_out, relu3_out, fc2_out, relu4_out,
            softmax_out,
        };

        (prediction, cache)
    }

    pub fn predict(&self, image: &[u8]) -> usize {
        self.forward_cached(image).0
    }

    pub fn train_one(&mut self, image: &[u8], label: usize) -> (usize, bool) {
        let (prediction, cache) = self.forward_cached(image);

        // Loss gradient
        let d_fc3_out = cross_entropy_gradient_i16(&cache.softmax_out, label);

        // FC3 backward
        let (d_fc3_w, d_fc3_b, d_relu4) = fc_backward(&cache.relu4_out, &self.fc3_w, &d_fc3_out);
        let d_fc2_out = relu_vec_backward(&cache.fc2_out, &d_relu4);

        // FC2 backward
        let (d_fc2_w, d_fc2_b, d_relu3) = fc_backward(&cache.relu3_out, &self.fc2_w, &d_fc2_out);
        let d_fc1_out = relu_vec_backward(&cache.fc1_out, &d_relu3);

        // FC1 backward
        let (d_fc1_w, d_fc1_b, d_flat) = fc_backward(&cache.fc1_in, &self.fc1_w, &d_fc1_out);

        // Unflatten
        let d_pool2 = GTensor3 { data: d_flat, c: cache.pool2_out.c, h: cache.pool2_out.h, w: cache.pool2_out.w };

        let d_relu2 = avgpool_backward(&d_pool2, cache.relu2_out.h, cache.relu2_out.w);
        let d_conv2_out = relu_backward_3d(&cache.conv2_out, &d_relu2);
        let (d_conv2_k, d_conv2_b, d_pool1) = conv2d_backward(&cache.pool1_out, &self.conv2_k, &d_conv2_out);

        let d_relu1 = avgpool_backward(&d_pool1, cache.relu1_out.h, cache.relu1_out.w);
        let d_conv1_out = relu_backward_3d(&cache.conv1_out, &d_relu1);
        let (d_conv1_k, d_conv1_b, _) = conv2d_backward(&cache.input, &self.conv1_k, &d_conv1_out);

        // SGD update
        sgd_update_conv(&mut self.conv1_k, &d_conv1_k, &mut self.conv1_b, &d_conv1_b);
        sgd_update_conv(&mut self.conv2_k, &d_conv2_k, &mut self.conv2_b, &d_conv2_b);
        sgd_update_fc(&mut self.fc1_w, &d_fc1_w, &mut self.fc1_b, &d_fc1_b);
        sgd_update_fc(&mut self.fc2_w, &d_fc2_w, &mut self.fc2_b, &d_fc2_b);
        sgd_update_fc(&mut self.fc3_w, &d_fc3_w, &mut self.fc3_b, &d_fc3_b);

        (prediction, prediction == label)
    }

    pub fn weight_stats(&self) -> I16WeightStats {
        let all: Vec<i16> = [
            &self.conv1_k.data[..], &self.conv1_b.data[..],
            &self.conv2_k.data[..], &self.conv2_b.data[..],
            &self.fc1_w.data[..], &self.fc1_b.data[..],
            &self.fc2_w.data[..], &self.fc2_b.data[..],
            &self.fc3_w.data[..], &self.fc3_b.data[..],
        ].iter().flat_map(|s| s.iter().copied()).collect();

        let n = all.len() as f64;
        let mean = all.iter().map(|&x| x as f64).sum::<f64>() / n;
        let std = (all.iter().map(|x| { let d = *x as f64 - mean; d * d }).sum::<f64>() / n).sqrt();

        I16WeightStats {
            count: all.len(),
            bytes: all.len() * 2,
            mean, std,
        }
    }
}

pub struct I16WeightStats {
    pub count: usize,
    pub bytes: usize,
    pub mean: f64,
    pub std: f64,
}
