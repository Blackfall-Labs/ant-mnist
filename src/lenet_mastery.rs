//! LeNet-5 with frozen random projections + mastery learning.
//!
//! Conv1, Conv2, FC1, FC2: frozen random i16 weights (never updated).
//! FC3 (84 → 10): mastery learning with participation-gated pressure.
//!
//! The hypothesis: random spatial filters in conv layers create natural
//! class separation. Mastery learning on the output layer refines alignment.
//! No backpropagation. No gradient computation through frozen layers.

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_u8(&mut self) -> u8 { (self.next_u64() & 0xFF) as u8 }
    fn range_i8(&mut self) -> i8 { (self.next_u64() % 3) as i8 - 1 }
    fn range_i16(&mut self, lo: i16, hi: i16) -> i16 {
        let range = (hi - lo + 1) as u64;
        (lo as i64 + (self.next_u64() % range) as i64) as i16
    }
}

// ---------------------------------------------------------------------------
// Minimal tensor types (forward-only for frozen layers)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct T1 { data: Vec<i16> }
#[derive(Clone)]
struct T2 { data: Vec<i16>, rows: usize, cols: usize }
#[derive(Clone)]
struct T3 { data: Vec<i16>, c: usize, h: usize, w: usize }
#[derive(Clone)]
struct T4 { data: Vec<i16>, n: usize, c: usize, h: usize, w: usize }

impl T1 {
    fn zeros(len: usize) -> Self { Self { data: vec![0; len] } }
    fn random(len: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..len).map(|_| rng.range_i16(-range, range)).collect() }
    }
}
impl T2 {
    fn random(rows: usize, cols: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..rows * cols).map(|_| rng.range_i16(-range, range)).collect(), rows, cols }
    }
    fn at(&self, r: usize, c: usize) -> i16 { self.data[r * self.cols + c] }
}
impl T3 {
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
impl T4 {
    fn random(n: usize, c: usize, h: usize, w: usize, rng: &mut Rng, range: i16) -> Self {
        Self { data: (0..n * c * h * w).map(|_| rng.range_i16(-range, range)).collect(), n, c, h, w }
    }
    fn at(&self, n: usize, c: usize, y: usize, x: usize) -> i16 {
        self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x]
    }
}

// ---------------------------------------------------------------------------
// Forward-only operations (no backward needed for frozen layers)
// ---------------------------------------------------------------------------

const CONV_SHIFT: i32 = 5;
const FC_SHIFT: i32 = 6;

fn conv2d(input: &T3, kernel: &T4, bias: &T1) -> T3 {
    let out_h = input.h - kernel.h + 1;
    let out_w = input.w - kernel.w + 1;
    let mut output = T3::zeros(kernel.n, out_h, out_w);
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

fn relu(input: &T3) -> T3 {
    T3 { data: input.data.iter().map(|&v| v.max(0)).collect(), c: input.c, h: input.h, w: input.w }
}

fn avgpool(input: &T3) -> T3 {
    let oh = input.h / 2;
    let ow = input.w / 2;
    let mut out = T3::zeros(input.c, oh, ow);
    for c in 0..input.c {
        for y in 0..oh {
            for x in 0..ow {
                let s = input.at(c, y*2, x*2) as i32 + input.at(c, y*2+1, x*2) as i32
                    + input.at(c, y*2, x*2+1) as i32 + input.at(c, y*2+1, x*2+1) as i32;
                *out.at_mut(c, y, x) = (s / 4) as i16;
            }
        }
    }
    out
}

fn fc(input: &[i16], w: &T2, bias: &T1) -> Vec<i16> {
    let mut out = vec![0i16; w.rows];
    for i in 0..w.rows {
        let mut sum: i32 = (bias.data[i] as i32) << FC_SHIFT;
        for j in 0..w.cols {
            sum += input[j] as i32 * w.at(i, j) as i32;
        }
        out[i] = (sum >> FC_SHIFT).clamp(-32000, 32000) as i16;
    }
    out
}

fn relu_vec(v: &[i16]) -> Vec<i16> { v.iter().map(|&x| x.max(0)).collect() }
fn relu_vec_i16(v: &[i16]) -> Vec<i16> { v.iter().map(|&x| x.max(0)).collect() }

// ---------------------------------------------------------------------------
// Mastery learning layer (the only plastic layer)
// ---------------------------------------------------------------------------

/// Mastery weight: P×M with thermal monitoring.
///
/// Polarity = structural direction (-1, 0, +1)
/// Magnitude = strength (0-255)
/// Temperature = stability tracker (255=HOT, 0=COLD) — monitoring only, does NOT gate learning
/// Pressure = accumulated learning signal
#[derive(Clone)]
pub struct MasteryWeight {
    pub polarity: i8,      // -1, 0, +1
    pub magnitude: u8,     // 0-255
    pub temperature: u8,   // 255=HOT, 0=COLD (monitoring, not gating)
    pub pressure: i16,     // accumulated directional pressure
}

impl MasteryWeight {
    fn zero() -> Self {
        Self { polarity: 0, magnitude: 0, temperature: 255, pressure: 0 }
    }

    /// P × M
    pub fn current(&self) -> i16 {
        self.polarity as i16 * self.magnitude as i16
    }

    /// Cool toward stability. Call on correct classifications.
    fn cool(&mut self, amount: u8) {
        self.temperature = self.temperature.saturating_sub(amount);
    }

    /// Warm back up. Call on errors involving this weight.
    fn warm(&mut self, amount: u8) {
        self.temperature = self.temperature.saturating_add(amount);
    }
}

pub struct MasteryLayer {
    pub weights: Vec<MasteryWeight>,  // rows × cols
    bias: Vec<i16>,
    pub rows: usize,
    pub cols: usize,
    pressure_threshold: i16,
    magnitude_step: u8,
    pub steps: u64,
    pub transitions: u64,
}

impl MasteryLayer {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            weights: (0..rows * cols).map(|_| MasteryWeight::zero()).collect(),
            bias: vec![0i16; rows],
            rows, cols,
            pressure_threshold: 8,
            magnitude_step: 5,
            steps: 0,
            transitions: 0,
        }
    }

    /// Frozen random: initialized with random weights, pressure_threshold set
    /// impossibly high so no learning occurs. Forward pass works normally.
    pub fn frozen_random(rows: usize, cols: usize, rng: &mut Rng) -> Self {
        let mut layer = Self::random_init(rows, cols, rng);
        layer.pressure_threshold = i16::MAX; // never transitions
        layer
    }

    pub fn random_init(rows: usize, cols: usize, rng: &mut Rng) -> Self {
        let weights = (0..rows * cols).map(|_| {
            let p = rng.range_i8();
            if p == 0 {
                MasteryWeight::zero()
            } else {
                MasteryWeight {
                    polarity: p,
                    magnitude: (rng.next_u8() % 30) + 10,
                    temperature: 255,
                    pressure: 0,
                }
            }
        }).collect();
        Self {
            weights,
            bias: vec![0i16; rows],
            rows, cols,
            pressure_threshold: 8,
            magnitude_step: 5,
            steps: 0,
            transitions: 0,
        }
    }

    /// Forward: clean dot product. Temperature does NOT participate here.
    pub fn forward(&self, input: &[i16]) -> Vec<i16> {
        let mut output = vec![0i16; self.rows];
        for i in 0..self.rows {
            let mut sum: i32 = self.bias[i] as i32 * 64;
            for j in 0..self.cols {
                sum += input[j] as i32 * self.weights[i * self.cols + j].current() as i32;
            }
            output[i] = (sum / self.cols as i32).clamp(-32000, 32000) as i16;
        }
        output
    }

    /// Mastery learning step.
    ///
    /// Only updates weights feeding the target class and the predicted class.
    /// Participation-gated: only active inputs contribute.
    /// Pressure accumulates over samples. Transitions happen when pressure
    /// crosses threshold.
    pub fn learn(&mut self, input: &[i16], output: &[i16], label: usize) {
        self.steps += 1;

        let predicted = output.iter().enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i).unwrap_or(0);

        // Participation threshold: top 25% of input activity
        let max_input = input.iter().map(|x| x.unsigned_abs() as u32).max().unwrap_or(1).max(1);
        let activity_threshold = (max_input / 4) as u16;

        // Only update two rows: the correct class and the predicted class
        let rows_to_update: Vec<(usize, i16)> = if predicted == label {
            // Correct: reinforce the correct class
            vec![(label, 1)]
        } else {
            // Wrong: strengthen correct, weaken predicted
            vec![(label, 1), (predicted, -1)]
        };

        let is_correct = predicted == label;

        for &(row, direction) in &rows_to_update {
            for j in 0..self.cols {
                let input_abs = input[j].unsigned_abs() as u16;
                if input_abs <= activity_threshold { continue; }

                let input_sign = if input[j] > 0 { 1i16 } else { -1 };

                // Pressure: direction × input_sign × activity_strength
                let activity = ((input_abs - activity_threshold) as i32 * 8
                    / max_input as i32).max(1) as i16;
                let raw_pressure = direction * input_sign * activity;

                let idx = row * self.cols + j;
                let w = &mut self.weights[idx];

                // Mastery pressure (no thermal scaling — full signal)
                w.pressure = w.pressure.saturating_add(raw_pressure);

                // Thermal tracking: cool on correct, warm on error
                if is_correct {
                    w.cool(1);
                } else {
                    w.warm(3);
                }

                // Threshold gate
                if w.pressure.abs() >= self.pressure_threshold {
                    let needed = if w.pressure > 0 { 1i8 } else { -1 };

                    // Thermal-adaptive step: hot=big, cold=precise
                    let step = if w.temperature >= 192 { 5u8 }
                        else if w.temperature >= 128 { 3 }
                        else if w.temperature >= 64 { 2 }
                        else { 1 };

                    if w.polarity == needed {
                        w.magnitude = w.magnitude.saturating_add(step);
                    } else if w.polarity == -needed {
                        if w.magnitude <= step {
                            w.magnitude = 0;
                            w.polarity = 0;
                        } else {
                            w.magnitude -= step;
                        }
                    } else {
                        w.polarity = needed;
                        w.magnitude = step;
                    }
                    self.transitions += 1;
                    w.pressure = 0;
                }
            }

            // Bias update
            if direction > 0 {
                self.bias[row] = (self.bias[row] as i32 + 1).min(500) as i16;
            } else {
                self.bias[row] = (self.bias[row] as i32 - 1).max(-500) as i16;
            }
        }
    }

    fn apply_transition(&mut self, idx: usize, needed: i8) {
        let w = &mut self.weights[idx];
        if w.polarity == needed {
            // Same direction: strengthen
            w.magnitude = w.magnitude.saturating_add(self.magnitude_step);
        } else if w.polarity == -needed {
            // Opposite: weaken first
            if w.magnitude <= self.magnitude_step {
                w.magnitude = 0;
                w.polarity = 0;
            } else {
                w.magnitude -= self.magnitude_step;
            }
        } else {
            // Zero: initialize in needed direction
            w.polarity = needed;
            w.magnitude = self.magnitude_step;
        }
        self.transitions += 1;
    }

    /// Learn as a hidden layer: use output layer weights to derive error signal.
    ///
    /// For each hidden neuron j:
    ///   If output_weights[label][j] is positive → this neuron SHOULD fire for correct class
    ///   If output_weights[predicted][j] is positive → this neuron contributed to wrong answer
    ///
    /// Direction = sign(output_weight[label][j]) - sign(output_weight[predicted][j])
    pub fn learn_hidden(
        &mut self,
        input: &[i16],        // 120-dim (from frozen FC1)
        raw_output: &[i16],   // 84-dim (pre-relu hidden output)
        output_layer: &MasteryLayer,  // the output layer (10×84)
        label: usize,
        predicted: usize,
    ) {
        if predicted == label { return; } // correct — don't mess with hidden
        self.steps += 1;

        let max_input = input.iter().map(|x| x.unsigned_abs() as u32).max().unwrap_or(1).max(1);
        let activity_threshold = (max_input / 4) as u16;

        for j in 0..self.rows {
            // Derive direction from output layer weights
            let correct_w = output_layer.weights[label * output_layer.cols + j].current();
            let wrong_w = output_layer.weights[predicted * output_layer.cols + j].current();

            // If correct class wants this neuron positive but wrong class also wants it:
            // the hidden neuron needs to differentiate
            let direction: i16 = (correct_w.signum() - wrong_w.signum()) as i16;
            if direction == 0 { continue; }

            // Only update weights for active inputs
            for k in 0..self.cols {
                let input_abs = input[k].unsigned_abs() as u16;
                if input_abs <= activity_threshold { continue; }

                let input_sign = if input[k] > 0 { 1i16 } else { -1 };
                let activity = ((input_abs - activity_threshold) as i32 * 8
                    / max_input as i32).max(1) as i16;
                let pressure_delta = direction * input_sign * activity;

                let idx = j * self.cols + k;
                let w = &mut self.weights[idx];
                w.pressure = w.pressure.saturating_add(pressure_delta);

                if w.pressure.abs() >= self.pressure_threshold {
                    let needed = if w.pressure > 0 { 1i8 } else { -1 };
                    if w.polarity == needed {
                        w.magnitude = w.magnitude.saturating_add(self.magnitude_step);
                    } else if w.polarity == -needed {
                        if w.magnitude <= self.magnitude_step {
                            w.magnitude = 0;
                            w.polarity = 0;
                        } else {
                            w.magnitude -= self.magnitude_step;
                        }
                    } else {
                        w.polarity = needed;
                        w.magnitude = self.magnitude_step;
                    }
                    self.transitions += 1;
                    w.pressure = 0;
                }
            }
        }
    }

    /// Decay pressure toward zero. Call between cycles.
    pub fn decay(&mut self) {
        for w in &mut self.weights {
            if w.pressure > 0 { w.pressure -= 1; }
            else if w.pressure < 0 { w.pressure += 1; }
        }
    }

    pub fn nonzero_count(&self) -> usize {
        self.weights.iter().filter(|w| w.magnitude > 0).count()
    }

    pub fn weight_stats(&self) -> (f64, f64) {
        let n = self.weights.len() as f64;
        let mean = self.weights.iter().map(|w| w.current().abs() as f64).sum::<f64>() / n;
        let std = (self.weights.iter().map(|w| {
            let d = w.current().abs() as f64 - mean;
            d * d
        }).sum::<f64>() / n).sqrt();
        (mean, std)
    }
}

// ---------------------------------------------------------------------------
// Full network
// ---------------------------------------------------------------------------

pub struct LeNetMastery {
    // Frozen random layers (never updated)
    conv1_k: T4, conv1_b: T1,
    conv2_k: T4, conv2_b: T1,
    fc1_w: T2, fc1_b: T1,
    // Plastic layers (mastery learning)
    pub hidden: MasteryLayer,   // FC2: 120 → 84
    pub output: MasteryLayer,   // FC3: 84 → 10
}

impl LeNetMastery {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            conv1_k: T4::random(6, 1, 5, 5, &mut rng, 25),
            conv1_b: T1::random(6, &mut rng, 5),
            conv2_k: T4::random(16, 6, 5, 5, &mut rng, 10),
            conv2_b: T1::random(16, &mut rng, 5),
            fc1_w: T2::random(120, 256, &mut rng, 5),
            fc1_b: T1::random(120, &mut rng, 5),
            // Hidden: frozen random (this is what gives 65%)
            hidden: MasteryLayer::frozen_random(84, 120, &mut rng),
            output: MasteryLayer::new(10, 84),
        }
    }

    /// Variant with plastic hidden layer (for alternating freeze experiment).
    pub fn new_with_plastic_hidden(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            conv1_k: T4::random(6, 1, 5, 5, &mut rng, 25),
            conv1_b: T1::random(6, &mut rng, 5),
            conv2_k: T4::random(16, 6, 5, 5, &mut rng, 10),
            conv2_b: T1::random(16, &mut rng, 5),
            fc1_w: T2::random(120, 256, &mut rng, 5),
            fc1_b: T1::random(120, &mut rng, 5),
            hidden: MasteryLayer::random_init(84, 120, &mut rng),
            output: MasteryLayer::new(10, 84),
        }
    }

    /// Forward through frozen conv/fc1, returning 120-dim features.
    fn frozen_features(&self, image: &[u8]) -> Vec<i16> {
        let mut input = T3::zeros(1, 28, 28);
        for y in 0..28 {
            for x in 0..28 {
                *input.at_mut(0, y, x) = (image[y * 28 + x] as i16) / 2;
            }
        }

        let x = relu(&conv2d(&input, &self.conv1_k, &self.conv1_b));
        let x = avgpool(&x);
        let x = relu(&conv2d(&x, &self.conv2_k, &self.conv2_b));
        let x = avgpool(&x);
        relu_vec(&fc(&x.data, &self.fc1_w, &self.fc1_b))
    }

    /// Full forward: frozen → hidden (plastic) → output (plastic)
    pub fn features(&self, image: &[u8]) -> Vec<i16> {
        let fc1_out = self.frozen_features(image);
        relu_vec_i16(&self.hidden.forward(&fc1_out))
    }

    pub fn predict(&self, image: &[u8]) -> usize {
        let scores = self.scores(image);
        scores.iter().enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i).unwrap_or(0)
    }

    /// Return raw output scores for all 10 classes.
    pub fn scores(&self, image: &[u8]) -> Vec<i16> {
        let hidden_out = self.features(image);
        self.output.forward(&hidden_out)
    }

    pub fn train_one(&mut self, image: &[u8], label: usize) -> (usize, bool) {
        // Forward through frozen layers
        let fc1_out = self.frozen_features(image);

        // Forward through hidden (plastic)
        let hidden_raw = self.hidden.forward(&fc1_out);
        let hidden_out = relu_vec_i16(&hidden_raw);

        // Forward through output (plastic)
        let output = self.output.forward(&hidden_out);
        let predicted = output.iter().enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i).unwrap_or(0);

        // Learn output layer (direct: knows the label)
        self.output.learn(&hidden_out, &output, label);

        // Learn hidden layer (indirect: strengthen neurons that contributed
        // to the correct output, weaken those that contributed to wrong output)
        // The error signal for the hidden layer is derived from the output error:
        // which hidden neurons fed the correct class vs the predicted class.
        self.hidden.learn_hidden(&fc1_out, &hidden_raw, &self.output, label, predicted);

        (predicted, predicted == label)
    }

    pub fn decay(&mut self) {
        self.hidden.decay();
        self.output.decay();
    }

    pub fn total_weights(&self) -> usize {
        self.conv1_k.data.len() + self.conv1_b.data.len()
        + self.conv2_k.data.len() + self.conv2_b.data.len()
        + self.fc1_w.data.len() + self.fc1_b.data.len()
        + self.hidden.weights.len() + self.hidden.bias.len()
        + self.output.weights.len() + self.output.bias.len()
    }

    pub fn plastic_weights(&self) -> usize {
        self.hidden.weights.len() + self.hidden.bias.len()
        + self.output.weights.len() + self.output.bias.len()
    }
}
