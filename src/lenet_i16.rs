//! LeNet-5 with P×M×K signal weights.
//!
//! Every weight is a ternary signal: polarity × magnitude × multiplier.
//! Conv layers: frozen random projections (spatial feature extraction).
//! FC output layer: learns via basin update rule with temperature gating.
//!
//! The basin model:
//!   Polarity = direction of the arc (excitatory/inhibitory)
//!   Magnitude = depth of the basin (response strength)
//!   Multiplier = width of the basin (sensitivity range)
//!   Temperature = rigidity (resistance to change)

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
    fn range_i8(&mut self) -> i8 {
        let v = (self.next_u64() % 3) as i8;
        v - 1  // -1, 0, +1
    }
}

// ---------------------------------------------------------------------------
// P×M×K Signal Weight
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct SignalWeight {
    pub polarity: i8,       // -1, 0, +1
    pub magnitude: u8,      // 0-255 (depth of basin)
    pub multiplier: u8,     // 0-255 (width of basin)
    pub temperature: u8,    // 255=HOT (plastic), 0=COLD (frozen)
    pub pressure: i16,      // accumulated update pressure
}

impl SignalWeight {
    fn zero() -> Self {
        Self { polarity: 0, magnitude: 0, multiplier: 0, temperature: 255, pressure: 0 }
    }

    fn random(rng: &mut Rng) -> Self {
        let p = rng.range_i8();
        if p == 0 {
            Self::zero()
        } else {
            Self {
                polarity: p,
                magnitude: (rng.next_u8() % 40) + 10,  // 10-49
                multiplier: (rng.next_u8() % 3) + 1,    // 1-3
                temperature: 255,
                pressure: 0,
            }
        }
    }

    /// Effective value: P × M × K
    pub fn current(&self) -> i32 {
        self.polarity as i32 * self.magnitude as i32 * self.multiplier as i32
    }

    /// Apply the basin update rule given a raw error signal.
    ///
    /// raw_signal: error × input (from the learning context)
    /// Returns true if a transition occurred.
    fn update(&mut self, raw_signal: i32, pressure_threshold: i16) -> bool {
        // Temperature gates how much signal gets through
        if self.temperature == 0 { return false; }
        let temp_scale = self.temperature as i32;

        // Accumulate pressure (scaled by temperature)
        let pressure_delta = ((raw_signal * temp_scale) / 256).clamp(-500, 500) as i16;
        self.pressure = self.pressure.saturating_add(pressure_delta);

        // Check if pressure crossed threshold
        if self.pressure.abs() < pressure_threshold { return false; }

        let direction = if self.pressure > 0 { 1i8 } else { -1i8 };
        self.pressure = 0;

        // Basin update: which knob to turn?
        if self.polarity == 0 {
            // Silent weight — initialize in the needed direction
            self.polarity = direction;
            self.magnitude = 5;
            self.multiplier = 1;
            return true;
        }

        if self.polarity == direction {
            // Same direction as pressure — deepen the basin (strengthen)
            if self.magnitude < 250 {
                self.magnitude += 5;
            } else if self.multiplier < 255 {
                // Magnitude maxed — widen instead
                self.multiplier = self.multiplier.saturating_add(1);
            }
        } else {
            // Opposite direction — shallow the basin (weaken before flip)
            if self.magnitude > 5 {
                self.magnitude -= 5;
            } else {
                // Basin collapsed — polarity dies
                self.magnitude = 0;
                self.polarity = 0;
                self.multiplier = 0;
            }
        }

        // Cool slightly on each transition (evidence of participation)
        if self.temperature > 2 {
            self.temperature -= 2;
        }

        true
    }

    /// Decay pressure toward zero. Call between cycles.
    fn coast(&mut self) {
        if self.pressure > 0 { self.pressure -= 1; }
        else if self.pressure < 0 { self.pressure += 1; }
    }
}

// ---------------------------------------------------------------------------
// Signal matrix (frozen or plastic)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SignalMatrix {
    pub data: Vec<SignalWeight>,
    pub rows: usize,
    pub cols: usize,
}

impl SignalMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![SignalWeight::zero(); rows * cols], rows, cols }
    }

    fn random(rows: usize, cols: usize, rng: &mut Rng) -> Self {
        Self { data: (0..rows * cols).map(|_| SignalWeight::random(rng)).collect(), rows, cols }
    }

    /// Forward: dot product using current() values.
    /// Accumulate in i64, scale to i16 range.
    fn forward(&self, input: &[i16]) -> Vec<i16> {
        let mut output = vec![0i16; self.rows];
        for i in 0..self.rows {
            let mut sum: i64 = 0;
            for j in 0..self.cols {
                sum += input[j] as i64 * self.data[i * self.cols + j].current() as i64;
            }
            // Scale by column count to keep in i16 range
            output[i] = (sum / (self.cols as i64).max(1)).clamp(-32000, 32000) as i16;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Signal conv kernel (frozen)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SignalConv {
    data: Vec<SignalWeight>,
    n: usize, c: usize, h: usize, w: usize,
    bias: Vec<i16>,
}

impl SignalConv {
    fn random(n: usize, c: usize, h: usize, w: usize, rng: &mut Rng) -> Self {
        let data: Vec<SignalWeight> = (0..n * c * h * w).map(|_| SignalWeight::random(rng)).collect();
        let bias: Vec<i16> = (0..n).map(|_| rng.range_i8() as i16 * 2).collect();
        Self { data, n, c, h, w, bias }
    }

    fn at(&self, n: usize, c: usize, y: usize, x: usize) -> i32 {
        self.data[n * self.c * self.h * self.w + c * self.h * self.w + y * self.w + x].current()
    }

    fn forward(&self, input: &[i16], ic: usize, ih: usize, iw: usize) -> (Vec<i16>, usize, usize, usize) {
        let oh = ih - self.h + 1;
        let ow = iw - self.w + 1;
        let mut output = vec![0i16; self.n * oh * ow];
        let fan_in = (self.c * self.h * self.w) as i64;

        for f in 0..self.n {
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum: i64 = self.bias[f] as i64 * fan_in;
                    for c in 0..self.c {
                        for ky in 0..self.h {
                            for kx in 0..self.w {
                                let in_idx = c * ih * iw + (y + ky) * iw + (x + kx);
                                sum += input[in_idx] as i64 * self.at(f, c, ky, kx) as i64;
                            }
                        }
                    }
                    output[f * oh * ow + y * ow + x] =
                        (sum / fan_in).clamp(-32000, 32000) as i16;
                }
            }
        }
        (output, self.n, oh, ow)
    }
}

// ---------------------------------------------------------------------------
// Helper ops
// ---------------------------------------------------------------------------

fn relu_vec(v: &[i16]) -> Vec<i16> { v.iter().map(|&x| x.max(0)).collect() }

fn avgpool(input: &[i16], c: usize, h: usize, w: usize) -> (Vec<i16>, usize, usize, usize) {
    let oh = h / 2;
    let ow = w / 2;
    let mut out = vec![0i16; c * oh * ow];
    for ch in 0..c {
        for y in 0..oh {
            for x in 0..ow {
                let base = ch * h * w;
                let s = input[base + y*2*w + x*2] as i32
                    + input[base + (y*2+1)*w + x*2] as i32
                    + input[base + y*2*w + x*2+1] as i32
                    + input[base + (y*2+1)*w + x*2+1] as i32;
                out[ch * oh * ow + y * ow + x] = (s / 4) as i16;
            }
        }
    }
    (out, c, oh, ow)
}

// ---------------------------------------------------------------------------
// LeNet-5 with P×M×K signal weights
// ---------------------------------------------------------------------------

pub struct LeNetI16 {
    // Frozen random layers
    conv1: SignalConv,
    conv2: SignalConv,
    fc1: SignalMatrix,
    fc2: SignalMatrix,
    // Plastic output layer
    fc3: SignalMatrix,
    pub steps: u64,
    pub transitions: u64,
}

impl LeNetI16 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            conv1: SignalConv::random(6, 1, 5, 5, &mut rng),
            conv2: SignalConv::random(16, 6, 5, 5, &mut rng),
            fc1: SignalMatrix::random(120, 256, &mut rng),
            fc2: SignalMatrix::random(84, 120, &mut rng),
            fc3: SignalMatrix::zeros(10, 84),  // starts empty, learns from scratch
            steps: 0,
            transitions: 0,
        }
    }

    /// Forward through frozen layers → 84-dim features.
    fn features(&self, image: &[u8]) -> Vec<i16> {
        // Input: pixels to [0, 127]
        let input: Vec<i16> = image.iter().map(|&p| (p as i16) / 2).collect();

        let (x, c, h, w) = self.conv1.forward(&input, 1, 28, 28);
        let x = relu_vec(&x);
        let (x, c, h, w) = avgpool(&x, c, h, w);
        let (x, c, h, w) = self.conv2.forward(&x, c, h, w);
        let x = relu_vec(&x);
        let (x, _c, _h, _w) = avgpool(&x, c, h, w);
        let x = relu_vec(&self.fc1.forward(&x));
        relu_vec(&self.fc2.forward(&x))
    }

    pub fn predict(&self, image: &[u8]) -> usize {
        let feats = self.features(image);
        let output = self.fc3.forward(&feats);
        output.iter().enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i).unwrap_or(0)
    }

    pub fn train_one(&mut self, image: &[u8], label: usize) -> (usize, bool) {
        let feats = self.features(image);
        let output = self.fc3.forward(&feats);
        let predicted = output.iter().enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i).unwrap_or(0);

        self.steps += 1;

        // Participation threshold: top 25% of feature activity
        let max_feat = feats.iter().map(|x| x.unsigned_abs() as u32).max().unwrap_or(1).max(1);
        let threshold = (max_feat / 4) as u16;

        // Update only the correct class row and (if wrong) the predicted class row
        let rows_to_update: Vec<(usize, i32)> = if predicted == label {
            vec![(label, 1)]
        } else {
            vec![(label, 1), (predicted, -1)]
        };

        let pressure_threshold: i16 = 6;

        for &(row, direction) in &rows_to_update {
            for j in 0..self.fc3.cols {
                let feat_abs = feats[j].unsigned_abs() as u16;
                if feat_abs <= threshold { continue; }

                // Raw signal: direction × feature value
                let raw = direction * feats[j] as i32;

                let w = &mut self.fc3.data[row * self.fc3.cols + j];
                if w.update(raw, pressure_threshold) {
                    self.transitions += 1;
                }
            }
        }

        (predicted, predicted == label)
    }

    pub fn decay(&mut self) {
        for w in &mut self.fc3.data {
            w.coast();
        }
    }

    pub fn weight_stats(&self) -> I16WeightStats {
        let plastic = &self.fc3.data;
        let n = plastic.len() as f64;
        let mean = plastic.iter().map(|w| w.current().abs() as f64).sum::<f64>() / n;
        let std = (plastic.iter().map(|w| {
            let d = w.current().abs() as f64 - mean; d * d
        }).sum::<f64>() / n).sqrt();

        let total_frozen = self.conv1.data.len() + self.conv2.data.len()
            + self.fc1.data.len() + self.fc2.data.len();

        let mut temp_dist = [0u32; 4];
        for w in plastic {
            if w.temperature >= 192 { temp_dist[0] += 1; }
            else if w.temperature >= 128 { temp_dist[1] += 1; }
            else if w.temperature >= 64 { temp_dist[2] += 1; }
            else { temp_dist[3] += 1; }
        }

        let nonzero = plastic.iter().filter(|w| w.magnitude > 0).count();

        I16WeightStats {
            count: total_frozen + plastic.len(),
            plastic_count: plastic.len(),
            bytes: (total_frozen + plastic.len()) * std::mem::size_of::<SignalWeight>(),
            mean, std, nonzero,
            temp_dist,
        }
    }
}

pub struct I16WeightStats {
    pub count: usize,
    pub plastic_count: usize,
    pub bytes: usize,
    pub mean: f64,
    pub std: f64,
    pub nonzero: usize,
    pub temp_dist: [u32; 4],
}
