//! Flow-based signal weights — experimental prototype.
//!
//! Each weight is a living oscillator with position, momentum, and derived temperature.
//! Learning is built into the weight itself: error signals produce nudges that accumulate
//! as momentum, which is applied to the weight's value with inertia and decay.
//!
//! No discrete transitions. No pressure thresholds. No hit counting.
//! Weights flow toward their target values and settle naturally.
//!
//! ```text
//! frame N:   [value: 42,  momentum: +8,  temp: 180]
//! frame N+1: [value: 50,  momentum: +5,  temp: 170]  // applied, decayed
//! frame N+2: [value: 53,  momentum: -2,  temp: 165]  // error pushed back
//! frame N+3: [value: 51,  momentum: -1,  temp: 155]  // settling
//! frame N+4: [value: 50,  momentum:  0,  temp: 145]  // stable — cooling
//! ```

/// A single flow weight — position + momentum oscillator.
///
/// 5 bytes per weight:
///   value:    i16  — signed weight value, full range [-32768, 32767]
///   momentum: i16  — velocity of change (accumulated nudges with decay)
///   temp:     u8   — derived from momentum history, not stored externally
#[derive(Clone, Debug)]
pub struct FlowWeight {
    /// Current weight value. Signed, full range.
    pub value: i16,
    /// Momentum — the velocity of change. Positive = trending up, negative = trending down.
    /// Persists between steps, decays naturally. Inertia prevents flip-flopping.
    pub momentum: i16,
    /// Running average of |momentum| — used to derive temperature.
    /// High = volatile (HOT), low = stable (COLD).
    pub volatility: u16,
}

impl Default for FlowWeight {
    fn default() -> Self {
        Self { value: 0, momentum: 0, volatility: 255 }
    }
}

/// Momentum decay factor as a fraction of 256 (i.e., 230/256 ≈ 0.898).
/// Higher = more inertia, slower to change direction.
const MOMENTUM_DECAY: i32 = 230;

/// Volatility decay — exponential moving average factor (of 256).
/// Higher = slower temperature changes.
const VOLATILITY_DECAY: i32 = 250;

/// Value clamp range.
const VALUE_MIN: i16 = -500;
const VALUE_MAX: i16 = 500;

impl FlowWeight {
    pub fn new(value: i16) -> Self {
        Self { value, momentum: 0, volatility: 255 }
    }

    /// Derived temperature: 255 (HOT) when volatile, 0 (COLD) when stable.
    pub fn temperature(&self) -> u8 {
        self.volatility.min(255) as u8
    }

    /// Temperature band name.
    pub fn band(&self) -> &'static str {
        let t = self.temperature();
        if t >= 192 { "HOT" }
        else if t >= 128 { "WARM" }
        else if t >= 64 { "COOL" }
        else { "COLD" }
    }

    /// Temperature factor: how much of the nudge gets applied.
    /// HOT weights respond fully. COLD weights are dampened.
    /// Returns a value in [16, 256] to avoid division.
    fn temp_factor(&self) -> i32 {
        // Minimum factor of 16/256 ≈ 6% so COLD weights can still be warmed by strong error
        let t = self.temperature() as i32;
        16 + (t * 240 / 255)
    }

    /// Apply one learning step.
    ///
    /// `nudge` is the raw error signal for this weight: error × input_strength.
    /// Momentum accumulates the nudge with decay. Value follows momentum.
    /// Temperature tracks stability.
    pub fn step(&mut self, nudge: i32) {
        // Accumulate momentum: decay old + add nudge scaled by temperature
        let scaled_nudge = (nudge * self.temp_factor()) / 256;
        let new_momentum = (self.momentum as i32 * MOMENTUM_DECAY / 256) + scaled_nudge;
        self.momentum = new_momentum.clamp(-32000, 32000) as i16;

        // Apply momentum to value
        // Momentum is the velocity — divide by 4 for responsive movement
        let delta = self.momentum as i32 / 4;
        let new_value = (self.value as i32 + delta).clamp(VALUE_MIN as i32, VALUE_MAX as i32);
        self.value = new_value as i16;

        // Update volatility: EMA of |momentum|
        let abs_momentum = self.momentum.unsigned_abs() as u32;
        let new_vol = (self.volatility as u32 * VOLATILITY_DECAY as u32 / 256)
            + (abs_momentum * (256 - VOLATILITY_DECAY as u32) / 256);
        self.volatility = new_vol.min(65535) as u16;
    }

    /// Decay momentum without applying a nudge. Call between cycles.
    pub fn coast(&mut self) {
        let new_momentum = self.momentum as i32 * MOMENTUM_DECAY / 256;
        self.momentum = new_momentum as i16;

        // Volatility decays toward zero
        self.volatility = (self.volatility as u32 * VOLATILITY_DECAY as u32 / 256) as u16;
    }
}

// ---------------------------------------------------------------------------
// Flow weight matrix
// ---------------------------------------------------------------------------

/// A matrix of FlowWeights with shape metadata. Row-major: data[row * cols + col].
#[derive(Clone, Debug)]
pub struct FlowMatrix {
    pub data: Vec<FlowWeight>,
    pub rows: usize,
    pub cols: usize,
}

impl FlowMatrix {
    /// Create a zero-initialized matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![FlowWeight::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create from initial values (e.g., from imprinting).
    pub fn from_values(rows: usize, cols: usize, values: &[i16]) -> Self {
        assert_eq!(values.len(), rows * cols);
        Self {
            data: values.iter().map(|&v| FlowWeight::new(v)).collect(),
            rows,
            cols,
        }
    }

    /// Matrix-vector multiply: output[i] = sum(data[i][j] * input[j]) for each row i.
    pub fn forward(&self, input: &[i16]) -> Vec<i32> {
        assert_eq!(input.len(), self.cols);
        let mut output = vec![0i32; self.rows];
        for i in 0..self.rows {
            let row_start = i * self.cols;
            let mut sum = 0i64;
            for j in 0..self.cols {
                sum += self.data[row_start + j].value as i64 * input[j] as i64;
            }
            // Scale down to prevent overflow in downstream computation
            output[i] = (sum / self.cols as i64).clamp(-32000, 32000) as i32;
        }
        output
    }

    /// One learning step for the entire matrix.
    ///
    /// `input`: the input vector (cols-length)
    /// `output`: the forward pass output (rows-length)
    /// `target`: what the output should have been (rows-length)
    ///
    /// For each weight[i][j]:
    ///   error = target[i] - output[i]
    ///   nudge = error × input[j] / scale_factor
    ///   weight.step(nudge)
    pub fn learn(&mut self, input: &[i16], output: &[i32], target: &[i32]) {
        assert_eq!(input.len(), self.cols);
        assert_eq!(output.len(), self.rows);
        assert_eq!(target.len(), self.rows);

        for i in 0..self.rows {
            let error = target[i] as i64 - output[i] as i64;
            if error == 0 { continue; }

            let row_start = i * self.cols;
            for j in 0..self.cols {
                let input_val = input[j] as i64;
                if input_val == 0 { continue; }

                // Nudge proportional to error × input
                // Scale to keep nudges in a reasonable range for i16 weights
                let nudge = (error * input_val / 100).clamp(-500, 500) as i32;
                self.data[row_start + j].step(nudge);
            }
        }
    }

    /// Coast all weights — decay momentum without nudges. Call between cycles.
    pub fn coast(&mut self) {
        for w in &mut self.data {
            w.coast();
        }
    }

    /// Get values as a flat i16 slice (for visualization or correlation).
    pub fn values(&self) -> Vec<i16> {
        self.data.iter().map(|w| w.value).collect()
    }

    /// Get temperatures as a flat u8 slice (for visualization).
    pub fn temperatures(&self) -> Vec<u8> {
        self.data.iter().map(|w| w.temperature()).collect()
    }

    /// Get momentum magnitudes as a flat u8 slice (for activity visualization).
    /// Scaled to 0-255 range.
    pub fn momenta(&self) -> Vec<u8> {
        let max_mom = self.data.iter()
            .map(|w| w.momentum.unsigned_abs() as u32)
            .max().unwrap_or(1).max(1);
        self.data.iter().map(|w| {
            (w.momentum.unsigned_abs() as u32 * 255 / max_mom).min(255) as u8
        }).collect()
    }

    /// Count nonzero weights.
    pub fn nonzero_count(&self) -> usize {
        self.data.iter().filter(|w| w.value != 0).count()
    }

    /// Sum of absolute values per row.
    pub fn row_magnitudes(&self) -> Vec<i64> {
        (0..self.rows).map(|i| {
            let start = i * self.cols;
            self.data[start..start + self.cols].iter()
                .map(|w| w.value.unsigned_abs() as i64)
                .sum()
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// Standalone MNIST experiment
// ---------------------------------------------------------------------------

/// Encode a pixel (0-255) to a signed i16 signal.
/// Dark = negative, mid = zero, bright = positive.
fn encode_pixel(px: u8) -> i16 {
    if px <= 84 { -(((84 - px as i32) * 60) / 84) as i16 }
    else if px >= 171 { (((px as i32 - 171) * 60) / 84) as i16 }
    else { 0 }
}

fn encode_image(pixels: &[u8]) -> Vec<i16> {
    pixels.iter().map(|&px| encode_pixel(px)).collect()
}

/// Simple 784 → 10 single-layer flow network for MNIST.
pub struct FlowClassifier {
    pub weights: FlowMatrix,
    pub steps: u64,
}

impl FlowClassifier {
    pub fn new() -> Self {
        Self {
            weights: FlowMatrix::zeros(10, 784),
            steps: 0,
        }
    }

    /// Forward pass: returns 10 output scores.
    pub fn forward(&self, input: &[i16]) -> Vec<i32> {
        self.weights.forward(input)
    }

    /// Predict: returns the class with highest score.
    pub fn predict(&self, input: &[i16]) -> usize {
        let output = self.forward(input);
        output.iter().enumerate()
            .max_by_key(|(_, s)| *s)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Train on one sample: forward, compute error, learn.
    /// Returns (predicted_class, correct).
    pub fn train(&mut self, input: &[i16], label: usize) -> (usize, bool) {
        let output = self.forward(input);
        let predicted = output.iter().enumerate()
            .max_by_key(|(_, s)| *s)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Target: +40 for correct class, -20 for all others
        let mut target = vec![-20i32; 10];
        if label < 10 { target[label] = 40; }

        self.weights.learn(input, &output, &target);
        self.steps += 1;

        (predicted, predicted == label)
    }

    /// Coast all weights between cycles.
    pub fn coast(&mut self) {
        self.weights.coast();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_weight_settles() {
        let mut w = FlowWeight::default();
        // Apply consistent positive nudge
        for _ in 0..50 {
            w.step(10);
        }
        assert!(w.value > 0, "weight should be positive after positive nudges: {}", w.value);
        assert!(w.momentum > 0, "momentum should be positive");

        // Now reverse
        let before = w.value;
        for _ in 0..10 {
            w.step(-20);
        }
        assert!(w.value < before, "weight should decrease after negative nudges");
        // But shouldn't flip instantly due to inertia
        assert!(w.value > 0 || w.momentum < 0, "should show inertia effect");
    }

    #[test]
    fn flow_weight_temperature_tracks_stability() {
        let mut w = FlowWeight::default();
        // Wild nudges = should stay hot
        for i in 0..100 {
            let nudge = if i % 2 == 0 { 50 } else { -50 };
            w.step(nudge);
        }
        let hot_temp = w.temperature();

        // Let it settle
        for _ in 0..200 {
            w.coast();
        }
        let cold_temp = w.temperature();

        assert!(cold_temp < hot_temp,
            "temperature should decrease when weight settles: hot={} cold={}",
            hot_temp, cold_temp);
    }

    #[test]
    fn flow_matrix_forward() {
        let mut m = FlowMatrix::zeros(2, 3);
        m.data[0].value = 10;  // row 0, col 0
        m.data[1].value = 20;  // row 0, col 1
        m.data[2].value = 30;  // row 0, col 2
        m.data[3].value = -10; // row 1, col 0
        m.data[4].value = -20; // row 1, col 1
        m.data[5].value = -30; // row 1, col 2

        let input = vec![1, 1, 1];
        let output = m.forward(&input);

        // row 0: (10+20+30)/3 = 20
        // row 1: (-10-20-30)/3 = -20
        assert_eq!(output[0], 20);
        assert_eq!(output[1], -20);
    }

    #[test]
    fn flow_matrix_learns() {
        let mut m = FlowMatrix::zeros(2, 3);
        let input = vec![60i16, 60, 60];
        let output = m.forward(&input); // all zeros
        let target = vec![40i32, -20];

        m.learn(&input, &output, &target);

        // Row 0 weights should have moved positive (target > output)
        let r0_sum: i16 = m.data[0..3].iter().map(|w| w.value).sum();
        // Row 1 weights should have moved negative (target < output)
        let r1_sum: i16 = m.data[3..6].iter().map(|w| w.value).sum();

        assert!(r0_sum > 0, "row 0 should be positive: {}", r0_sum);
        assert!(r1_sum < 0, "row 1 should be negative: {}", r1_sum);
    }

    #[test]
    fn flow_classifier_basic() {
        let mut clf = FlowClassifier::new();

        // Create a simple pattern: all bright pixels for digit 3
        let mut img = vec![200u8; 784];
        let input = encode_image(&img);

        // Train on it 100 times
        let mut last_correct = false;
        for _ in 0..100 {
            let (_, correct) = clf.train(&input, 3);
            last_correct = correct;
            clf.coast();
        }

        // Should eventually learn this single pattern
        let pred = clf.predict(&input);
        assert_eq!(pred, 3, "should predict 3 after repeated training");
    }

    #[test]
    fn inertia_prevents_flip_flop() {
        let mut w = FlowWeight::default();

        // Establish strong positive momentum
        for _ in 0..50 {
            w.step(20);
        }
        let after_positive = w.value;

        // Single opposing nudge should NOT flip the weight
        w.step(-20);
        assert!(w.value > 0,
            "single opposing nudge should not flip: was {} now {}",
            after_positive, w.value);

        // Even 5 opposing nudges shouldn't fully reverse 50 positive ones
        for _ in 0..5 {
            w.step(-20);
        }
        assert!(w.value > after_positive / 2,
            "5 opposing nudges should not halve 50 positive: was {} now {}",
            after_positive, w.value);
    }
}
