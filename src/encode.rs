//! Pixel → PackedSignal encoding.
//!
//! 3-band ternary encoding:
//!   0-84:   Inhibitory (dark/background)
//!   85-170: Silent (ambiguous mid-grey)
//!   171-255: Excitatory (bright/stroke)

use ternary_signal::PackedSignal;
use atomic_neural_transistors::core::weight_matrix::packed_from_current;

/// Encode a single pixel (0-255) to PackedSignal.
pub fn encode_pixel(px: u8) -> PackedSignal {
    if px <= 84 {
        // Dark → inhibitory. Darker = stronger inhibition.
        let strength = ((84 - px as i32) * 60) / 84; // 0-60
        packed_from_current(-strength)
    } else if px >= 171 {
        // Bright → excitatory. Brighter = stronger excitation.
        let strength = ((px as i32 - 171) * 60) / 84; // 0-60
        packed_from_current(strength)
    } else {
        PackedSignal::ZERO
    }
}

/// Encode a full 784-pixel image to PackedSignals.
pub fn encode_image(pixels: &[u8]) -> Vec<PackedSignal> {
    pixels.iter().map(|&px| encode_pixel(px)).collect()
}

/// Print encoded image as ternary visualization.
pub fn print_encoded(signals: &[PackedSignal]) {
    for row in 0..28 {
        for col in 0..28 {
            let s = signals[row * 28 + col];
            let c = s.current();
            let ch = if c > 20 { '#' }
                else if c > 0 { '+' }
                else if c == 0 { ' ' }
                else if c > -20 { '.' }
                else { '_' };
            print!("{}", ch);
        }
        println!();
    }
}
