//! MNIST IDX file parser.
//!
//! IDX format:
//!   Images: magic(2051) → count(u32) → rows(u32) → cols(u32) → pixels(u8...)
//!   Labels: magic(2049) → count(u32) → labels(u8...)

use std::path::Path;

pub struct MnistImage {
    pub pixels: Vec<u8>, // 784 pixels (28×28), 0-255
    pub label: u8,       // 0-9
}

pub struct MnistDataset {
    pub images: Vec<MnistImage>,
}

impl MnistDataset {
    pub fn load(image_path: &Path, label_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let img_data = std::fs::read(image_path)?;
        let lbl_data = std::fs::read(label_path)?;

        // Verify magic numbers
        let img_magic = u32::from_be_bytes([img_data[0], img_data[1], img_data[2], img_data[3]]);
        let lbl_magic = u32::from_be_bytes([lbl_data[0], lbl_data[1], lbl_data[2], lbl_data[3]]);
        if img_magic != 2051 {
            return Err(format!("Bad image magic: {} (expected 2051)", img_magic).into());
        }
        if lbl_magic != 2049 {
            return Err(format!("Bad label magic: {} (expected 2049)", lbl_magic).into());
        }

        let n_images = u32::from_be_bytes([img_data[4], img_data[5], img_data[6], img_data[7]]) as usize;
        let n_labels = u32::from_be_bytes([lbl_data[4], lbl_data[5], lbl_data[6], lbl_data[7]]) as usize;
        let rows = u32::from_be_bytes([img_data[8], img_data[9], img_data[10], img_data[11]]) as usize;
        let cols = u32::from_be_bytes([img_data[12], img_data[13], img_data[14], img_data[15]]) as usize;

        if n_images != n_labels {
            return Err(format!("Image count {} != label count {}", n_images, n_labels).into());
        }
        if rows != 28 || cols != 28 {
            return Err(format!("Expected 28×28, got {}×{}", rows, cols).into());
        }

        let pixel_start = 16;
        let label_start = 8;
        let px_per_image = rows * cols; // 784

        let mut images = Vec::with_capacity(n_images);
        for i in 0..n_images {
            let pix_offset = pixel_start + i * px_per_image;
            let pixels = img_data[pix_offset..pix_offset + px_per_image].to_vec();
            let label = lbl_data[label_start + i];
            images.push(MnistImage { pixels, label });
        }

        Ok(Self { images })
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Print a digit to terminal as ASCII art.
    pub fn print_image(image: &MnistImage) {
        println!("Label: {}", image.label);
        for row in 0..28 {
            for col in 0..28 {
                let px = image.pixels[row * 28 + col];
                let ch = if px > 200 { '#' }
                    else if px > 128 { '+' }
                    else if px > 64 { '.' }
                    else { ' ' };
                print!("{}", ch);
            }
            println!();
        }
    }
}
