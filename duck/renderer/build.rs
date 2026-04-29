//! Build script for the renderer crate.
//!
//! Generates the BRDF integration lookup table at build time using CPU math.
//! This is a port of the `brdf_lut.wgsl` compute shader to Rust, producing
//! the same 512x512 LUT stored as raw f32 RGBA values.
//!
//! The renderer converts f32 → Rgba16Float at texture upload time using the
//! `half` crate. If `FLOAT32_FILTERABLE` support becomes universal, we could
//! switch to storing and uploading as Rgba32Float directly.

use std::env;
use std::fs;
use std::path::Path;

const LUT_SIZE: u32 = 512;
const SAMPLE_COUNT: u32 = 1024;
const PI: f32 = std::f32::consts::PI;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("brdf_lut.bin");

    // 4 f32 channels per pixel, stored as little-endian bytes
    let bytes_per_pixel = 4 * 4; // 4 channels * 4 bytes per f32
    let total_bytes = (LUT_SIZE * LUT_SIZE) as usize * bytes_per_pixel;
    let mut data = vec![0u8; total_bytes];

    for y in 0..LUT_SIZE {
        for x in 0..LUT_SIZE {
            let n_dot_v = ((x as f32) + 0.5) / (LUT_SIZE as f32);
            let roughness = ((y as f32) + 0.5) / (LUT_SIZE as f32);
            let clamped_n_dot_v = n_dot_v.max(0.001);

            let (scale, bias) = integrate_brdf(clamped_n_dot_v, roughness);

            let pixel_offset = ((y * LUT_SIZE + x) as usize) * bytes_per_pixel;
            data[pixel_offset..pixel_offset + 4].copy_from_slice(&scale.to_le_bytes());
            data[pixel_offset + 4..pixel_offset + 8].copy_from_slice(&bias.to_le_bytes());
            data[pixel_offset + 8..pixel_offset + 12].copy_from_slice(&0.0f32.to_le_bytes());
            data[pixel_offset + 12..pixel_offset + 16].copy_from_slice(&1.0f32.to_le_bytes());
        }
    }

    fs::write(&out_path, &data).expect("Failed to write BRDF LUT");
}

fn integrate_brdf(n_dot_v: f32, roughness: f32) -> (f32, f32) {
    // View direction in tangent space where N = (0, 0, 1)
    let v = [
        (1.0 - n_dot_v * n_dot_v).sqrt(),
        0.0f32,
        n_dot_v,
    ];

    let mut a = 0.0f32;
    let mut b = 0.0f32;

    for i in 0..SAMPLE_COUNT {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness);

        let v_dot_h = (v[0] * h[0] + v[1] * h[1] + v[2] * h[2]).max(0.0);
        // l = 2 * dot(v, h) * h - v
        let l = [
            2.0 * v_dot_h * h[0] - v[0],
            2.0 * v_dot_h * h[1] - v[1],
            2.0 * v_dot_h * h[2] - v[2],
        ];

        let n_dot_l = l[2].max(0.0); // N = (0, 0, 1)
        let n_dot_h = h[2].max(0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = (1.0 - v_dot_h).powf(5.0);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    (a / SAMPLE_COUNT as f32, b / SAMPLE_COUNT as f32)
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    (bits as f32) * 2.328_306_4e-10
}

fn hammersley(i: u32, n: u32) -> [f32; 2] {
    [(i as f32) / (n as f32), radical_inverse_vdc(i)]
}

fn importance_sample_ggx(xi: [f32; 2], roughness: f32) -> [f32; 3] {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi[0];
    let cos_theta = ((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1])).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    [phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta]
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let a = roughness;
    let k = (a * a) / 2.0;
    n_dot_v / (n_dot_v * (1.0 - k) + k)
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness)
}
