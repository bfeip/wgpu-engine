// outline_screenspace.wesl
// Fullscreen post-process shader for screen-space selection outlines
// Samples mask texture at pixel offsets to detect edges

// Outline configuration uniform
struct OutlineUniform {
    color: vec4<f32>,
    width_pixels: f32,
    screen_width: f32,
    screen_height: f32,
    _padding: f32,
};

@group(0) @binding(0)
var mask_texture: texture_2d<f32>;

@group(0) @binding(1)
var mask_sampler: sampler;

@group(0) @binding(2)
var<uniform> outline: OutlineUniform;

// Fullscreen triangle vertex shader
// Generates a triangle that covers the entire screen from vertex index
@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Generate fullscreen triangle vertices:
    // vertex 0: (-1, -1)
    // vertex 1: ( 3, -1)
    // vertex 2: (-1,  3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Fragment shader that detects edges in the mask and draws outline
@fragment
fn fs_outline(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let screen_size = vec2<f32>(outline.screen_width, outline.screen_height);
    let uv = frag_coord.xy / screen_size;
    let pixel_size = 1.0 / screen_size;

    // Sample center of mask
    let center = textureSample(mask_texture, mask_sampler, uv).r;

    // If center pixel is selected (inside the object), don't draw outline here
    if center > 0.5 {
        discard;
    }

    // Sample neighbors in 8 directions at the outline radius
    let radius = outline.width_pixels;
    var max_sample = 0.0;

    // 8 direction samples around a circle
    let pi = 3.14159265359;
    for (var i = 0u; i < 8u; i++) {
        let angle = f32(i) * pi * 0.25; // 0, 45, 90, 135, 180, 225, 270, 315 degrees
        let offset = vec2<f32>(cos(angle), sin(angle)) * radius * pixel_size;
        let sample_val = textureSample(mask_texture, mask_sampler, uv + offset).r;
        max_sample = max(max_sample, sample_val);
    }

    // If any neighbor is inside the selection, we're on the outline
    if max_sample > 0.5 {
        return outline.color;
    }

    // Not on outline, discard to show original scene
    discard;
}
