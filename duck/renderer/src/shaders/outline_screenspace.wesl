// outline_screenspace.wesl
// Fullscreen post-process shader for screen-space selection outlines.
//
// Feature flag "multisampled":
//   true  — mask_texture is texture_multisampled_2d; coverage() averages all MSAA
//            samples per pixel, giving fractional values [0,1] at silhouette edges.
//   false — mask_texture is texture_2d; coverage() reads the single sample.
//
// In both cases coverage drives the outline alpha, so MSAA mode produces smooth
// anti-aliased edges while the non-MSAA mode stays binary (no regression).

struct OutlineUniform {
    color: vec4<f32>,
    width_pixels: f32,
    screen_width: f32,
    screen_height: f32,
    _padding: f32,
};

@if(multisampled)
@group(0) @binding(0)
var mask_texture: texture_multisampled_2d<f32>;

@if(!multisampled)
@group(0) @binding(0)
var mask_texture: texture_2d<f32>;

// Binding 1 intentionally absent — no sampler needed; both paths use textureLoad.

@group(0) @binding(2)
var<uniform> outline: OutlineUniform;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Returns fractional mask coverage for a pixel coordinate.
// MSAA: averages all samples → smooth [0,1] values at the silhouette boundary.
// Non-MSAA: reads the single sample → binary 0 or 1.
fn coverage(coord: vec2<i32>) -> f32 {
    @if(multisampled) {
        let n = i32(textureNumSamples(mask_texture));
        var total = 0.0;
        for (var i = 0i; i < n; i++) {
            total += textureLoad(mask_texture, coord, i).r;
        }
        return total / f32(n);
    }
    @if(!multisampled) {
        return textureLoad(mask_texture, coord, 0).r;
    }
    return 0.0;
}

@fragment
fn fs_outline(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let center_coord = vec2<i32>(i32(frag_coord.x), i32(frag_coord.y));

    // Pixels fully inside the selection don't need an outline.
    if coverage(center_coord) > 0.5 {
        discard;
    }

    // Sample 8 directions at the outline radius to find nearby selected pixels.
    let radius = outline.width_pixels;
    var max_coverage = 0.0;
    let pi = 3.14159265359;
    for (var i = 0u; i < 8u; i++) {
        let angle = f32(i) * pi * 0.25;
        let offset = vec2<i32>(
            i32(round(cos(angle) * radius)),
            i32(round(sin(angle) * radius)),
        );
        max_coverage = max(max_coverage, coverage(center_coord + offset));
    }

    if max_coverage <= 0.0 {
        discard;
    }

    // max_coverage is fractional at the silhouette boundary in MSAA mode,
    // giving a smooth alpha fade instead of a hard pixel edge.
    return vec4<f32>(outline.color.rgb, outline.color.a * max_coverage);
}
