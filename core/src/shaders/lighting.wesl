// lighting.wesl
// Light uniform binding and shared lighting utilities

import package::constants::{GROUP_LIGHTS, BINDING_LIGHTS_UNIFORM};
import package::common::LightsUniform;

@group(GROUP_LIGHTS) @binding(BINDING_LIGHTS_UNIFORM)
var<uniform> lights: LightsUniform;

// Calculate distance attenuation for point and spot lights.
// If range is 0 (infinite), returns 1.0 (no attenuation).
// If range > 0, applies smooth falloff to zero at range.
fn calculate_attenuation(distance: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        return 1.0;  // No attenuation for infinite range lights
    }
    // Range-based attenuation for finite-range lights
    let d = distance / range;
    let range_falloff = saturate(1.0 - d * d);
    return range_falloff * range_falloff / (distance * distance + 1.0);
}

// Calculate spotlight cone falloff.
// Returns 1.0 inside inner cone, 0.0 outside outer cone, smooth transition between.
fn calculate_spot_cone_falloff(
    light_to_surface_dir: vec3<f32>,
    light_direction: vec3<f32>,
    inner_cone_cos: f32,
    outer_cone_cos: f32
) -> f32 {
    let cos_angle = dot(-light_to_surface_dir, light_direction);
    return smoothstep(outer_cone_cos, inner_cone_cos, cos_angle);
}

