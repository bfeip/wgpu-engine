// lighting.wesl
// Multi-light support with point, directional, and spot lights

import package::constants::{GROUP_LIGHTS, BINDING_LIGHTS_UNIFORM};
import package::common::{
    LightsUniform, LightUniform,
    LIGHT_TYPE_POINT, LIGHT_TYPE_DIRECTIONAL, LIGHT_TYPE_SPOT, MAX_LIGHTS
};

@group(GROUP_LIGHTS) @binding(BINDING_LIGHTS_UNIFORM)
var<uniform> lights: LightsUniform;

// Calculate distance attenuation for point and spot lights.
// If range is 0, uses simple inverse square falloff.
// If range > 0, applies smooth falloff to zero at range.
fn calculate_attenuation(distance: f32, range: f32) -> f32 {
    if (range <= 0.0) {
        // Simple inverse square falloff (with offset to avoid division issues at distance 0)
        return 1.0 / (distance * distance + 1.0);
    }
    // Smooth falloff to zero at range
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

// Calculate contribution from a point light
fn calculate_point_light(
    light: LightUniform,
    world_position: vec3<f32>,
    world_normal: vec3<f32>
) -> vec3<f32> {
    let to_light = light.position - world_position;
    let distance = length(to_light);
    let light_dir = to_light / distance;

    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let attenuation = calculate_attenuation(distance, light.range);

    return light.color * light.intensity * n_dot_l * attenuation;
}

// Calculate contribution from a directional light
fn calculate_directional_light(
    light: LightUniform,
    world_normal: vec3<f32>
) -> vec3<f32> {
    // Direction points from light toward scene, so negate for surface-to-light
    let light_dir = -light.direction;
    let n_dot_l = max(dot(world_normal, light_dir), 0.0);

    return light.color * light.intensity * n_dot_l;
}

// Calculate contribution from a spotlight
fn calculate_spot_light(
    light: LightUniform,
    world_position: vec3<f32>,
    world_normal: vec3<f32>
) -> vec3<f32> {
    let to_light = light.position - world_position;
    let distance = length(to_light);
    let light_dir = to_light / distance;

    let n_dot_l = max(dot(world_normal, light_dir), 0.0);
    let attenuation = calculate_attenuation(distance, light.range);
    let cone_falloff = calculate_spot_cone_falloff(
        -light_dir, light.direction,
        light.inner_cone_cos, light.outer_cone_cos
    );

    return light.color * light.intensity * n_dot_l * attenuation * cone_falloff;
}

// Calculate total diffuse lighting from all lights
fn calculate_diffuse_lighting(
    world_position: vec3<f32>,
    world_normal: vec3<f32>
) -> vec3<f32> {
    var total = vec3<f32>(0.0);

    for (var i = 0u; i < lights.light_count && i < MAX_LIGHTS; i = i + 1u) {
        let light = lights.lights[i];

        if (light.light_type == LIGHT_TYPE_POINT) {
            total += calculate_point_light(light, world_position, world_normal);
        } else if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
            total += calculate_directional_light(light, world_normal);
        } else if (light.light_type == LIGHT_TYPE_SPOT) {
            total += calculate_spot_light(light, world_position, world_normal);
        }
    }

    return total;
}
