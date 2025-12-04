// lighting.wesl
// Lighting uniform binding at group 1

use package::constants::{GROUP_LIGHTS, BINDING_LIGHTS_UNIFORM};
use package::common::LightsUniform;

@group(GROUP_LIGHTS) @binding(BINDING_LIGHTS_UNIFORM)
pub var<uniform> lights: LightsUniform;

// Calculate diffuse lighting
pub fn calculate_diffuse_lighting(
    world_position: vec3<f32>,
    world_normal: vec3<f32>
) -> vec3<f32> {
    let light_dir = normalize(lights.position - world_position);
    let diffuse_strength = max(dot(world_normal, light_dir), 0.0) * lights.color.a;
    return lights.color.rgb * diffuse_strength;
}
