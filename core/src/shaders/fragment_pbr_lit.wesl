// fragment_pbr_lit.wesl
// PBR fragment shader with Cook-Torrance BRDF

import package::common::VertexOutput;
import package::camera::camera;
import package::lighting::lights;
import package::material_pbr::{get_base_color, get_metallic_roughness};
import package::normal_mapping::get_normal;
import package::pbr::calculate_pbr_direct_light;

fn fragment_pbr_lit(in: VertexOutput) -> vec4<f32> {
    // Get base color (texture * factor, or just factor)
    let base_color = get_base_color(in.tex_coords);
    let albedo = base_color.rgb;
    let alpha = base_color.a;

    // Get metallic and roughness
    let mr = get_metallic_roughness(in.tex_coords);
    let metallic = mr.x;
    let roughness = mr.y;

    // Get normal (with normal mapping if available)
    let n = get_normal(in.world_position, in.world_normal, in.tex_coords);

    // View direction (from surface to camera)
    let v = normalize(camera.eye_position - in.world_position);

    // Light direction (from surface to light)
    let l = normalize(lights.position - in.world_position);

    // Calculate PBR lighting
    let light_contribution = calculate_pbr_direct_light(
        lights.color.rgb * lights.color.a,
        l,
        v,
        n,
        albedo,
        metallic,
        roughness
    );

    // Add ambient term (simple constant for now)
    let ambient = vec3<f32>(0.03) * albedo;

    let color = ambient + light_contribution;

    // HDR tonemapping (simple Reinhard)
    let mapped = color / (color + vec3<f32>(1.0));

    // Gamma correction
    let gamma_corrected = pow(mapped, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(gamma_corrected, alpha);
}
