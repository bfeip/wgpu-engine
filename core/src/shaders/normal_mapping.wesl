// normal_mapping.wesl
// Normal map utilities with derivative-based tangent frame

import package::material_pbr::{t_normal, s_normal, pbr_params, has_normal_texture};

// Compute tangent frame from screen-space derivatives
// This avoids needing tangent attributes in vertex data
fn get_tbn_matrix(world_pos: vec3<f32>, world_normal: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let dp1 = dpdx(world_pos);
    let dp2 = dpdy(world_pos);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);

    let dp2perp = cross(dp2, world_normal);
    let dp1perp = cross(world_normal, dp1);

    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;

    let inv_max = inverseSqrt(max(dot(t, t), dot(b, b)));

    return mat3x3<f32>(t * inv_max, b * inv_max, world_normal);
}

// Get the final normal, applying normal map if present
fn get_normal(world_pos: vec3<f32>, world_normal: vec3<f32>, tex_coords: vec2<f32>) -> vec3<f32> {
    let n = normalize(world_normal);

    if has_normal_texture() {
        // Sample normal map and convert from [0,1] to [-1,1]
        let normal_sample = textureSample(t_normal, s_normal, tex_coords).rgb;
        var map_normal = normal_sample * 2.0 - vec3<f32>(1.0);

        // Apply normal scale
        map_normal.x = map_normal.x * pbr_params.normal_scale;
        map_normal.y = map_normal.y * pbr_params.normal_scale;
        map_normal = normalize(map_normal);

        // Transform from tangent space to world space
        let tbn = get_tbn_matrix(world_pos, n, tex_coords);
        return normalize(tbn * map_normal);
    } else {
        return n;
    }
}
