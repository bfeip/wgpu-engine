// Gooch shading — warm/cool color interpolation based on surface normal direction.
//
// Engine types come from package::common, so this shader stays in sync with the
// engine automatically. If VertexInput or LightUniform change, the WESL compiler
// catches the mismatch rather than silently producing a broken render.

import package::common::{VertexInput, InstanceInput, LIGHT_TYPE_DIRECTIONAL};
import package::camera::camera;    // CameraUniform global at @group(0) @binding(0)
import package::lighting::lights;  // LightsUniform global at @group(1) @binding(0)

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal:   vec3<f32>,
    @location(1) world_position: vec3<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        inst.model_matrix_0, inst.model_matrix_1,
        inst.model_matrix_2, inst.model_matrix_3,
    );
    let normal_mat = mat3x3<f32>(
        inst.normal_matrix_0, inst.normal_matrix_1, inst.normal_matrix_2,
    );
    let world_pos = model * vec4<f32>(vert.position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal   = normalize(normal_mat * vert.normal);
    out.world_position = world_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let warm = vec3<f32>(0.95, 0.75, 0.2);  // lit side: warm gold
    let cool = vec3<f32>(0.0,  0.2,  0.7);  // shadow side: cool blue

    // Determine light direction: use the first scene light if present,
    // otherwise fall back to a default directional light.
    var light_dir = normalize(vec3<f32>(0.5, 1.0, 0.5));
    if lights.light_count > 0u {
        let L = lights.lights[0];
        if L.light_type == LIGHT_TYPE_DIRECTIONAL {
            light_dir = normalize(-L.direction);
        } else {
            light_dir = normalize(L.position - in.world_position);
        }
    }

    // Gooch interpolation: remap n·l from [-1, 1] → [0, 1]
    let n_dot_l = dot(in.world_normal, light_dir);
    let t = (n_dot_l + 1.0) * 0.5;
    let gooch = mix(cool, warm, t);

    // Silhouette darkening: fade toward the cool color at grazing angles
    let view_dir = normalize(camera.eye_position - in.world_position);
    let n_dot_v  = max(dot(in.world_normal, view_dir), 0.0);
    let edge     = smoothstep(0.0, 0.35, n_dot_v);
    let color    = mix(cool * 0.4, gooch, edge);

    return vec4<f32>(color, 1.0);
}
