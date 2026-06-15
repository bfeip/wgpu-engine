// vertex.wesl
// Standard vertex shader with instancing support

import package::common::{VertexInput, InstanceInput, VertexOutput};
import package::camera::camera;

// Position-only transform: reconstruct the instance model matrix and project to
// clip space. Shared by passes that need only a clip position (flat color,
// outline mask) and don't expand normals or texcoords.
fn vertex_position_only(model: VertexInput, instance: InstanceInput) -> vec4<f32> {
    let transform = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );
    let world_position = transform * vec4<f32>(model.position, 1.0);
    return camera.view_proj * world_position;
}

// Standard vertex transformation
fn vertex_standard(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    // Reconstruct transform matrix from instance attributes
    let transform = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );

    // Reconstruct normal matrix from instance attributes
    let normal_mat = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    var out: VertexOutput;
    out.tex_coords = model.tex_coords.xy; // Third coord unused for now
    out.world_normal = normal_mat * model.normal;
    let world_position: vec4<f32> = transform * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}
