// hidden_line_solid.wesl
// Flat-white vertex + fragment shader for the hidden-line solid pass.
// Transforms geometry using camera + instance data; outputs opaque white.

import package::common::{VertexInput, InstanceInput};
import package::camera::camera;

@vertex
fn vs_solid(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    let transform = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );
    let world_position = transform * vec4<f32>(model.position, 1.0);
    return camera.view_proj * world_position;
}

@fragment
fn fs_solid() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
