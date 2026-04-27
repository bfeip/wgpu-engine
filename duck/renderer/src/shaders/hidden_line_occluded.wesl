// hidden_line_occluded.wesl
// Gray line shader for the hidden-line occluded pass.
// Renders line geometry with the same camera transform as hidden_line_solid,
// outputting a flat gray. The pipeline uses depth compare Greater so only
// fragments behind the solid geometry (i.e. occluded lines) are drawn.

import package::common::{VertexInput, InstanceInput};
import package::camera::camera;

@vertex
fn vs_occluded(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
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
fn fs_occluded() -> @location(0) vec4<f32> {
    return vec4<f32>(0.6, 0.6, 0.6, 1.0);
}
