// outline_mask.wesl
// Shader for rendering selected objects to a mask texture
// Outputs solid 1.0 to R channel for selected object pixels

import package::common::{VertexInput, InstanceInput, CameraUniform};

// Camera uniform at binding 0
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Vertex shader - standard transform, no normal expansion
@vertex
fn vs_mask(
    model: VertexInput,
    instance: InstanceInput
) -> @builtin(position) vec4<f32> {
    // Reconstruct transform matrix from instance attributes
    let transform = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );

    // Transform position to world space
    let world_position = transform * vec4<f32>(model.position, 1.0);

    // Project to clip space
    return camera.view_proj * world_position;
}

// Fragment shader - output solid 1.0 to mask
@fragment
fn fs_mask() -> @location(0) f32 {
    return 1.0;
}
