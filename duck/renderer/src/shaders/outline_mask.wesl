// outline_mask.wesl
// Shader for rendering selected objects to a mask texture.
// Outputs solid 1.0 to R channel for selected object pixels.

import package::common::{VertexInput, InstanceInput};
import package::vertex::vertex_position_only;

// Vertex shader - standard transform, no normal expansion
@vertex
fn vs_mask(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    return vertex_position_only(model, instance);
}

// Fragment shader - output solid 1.0 to mask
@fragment
fn fs_mask() -> @location(0) f32 {
    return 1.0;
}
