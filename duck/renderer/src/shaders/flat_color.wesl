// flat_color.wesl
// Position-only vertex + material-color fragment shader.
// Used by overlay passes that render geometry in a single flat color driven by a
// `material_color` uniform (camera at group 0, color at group 2).

import package::common::{VertexInput, InstanceInput};
import package::vertex::vertex_position_only;
import package::material_color::material_color;

@vertex
fn vs_flat_color(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    return vertex_position_only(model, instance);
}

@fragment
fn fs_flat_color() -> @location(0) vec4<f32> {
    return material_color;
}
