// flat_color.wesl
// Position-only vertex + material-color fragment shader.
// Used by passes that render geometry in a single flat color driven by a material uniform.

import package::common::{VertexInput, InstanceInput};
import package::camera::camera;
import package::material_color::material_color;

@vertex
fn vs_flat_color(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
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
fn fs_flat_color() -> @location(0) vec4<f32> {
    return material_color;
}
