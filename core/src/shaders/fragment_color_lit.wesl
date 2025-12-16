// fragment_color_lit.wesl
// Fragment shader for lit color materials (faces with lighting)

import package::common::VertexOutput;
import package::material_color::material_color;
import package::lighting::{lights, calculate_diffuse_lighting};

// Fragment shader with color and lighting
fn fragment_color_lit(in: VertexOutput) -> vec4<f32> {
    let diffuse_color = calculate_diffuse_lighting(in.world_position, in.world_normal);
    let result = diffuse_color * material_color.rgb;
    return vec4<f32>(result, material_color.a);
}
