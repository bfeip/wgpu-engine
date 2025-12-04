// fragment_color_lit.wesl
// Fragment shader for lit color materials (faces with lighting)

use package::common::VertexOutput;
use package::material_color::material_color;
use package::lighting::{lights, calculate_diffuse_lighting};

@fragment
pub fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let diffuse_color = calculate_diffuse_lighting(in.world_position, in.world_normal);
    let result = diffuse_color * material_color.rgb;
    return vec4<f32>(result, material_color.a);
}
