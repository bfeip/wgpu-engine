// fragment_color_unlit.wesl
// Fragment shader for unlit color materials (lines, points - no lighting)

use package::common::VertexOutput;
use package::material_color::material_color;

@fragment
pub fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return material_color;
}
