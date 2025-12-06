// fragment_color_unlit.wesl
// Fragment shader for unlit color materials (lines, points - no lighting)

import package::common::VertexOutput;
import package::material_color::material_color;

// Fragment shader with color, no lighting
fn fragment_color_unlit(in: VertexOutput) -> vec4<f32> {
    return material_color;
}
