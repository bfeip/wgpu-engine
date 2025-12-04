// fragment_texture_lit.wesl
// Fragment shader for lit texture materials (faces with texture and lighting)

use package::common::VertexOutput;
use package::material_texture::{t_diffuse, s_diffuse};
use package::lighting::{lights, calculate_diffuse_lighting};

@fragment
pub fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    let diffuse_color = calculate_diffuse_lighting(in.world_position, in.world_normal);
    let result = diffuse_color * object_color.rgb;
    return vec4<f32>(result, object_color.a);
}
