// point_color_fragment.wgsl
// Fragment shader for point color materials
// Points use solid color without lighting calculations

@group(2) @binding(0)
var<uniform> material_color: vec4<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return material_color;
}
