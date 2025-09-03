// color_fragment.wgsl
// Fragment shader for color materials

@group(2) @binding(0)
var<uniform> material_color: vec4<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(lights.position - in.world_position);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0) * lights.color.a;
    let diffuse_color = lights.color.rgb * diffuse_strength;

    let result = diffuse_color * material_color.rgb;

    return vec4<f32>(result, material_color.a);
}