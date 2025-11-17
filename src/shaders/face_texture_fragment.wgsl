// texture_fragment.wgsl
// Fragment shader for textured materials

@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let light_dir = normalize(lights.position - in.world_position);

    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0) * lights.color.a;
    let diffuse_color = lights.color.rgb * diffuse_strength;

    let result = diffuse_color * object_color.rgb;

    return vec4<f32>(result, object_color.a);
}