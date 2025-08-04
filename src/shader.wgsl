struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct LightsUniform {
    position: vec3<f32>,
    color: vec4<f32>
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec3<f32>,
    @location(2) normal: vec3<f32>
};

struct InstanceInput {
    @location(3) model_matrix_0: vec4<f32>,
    @location(4) model_matrix_1: vec4<f32>,
    @location(5) model_matrix_2: vec4<f32>,
    @location(6) model_matrix_3: vec4<f32>,
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>
};

// VERTEX //////////////////
@group(1) @binding(0)
var<uniform> camera: CameraUniform;
@group(2) @binding(0)
var<uniform> lights: LightsUniform;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    let transform = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );

    var out: VertexOutput;
    out.tex_coords = model.tex_coords.xy; // Third coord unused for now
    out.world_normal = model.normal;
    let world_position: vec4<f32> = transform * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}


// FRAGMENT /////////////////////
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
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