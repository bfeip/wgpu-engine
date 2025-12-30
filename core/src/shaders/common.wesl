// common.wesl
// Common type definitions used by all shaders

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
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
    @location(7) normal_matrix_0: vec3<f32>,
    @location(8) normal_matrix_1: vec3<f32>,
    @location(9) normal_matrix_2: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>
};
