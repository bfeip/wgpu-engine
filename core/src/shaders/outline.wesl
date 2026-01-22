// outline.wesl
// Shader for rendering selection outlines
// Uses vertex expansion along normals and solid color output

import package::common::{VertexInput, InstanceInput, CameraUniform};

// Camera uniform at binding 0
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Outline configuration uniform
struct OutlineUniform {
    color: vec4<f32>,
    width: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(1) @binding(0)
var<uniform> outline: OutlineUniform;

// Simplified vertex output for outline rendering
struct OutlineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// Vertex shader that expands vertices along normals for outline effect
@vertex
fn vs_outline(
    model: VertexInput,
    instance: InstanceInput
) -> OutlineVertexOutput {
    // Reconstruct transform matrix from instance attributes
    let transform = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );

    // Reconstruct normal matrix from instance attributes
    let normal_mat = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // Transform position to world space
    let world_position = transform * vec4<f32>(model.position, 1.0);

    // Transform normal to world space and normalize
    let world_normal = normalize(normal_mat * model.normal);

    // Expand position along normal by outline width
    let expanded_position = world_position.xyz + world_normal * outline.width;

    var out: OutlineVertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(expanded_position, 1.0);
    return out;
}

// Simple solid color fragment shader for outline
@fragment
fn fs_outline(in: OutlineVertexOutput) -> @location(0) vec4<f32> {
    return outline.color;
}
