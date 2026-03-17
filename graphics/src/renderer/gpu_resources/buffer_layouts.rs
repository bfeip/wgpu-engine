use crate::scene::Vertex;

use super::uniforms::GpuInstance;

// Vertex shader attribute locations
pub(crate) enum VertexShaderLocations {
    VertexPosition = 0,
    TextureCoords,
    VertexNormal,
    InstanceTransformRow0,
    InstanceTransformRow1,
    InstanceTransformRow2,
    InstanceTransformRow3,
    InstanceNormalRow0,
    InstanceNormalRow1,
    InstanceNormalRow2,
}

/// Returns the vertex buffer layout for Vertex structs.
///
/// This describes how vertex data is laid out in GPU memory and maps to shader locations.
pub(crate) fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: VertexShaderLocations::VertexPosition as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: VertexShaderLocations::TextureCoords as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3 * 2]>() as wgpu::BufferAddress,
                shader_location: VertexShaderLocations::VertexNormal as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Returns the instance buffer layout for GpuInstance structs.
///
/// This describes how instance data is laid out in GPU memory for instanced rendering.
pub(crate) fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    use VertexShaderLocations as VSL;

    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: VSL::InstanceTransformRow0 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 1]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow1 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 2]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow2 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 3]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow3 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 4]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow0 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; (4 * 4) + (3 * 1)]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow1 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; (4 * 4) + (3 * 2)]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow2 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}
