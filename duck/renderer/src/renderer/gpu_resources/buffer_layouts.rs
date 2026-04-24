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

/// Returns the vertex buffer layout for [`Vertex`] structs.
///
/// Prefer [`Renderer::custom_pipeline_builder`] for building custom render pipelines —
/// it applies both buffer layouts automatically. These functions are a lower-level
/// escape hatch for cases where the builder's defaults don't fit. Callers that use
/// the layouts directly must also duplicate the engine's shader struct definitions,
/// which will silently break if the engine types change.
pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
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

/// Returns the instance buffer layout for `GpuInstance` structs.
///
/// See [`vertex_buffer_layout`] — the same caveats apply.
pub fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
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
