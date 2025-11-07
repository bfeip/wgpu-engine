use std::mem::size_of;

use crate::material::MaterialId;
use super::mesh::MeshId;

pub type InstanceId = u32;

/// An instance references a mesh and material to be rendered.
pub struct Instance {
    pub id: InstanceId,
    pub mesh: MeshId,
    pub material: MaterialId,
}

impl Instance {
    /// Creates a new instance referencing the given mesh and material.
    pub fn new(id: InstanceId, mesh: MeshId, material: MaterialId) -> Self {
        Self {
            id,
            mesh,
            material,
        }
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub transform: [[f32; 4]; 4],
    pub normal_mat: [[f32; 3]; 3]
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: crate::VertexShaderLocations::InstanceTransformRow0 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*1]>() as wgpu::BufferAddress,
                    shader_location: crate::VertexShaderLocations::InstanceTransformRow1 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*2]>() as wgpu::BufferAddress,
                    shader_location: crate::VertexShaderLocations::InstanceTransformRow2 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*3]>() as wgpu::BufferAddress,
                    shader_location: crate::VertexShaderLocations::InstanceTransformRow3 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },

                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*4]>() as wgpu::BufferAddress,
                    shader_location: crate::VertexShaderLocations::InstanceNormalRow0 as u32,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; (4*4) + (3*1)]>() as wgpu::BufferAddress,
                    shader_location: crate::VertexShaderLocations::InstanceNormalRow1 as u32,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; (4*4) + (3*2)]>() as wgpu::BufferAddress,
                    shader_location: crate::VertexShaderLocations::InstanceNormalRow2 as u32,
                    format: wgpu::VertexFormat::Float32x3
                },
            ]
        }
    }
}