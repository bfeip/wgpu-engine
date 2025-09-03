use std::mem::size_of;
use cgmath::{Matrix3, Matrix4, Rotation3, Zero};

use crate::{
    material::MaterialId,
};

use super::mesh::Mesh;

pub struct Instance {
    mesh: *const Mesh,
    pub material: MaterialId,
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>
}

impl Instance {
    pub fn new(mesh: *const Mesh, material: MaterialId) -> Self {
        Self {
            mesh,
            material,
            position: cgmath::Vector3::zero(),
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0))
        }
    }

    pub fn with_position(mesh: *const Mesh, material: MaterialId, position: cgmath::Vector3<f32>) -> Self {
        Self {
            mesh,
            material,
            position,
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0))
        }
    }

    pub fn to_raw(&self) -> InstanceRaw {
        let translation = Matrix4::from_translation(self.position);
        let transform: Matrix4<f32> = translation * Matrix4::from(self.rotation);
        let rotation = Matrix3::from(self.rotation);

        InstanceRaw {
            transform: transform.into(),
            normal_mat: rotation.into()
        }
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    transform: [[f32; 4]; 4],
    normal_mat: [[f32; 3]; 3]
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