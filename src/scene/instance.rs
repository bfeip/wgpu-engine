use std::mem::size_of;
use std::cell::Cell;
use cgmath::{EuclideanSpace, Matrix3, Matrix4, Rotation3};

use crate::{
    material::MaterialId,
};

use super::mesh::MeshId;

pub type InstanceId = u32;

pub struct Instance {
    pub id: InstanceId,

    pub mesh: MeshId,
    pub material: MaterialId,
    position: cgmath::Point3<f32>,
    rotation: cgmath::Quaternion<f32>,

    raw: Cell<InstanceRaw>,
    dirty: bool
}

impl Instance {
    pub fn new(
        id: InstanceId,
        mesh: MeshId,
        material: MaterialId,
        position: Option<cgmath::Point3<f32>>,
        rotation: Option<cgmath::Quaternion<f32>>
    ) -> Self {
        let position = position.unwrap_or(cgmath::Point3::origin());
        let rotation = rotation.unwrap_or(
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0))
        );
        Self {
            id,
            mesh,
            material,
            position,
            rotation,
            raw: Cell::new(unsafe { std::mem::zeroed() }),
            dirty: true
        }
    }

    pub fn to_raw(&self) -> InstanceRaw {
        if !self.dirty {
            return self.raw.get();
        }
        
        let translation = Matrix4::from_translation(self.position.to_vec());
        let transform: Matrix4<f32> = translation * Matrix4::from(self.rotation);
        let rotation = Matrix3::from(self.rotation);

        let raw = InstanceRaw {
            transform: transform.into(),
            normal_mat: rotation.into()
        };
        self.raw.set(raw);

        raw
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