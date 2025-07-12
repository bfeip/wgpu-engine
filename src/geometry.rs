use std::{fs::File, io::BufReader, ops::Range, path::Path};
use cgmath::{Matrix4, Rotation3, SquareMatrix, Zero};
use wgpu::util::DeviceExt;

use crate::ShaderLocations;

type MeshIndex = u16;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2]
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: ShaderLocations::VertexPosition as u32,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: ShaderLocations::TextureCoords as u32,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<MeshIndex>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer
}

impl Mesh {
    pub fn new(device: &wgpu::Device, label: Option<&str>) -> Self {
        let vertices = Vec::new();
        let indices = Vec::new();
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: 0,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size: 0,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: false
        });

        Self {
            vertices,
            indices,
            vertex_buffer,
            index_buffer
        }
    }

    pub fn from_obj(device: &wgpu::Device, obj: obj::Obj<obj::TexturedVertex>)-> anyhow::Result<Self> {
        let vertices: Vec<Vertex> = obj.vertices.iter().map(|v: &obj::TexturedVertex| {
            Vertex { position: v.position, tex_coords: v.texture[..2].try_into().unwrap() }
        }).collect();
        let indices: Vec<u16> = obj.indices;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            vertices,
            indices,
            vertex_buffer,
            index_buffer
        })
    }

    pub fn from_obj_bytes(device: &wgpu::Device, obj_bytes: &[u8]) -> anyhow::Result<Self> {
        let obj: obj::Obj<obj::TexturedVertex> = obj::load_obj(obj_bytes)?;
        Self::from_obj(device, obj)
    }

    pub fn from_obj_path<P: AsRef<Path>>(device: &wgpu::Device, obj_path: P) -> anyhow::Result<Self> {
        let obj_file = File::open(obj_path)?;
        let obj_reader = BufReader::new(obj_file);
        let obj: obj::Obj<obj::TexturedVertex> = obj::load_obj(obj_reader)?;
        Self::from_obj(device, obj)
    }

    pub fn draw_instances(&self, render_pass: &mut wgpu::RenderPass, instances: Range<u32>) {
        let n_indices = self.indices.len() as u32;
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..n_indices, 0, instances);
    }
}


pub struct Instance {
    mesh: *const Mesh,
    pub position: cgmath::Vector3<f32>,
    pub rotation: cgmath::Quaternion<f32>
}

impl Instance {
    pub fn new(mesh: *const Mesh) -> Self {
        Self {
            mesh,
            position: cgmath::Vector3::zero(),
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0))
        }
    }

    pub fn with_position(mesh: *const Mesh, position: cgmath::Vector3<f32>) -> Self {
        Self {
            mesh,
            position,
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(0.0))
        }
    }

    pub fn to_raw(&self) -> InstanceRaw {
        let translation = Matrix4::from_translation(self.position);
        let rotation = Matrix4::from(self.rotation);
        let transform = translation * rotation;

        InstanceRaw {
            transform: transform.into()
        }
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    transform: [[f32; 4]; 4]
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: crate::ShaderLocations::InstanceTransformRow0 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*1]>() as wgpu::BufferAddress,
                    shader_location: crate::ShaderLocations::InstanceTransformRow1 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*2]>() as wgpu::BufferAddress,
                    shader_location: crate::ShaderLocations::InstanceTransformRow2 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*3]>() as wgpu::BufferAddress,
                    shader_location: crate::ShaderLocations::InstanceTransformRow3 as u32,
                    format: wgpu::VertexFormat::Float32x4
                }
            ]
        }
    }
}