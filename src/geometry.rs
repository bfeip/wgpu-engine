use std::{fs::File, io::BufReader, path::Path};
use wgpu::util::DeviceExt;

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
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
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
}