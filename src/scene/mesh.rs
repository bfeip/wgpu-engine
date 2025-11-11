use std::{fs::File, io::BufReader, path::Path};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    VertexShaderLocations
};

pub type MeshId = u32;
type MeshIndex = u16;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 3],
    pub normal: [f32; 3]
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
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
                    offset: std::mem::size_of::<[f32; 3*2]>() as wgpu::BufferAddress,
                    shader_location: VertexShaderLocations::VertexNormal as u32,
                    format: wgpu::VertexFormat::Float32x3
                }
            ],
        }
    }
}


// TODO: I don't like having these generic parameters
// They have to be specified for every branch of this enum, even when irrelevant
pub enum ObjMesh<'a, P: AsRef<Path>> {
    Bytes(&'a [u8]),
    Path(P)
}

pub enum MeshDescriptor<'a, P: AsRef<Path>> {
    Empty,
    Obj(ObjMesh<'a, P>),
    Raw {
        vertices: Vec<Vertex>,
        indices: Vec<MeshIndex>,
    },
}

pub struct Mesh {
    pub id: MeshId,
    vertices: Vec<Vertex>,
    indices: Vec<MeshIndex>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

impl Mesh {
    pub fn new<P: AsRef<Path>>(
        id: MeshId,
        device: &wgpu::Device,
        descriptor: MeshDescriptor<P>,
        label: Option<&str>
    ) -> anyhow::Result<Self> {
        match descriptor {
            MeshDescriptor::Empty => Ok(Self::new_empty(id, device, label)),
            MeshDescriptor::Obj(obj_desc) => {
                match obj_desc {
                    ObjMesh::Bytes(bytes) => Self::from_obj_bytes(id, device, &bytes, label),
                    ObjMesh::Path(path) => Self::from_obj_path(id, device, path, label)
                }
            }
            MeshDescriptor::Raw { vertices, indices } => {
                Ok(Self::from_raw(id, device, vertices, indices, label))
            }
        }
    }

    fn from_raw(
        id: MeshId,
        device: &wgpu::Device,
        vertices: Vec<Vertex>,
        indices: Vec<MeshIndex>,
        label: Option<&str>
    ) -> Self {
        use wgpu::util::DeviceExt;

        let vertex_buffer_label = label.map(|l| format!("{}_vertex_buffer", l));
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: vertex_buffer_label.as_deref(),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer_label = label.map(|l| format!("{}_index_buffer", l));
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: index_buffer_label.as_deref(),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            id,
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
        }
    }

    fn new_empty(id: MeshId, device: &wgpu::Device, label: Option<&str>) -> Self {
        let vertices = Vec::new();
        let indices = Vec::new();

        let vertex_buffer_label = label.and_then(| mesh_label | {
            Some(mesh_label.to_owned() + "_vertex_buffer")
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: vertex_buffer_label.as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });

        let index_buffer_label = label.and_then(| mesh_label | {
            Some(mesh_label.to_owned() + "_index_buffer")
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: index_buffer_label.as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: false
        });

        Self {
            id,
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
        }
    }

    fn from_obj(
        id: MeshId,
        device: &wgpu::Device,
        obj: obj::Obj<obj::TexturedVertex>,
        label: Option<&str>
    )-> anyhow::Result<Self> {
        let vertices: Vec<Vertex> = obj.vertices.iter().map(|v: &obj::TexturedVertex| {
            Vertex { 
                position: v.position,
                tex_coords: v.texture,
                normal: v.normal
            }
        }).collect();
        let indices: Vec<u16> = obj.indices;

        let vertex_buffer_label = label.and_then(| mesh_label | {
            Some(mesh_label.to_owned() + "_vertex_buffer")
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: vertex_buffer_label.as_deref(),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer_label = label.and_then(| mesh_label | {
            Some(mesh_label.to_owned() + "_index_buffer")
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: index_buffer_label.as_deref(),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            id,
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
        })
    }

    fn from_obj_bytes(id: MeshId, device: &wgpu::Device, obj_bytes: &[u8], label: Option<&str>) -> anyhow::Result<Self> {
        let obj: obj::Obj<obj::TexturedVertex> = obj::load_obj(obj_bytes)?;
        Self::from_obj(id, device, obj, label)
    }

    fn from_obj_path<P: AsRef<Path>>(id: MeshId, device: &wgpu::Device, obj_path: P, label: Option<&str>) -> anyhow::Result<Self> {
        let obj_file = File::open(obj_path)?;
        let obj_reader = BufReader::new(obj_file);
        let obj: obj::Obj<obj::TexturedVertex> = obj::load_obj(obj_reader)?;
        Self::from_obj(id, device, obj, label)
    }

    /// Draws instances of this mesh using pre-computed transforms.
    ///
    /// Transforms are provided by the scene tree traversal and batch collection.
    /// This method creates an instance buffer from the transforms and issues a draw call.
    pub fn draw_instances(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass,
        instance_transforms: &[super::tree::InstanceTransform]
    ) {
        use super::InstanceRaw;

        // Convert InstanceTransforms to InstanceRaw for the GPU
        let instance_raws: Vec<InstanceRaw> = instance_transforms.iter().map(|inst_transform| {
            InstanceRaw {
                transform: inst_transform.world_transform.into(),
                normal_mat: inst_transform.normal_matrix.into(),
            }
        }).collect();

        // Create instance buffer
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_raws),
            usage: wgpu::BufferUsages::VERTEX
        });

        // Draw
        let n_indices = self.indices.len() as u32;
        let n_instances = instance_transforms.len() as u32;
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..n_indices, 0, 0..n_instances);
    }

    /// Returns a reference to the mesh's vertex data.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Returns a reference to the mesh's index data.
    pub fn indices(&self) -> &[MeshIndex] {
        &self.indices
    }
}