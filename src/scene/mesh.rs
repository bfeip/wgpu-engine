use std::{collections::HashMap, fs::File, io::BufReader, path::Path};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    material::MaterialId,
    VertexShaderLocations
};

use super::instance::{Instance, InstanceRaw};

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

pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<MeshIndex>,
    instances: HashMap<MaterialId, Vec<Instance>>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
}

impl Mesh {
    pub fn new(device: &wgpu::Device, label: Option<&str>) -> Self {
        let vertices = Vec::new();
        let indices = Vec::new();
        let instances = HashMap::new();

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

        let instance_buffer_label = label.and_then(| mesh_label | {
            Some(mesh_label.to_owned() + "_instance_buffer")
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: instance_buffer_label.as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });

        Self {
            vertices,
            indices,
            instances,
            vertex_buffer,
            index_buffer,
            instance_buffer,
        }
    }

    pub fn from_obj(device: &wgpu::Device, obj: obj::Obj<obj::TexturedVertex>)-> anyhow::Result<Self> {
        let vertices: Vec<Vertex> = obj.vertices.iter().map(|v: &obj::TexturedVertex| {
            Vertex { 
                position: v.position,
                tex_coords: v.texture,
                normal: v.normal
            }
        }).collect();
        let indices: Vec<u16> = obj.indices;
        let instances = HashMap::new();

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

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            vertices,
            indices,
            instances,
            vertex_buffer,
            index_buffer,
            instance_buffer,
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

    pub fn add_instance(&mut self, instance: Instance) {
        if !self.instances.contains_key(&instance.material) {
            self.instances.insert(instance.material, Vec::new());
        }
        
        let material_instances = self.instances.get_mut(&instance.material).unwrap();
        material_instances.push(instance);
    }

    pub fn get_instances_by_material(&self) -> &HashMap<MaterialId, Vec<Instance>> {
        &self.instances
    }

    // TODO: This function should not exist.
    // We are creating buffers for groups of instances based on material while we are drawing them.
    // We should maintain this data so that we can simply retrieve it when we are drawing.
    // We should not be passing in the instances we want to draw from outside this function.
    pub fn draw_instances(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass,
        instances: &[Instance]
    ) {
        let instance_raws: Vec<InstanceRaw> = instances.iter().map(|instance| {
            instance.to_raw()
        }).collect();
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_raws),
            usage: wgpu::BufferUsages::VERTEX
        });
        let n_indices = self.indices.len() as u32;
        let n_instances = instances.len() as u32;
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..n_indices, 0, 0..n_instances);
    }
}