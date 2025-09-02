use std::{collections::HashMap, fs::File, io::BufReader, path::Path};
use cgmath::{Matrix3, Matrix4, Rotation3, Zero};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{material::MaterialId, VertexShaderLocations};

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