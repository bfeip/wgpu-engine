use std::{cell::Cell, fs::File, io::BufReader, path::{Path, PathBuf}};
use cgmath::Point3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    VertexShaderLocations,
    common::{Aabb, Ray},
};

pub type MeshId = u32;
type MeshIndex = u16;

/// Primitive types for mesh rendering
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    TriangleList,
    LineList,
    PointList,
}

/// A collection of indices representing a single primitive type in a mesh
#[derive(Debug, Clone)]
pub struct MeshPrimitive {
    pub primitive_type: PrimitiveType,
    pub indices: Vec<MeshIndex>,
}

/// Result of a ray-mesh intersection test in local mesh space.
#[derive(Debug, Clone)]
pub struct MeshHit {
    /// Distance along the ray to the hit point (in local space)
    pub distance: f32,
    /// Hit location in local mesh space
    pub hit_point: Point3<f32>,
    /// Index of the triangle that was hit (index into the mesh's index buffer / 3)
    pub triangle_index: usize,
    /// Barycentric coordinates of the hit point on the triangle (u, v, w) where w = 1 - u - v
    pub barycentric: (f32, f32, f32),
}

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


pub enum ObjMesh<'a> {
    Bytes(&'a [u8]),
    Path(PathBuf)
}

pub enum MeshDescriptor<'a> {
    Empty,
    Obj(ObjMesh<'a>),
    Raw {
        vertices: Vec<Vertex>,
        primitives: Vec<MeshPrimitive>,
    },
}

pub struct Mesh {
    pub id: MeshId,
    vertices: Vec<Vertex>,
    primitives: Vec<MeshPrimitive>,

    vertex_buffer: wgpu::Buffer,
    // Index buffers for each primitive type (created on-demand)
    triangle_index_buffer: wgpu::Buffer,
    line_index_buffer: wgpu::Buffer,
    point_index_buffer: wgpu::Buffer,

    cached_bounding: Cell<Option<Aabb>>
}

impl Mesh {
    pub fn new(
        id: MeshId,
        device: &wgpu::Device,
        descriptor: MeshDescriptor,
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
            MeshDescriptor::Raw { vertices, primitives } => {
                Ok(Self::from_raw(id, device, vertices, primitives, label))
            }
        }
    }

    fn from_raw(
        id: MeshId,
        device: &wgpu::Device,
        vertices: Vec<Vertex>,
        primitives: Vec<MeshPrimitive>,
        label: Option<&str>
    ) -> Self {
        use wgpu::util::DeviceExt;

        let vertex_buffer_label = label.map(|l| format!("{}_vertex_buffer", l));
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: vertex_buffer_label.as_deref(),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Helper function to create index buffer for a specific primitive type
        let create_index_buffer = |prim_type: PrimitiveType, label_suffix: &str| -> wgpu::Buffer {
            let indices: Vec<MeshIndex> = primitives.iter()
                .filter(|p| p.primitive_type == prim_type)
                .flat_map(|p| p.indices.iter().copied())
                .collect();

            if indices.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: label.map(|l| format!("{}_{}", l, label_suffix)).as_deref(),
                    size: 0,
                    usage: wgpu::BufferUsages::INDEX,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: label.map(|l| format!("{}_{}", l, label_suffix)).as_deref(),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
            }
        };

        let triangle_index_buffer = create_index_buffer(PrimitiveType::TriangleList, "triangle_index_buffer");
        let line_index_buffer = create_index_buffer(PrimitiveType::LineList, "line_index_buffer");
        let point_index_buffer = create_index_buffer(PrimitiveType::PointList, "point_index_buffer");

        Self {
            id,
            vertices,
            primitives,
            vertex_buffer,
            triangle_index_buffer,
            line_index_buffer,
            point_index_buffer,
            cached_bounding: Cell::new(None)
        }
    }

    fn new_empty(id: MeshId, device: &wgpu::Device, label: Option<&str>) -> Self {
        let vertices = Vec::new();
        let primitives = Vec::new();

        let vertex_buffer_label = label.and_then(| mesh_label | {
            Some(mesh_label.to_owned() + "_vertex_buffer")
        });
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: vertex_buffer_label.as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false
        });

        // Create empty index buffers for all primitive types
        let triangle_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_triangle_index_buffer", l)).as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: false
        });

        let line_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_line_index_buffer", l)).as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: false
        });

        let point_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: label.map(|l| format!("{}_point_index_buffer", l)).as_deref(),
            size: 0,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: false
        });

        Self {
            id,
            vertices,
            primitives,
            vertex_buffer,
            triangle_index_buffer,
            line_index_buffer,
            point_index_buffer,
            cached_bounding: Cell::new(None)
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

        // OBJ files contain triangles, so create a triangle primitive
        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: obj.indices,
        }];

        Ok(Self::from_raw(id, device, vertices, primitives, label))
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
        primitive_type: PrimitiveType,
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

        // Select the appropriate index buffer based on primitive type
        let (index_buffer, n_indices) = match primitive_type {
            PrimitiveType::TriangleList => {
                let count = self.primitives.iter()
                    .filter(|p| p.primitive_type == PrimitiveType::TriangleList)
                    .map(|p| p.indices.len())
                    .sum::<usize>() as u32;
                (&self.triangle_index_buffer, count)
            }
            PrimitiveType::LineList => {
                let count = self.primitives.iter()
                    .filter(|p| p.primitive_type == PrimitiveType::LineList)
                    .map(|p| p.indices.len())
                    .sum::<usize>() as u32;
                (&self.line_index_buffer, count)
            }
            PrimitiveType::PointList => {
                let count = self.primitives.iter()
                    .filter(|p| p.primitive_type == PrimitiveType::PointList)
                    .map(|p| p.indices.len())
                    .sum::<usize>() as u32;
                (&self.point_index_buffer, count)
            }
        };

        // Skip drawing if there are no indices for this primitive type
        if n_indices == 0 {
            return;
        }

        // Draw
        let n_instances = instance_transforms.len() as u32;
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..n_indices, 0, 0..n_instances);
    }
    
    /// Computes the local-space axis-aligned bounding box for a mesh.
    /// Returns None if the mesh has no vertices.
    pub fn bounding(&self) -> Option<Aabb> {
        let cached_bounding = self.cached_bounding.get();
        if cached_bounding.is_some() {
            // We only have to compute a bounding once per mesh unless we make
            // meshes mutable somehow.
            return cached_bounding
        }

        if self.vertices.is_empty() {
            return None;
        }

        // Extract positions from vertices
        let positions: Vec<Point3<f32>> = self.vertices
            .iter()
            .map(|v| Point3::new(v.position[0], v.position[1], v.position[2]))
            .collect();

        let bounding = Aabb::from_points(&positions);
        self.cached_bounding.set(bounding);
        bounding
    }

    /// Returns a reference to the mesh's vertex data.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Returns a reference to the mesh's primitives.
    pub fn primitives(&self) -> &[MeshPrimitive] {
        &self.primitives
    }

    /// Returns true if this mesh has any primitives of the specified type.
    pub fn has_primitive_type(&self, primitive_type: PrimitiveType) -> bool {
        self.primitives.iter().any(|p| p.primitive_type == primitive_type)
    }

    /// Gets the triangle indices (for backward compatibility and ray intersection).
    fn triangle_indices(&self) -> Vec<MeshIndex> {
        self.primitives.iter()
            .filter(|p| p.primitive_type == PrimitiveType::TriangleList)
            .flat_map(|p| p.indices.iter().copied())
            .collect()
    }

    /// Tests a ray against all triangles in the mesh.
    ///
    /// The ray should be in local mesh space. Returns all intersections found,
    /// unsorted (caller can sort by distance if needed).
    pub fn intersect_ray(&self, ray: &Ray) -> Vec<MeshHit> {
        let mut hits = Vec::new();

        // Get all triangle indices
        let triangle_indices = self.triangle_indices();

        // Iterate through triangles (indices come in groups of 3)
        for triangle_index in 0..(triangle_indices.len() / 3) {
            let i0 = triangle_indices[triangle_index * 3] as usize;
            let i1 = triangle_indices[triangle_index * 3 + 1] as usize;
            let i2 = triangle_indices[triangle_index * 3 + 2] as usize;

            let v0 = Point3::from(self.vertices[i0].position);
            let v1 = Point3::from(self.vertices[i1].position);
            let v2 = Point3::from(self.vertices[i2].position);

            // Test ray-triangle intersection
            if let Some((t, u, v)) = ray.intersect_triangle(v0, v1, v2) {
                let w = 1.0 - u - v;
                hits.push(MeshHit {
                    distance: t,
                    hit_point: ray.point_at(t),
                    triangle_index,
                    barycentric: (u, v, w),
                });
            }
        }

        hits
    }
}