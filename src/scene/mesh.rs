use std::{cell::Cell, fs::File, io::BufReader, path::{Path, PathBuf}};
use cgmath::Point3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    drawstate::VertexShaderLocations,
    common::{Aabb, ConvexPolyhedron, Ray},
};

/// Unique identifier for a mesh in the scene.
pub type MeshId = u32;

/// Index type used for mesh index buffers (u16 supports up to 65,536 vertices per mesh).
pub type MeshIndex = u16;

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

/// Result of a volume-mesh intersection test in local mesh space.
#[derive(Debug, Clone)]
pub struct MeshVolumeHit {
    /// Indices of triangles that intersect the volume
    pub triangle_indices: Vec<usize>,
    /// True if all triangles in the mesh are fully contained within the volume
    pub fully_contained: bool,
}

/// GPU-compatible vertex structure containing position, texture coordinates, and normal.
///
/// This struct is laid out in memory to match the vertex shader's expectations.
/// Each vertex is 36 bytes: 12 bytes position + 12 bytes tex_coords + 12 bytes normal.
///
/// # Memory Layout
/// - Uses `#[repr(C)]` for predictable layout
/// - Implements `Pod` and `Zeroable` for zero-copy GPU buffer uploads
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Vertex position in local mesh space [x, y, z]
    pub position: [f32; 3],
    /// Texture coordinates [u, v, w] (w unused, reserved for 3D textures)
    pub tex_coords: [f32; 3],
    /// Vertex normal vector [x, y, z]
    pub normal: [f32; 3]
}

impl Vertex {
    /// Returns the vertex buffer layout descriptor for the rendering pipeline.
    ///
    /// This describes how vertex data is laid out in GPU memory and maps to shader locations
    /// defined in `VertexShaderLocations`.
    pub(crate) fn desc() -> wgpu::VertexBufferLayout<'static> {
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

/// Source data for loading a Wavefront OBJ mesh.
///
/// OBJ meshes can be loaded from either in-memory bytes or a file path.
pub enum ObjMesh<'a> {
    /// OBJ data from an in-memory byte slice
    Bytes(&'a [u8]),
    /// OBJ data from a file path
    Path(PathBuf)
}

/// Descriptor for creating a mesh from various sources.
///
/// Meshes can be created as empty containers, loaded from OBJ files,
/// or constructed from raw vertex and primitive data.
pub enum MeshDescriptor<'a> {
    /// Creates an empty mesh with no vertices or primitives
    Empty,
    /// Loads mesh data from a Wavefront OBJ file
    Obj(ObjMesh<'a>),
    /// Creates mesh from raw vertex and primitive data
    Raw {
        /// Vertex data for the mesh
        vertices: Vec<Vertex>,
        /// Primitives (triangle lists, line lists, etc.) referencing the vertices
        primitives: Vec<MeshPrimitive>,
    },
}

/// A renderable mesh containing geometry data and GPU buffers.
///
/// Meshes store vertex data (positions, normals, texture coordinates) and primitives
/// (triangle lists, line lists, point lists) along with their corresponding GPU buffers.
///
/// # Features
/// - Multiple primitive types per mesh (triangles, lines, points)
/// - GPU buffer management for vertices and indices
/// - Local-space bounding box computation with caching
/// - Ray-mesh intersection testing for picking
/// - Instance rendering support
///
/// # Memory Management
/// - Vertex and index buffers are created at mesh construction time
/// - Separate index buffers maintained for each primitive type
/// - Bounding boxes computed lazily and cached
pub struct Mesh {
    /// Unique identifier for this mesh
    pub id: MeshId,
    /// CPU-side vertex data
    vertices: Vec<Vertex>,
    /// CPU-side primitive data (index lists grouped by type)
    primitives: Vec<MeshPrimitive>,

    /// GPU vertex buffer
    vertex_buffer: wgpu::Buffer,
    /// GPU index buffer for triangle primitives
    triangle_index_buffer: wgpu::Buffer,
    /// GPU index buffer for line primitives
    line_index_buffer: wgpu::Buffer,
    /// GPU index buffer for point primitives
    point_index_buffer: wgpu::Buffer,

    /// Cached local-space axis-aligned bounding box
    cached_bounding: Cell<Option<Aabb>>
}

impl Mesh {
    /// Creates a new mesh from a descriptor.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this mesh
    /// * `device` - WGPU device for creating GPU buffers
    /// * `descriptor` - Source data for the mesh (empty, OBJ file, or raw data)
    /// * `label` - Optional label for debugging GPU resources
    ///
    /// # Returns
    /// The created mesh or an error if loading fails (e.g., invalid OBJ file)
    pub(crate) fn new(
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

    /// Creates a mesh from raw vertex and primitive data.
    ///
    /// This method creates GPU buffers for all vertex data and separates primitives
    /// into type-specific index buffers (triangles, lines, points).
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this mesh
    /// * `device` - WGPU device for creating GPU buffers
    /// * `vertices` - Vertex data (positions, normals, texture coordinates)
    /// * `primitives` - Primitive data (index lists grouped by type)
    /// * `label` - Optional label for debugging GPU resources
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

    /// Creates an empty mesh with no vertices or primitives.
    ///
    /// All GPU buffers are created with zero size. This is useful for placeholder
    /// meshes or meshes that will be populated later.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this mesh
    /// * `device` - WGPU device for creating GPU buffers
    /// * `label` - Optional label for debugging GPU resources
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

    /// Creates a mesh from a parsed OBJ object.
    ///
    /// Converts OBJ vertex data to the engine's vertex format and creates a single
    /// triangle list primitive. OBJ files always contain triangle geometry.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this mesh
    /// * `device` - WGPU device for creating GPU buffers
    /// * `obj` - Parsed OBJ object containing vertices and indices
    /// * `label` - Optional label for debugging GPU resources
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

    /// Loads a mesh from OBJ data in a byte slice.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this mesh
    /// * `device` - WGPU device for creating GPU buffers
    /// * `obj_bytes` - OBJ file data as bytes
    /// * `label` - Optional label for debugging GPU resources
    ///
    /// # Errors
    /// Returns an error if the OBJ data is malformed or cannot be parsed
    fn from_obj_bytes(id: MeshId, device: &wgpu::Device, obj_bytes: &[u8], label: Option<&str>) -> anyhow::Result<Self> {
        let obj: obj::Obj<obj::TexturedVertex> = obj::load_obj(obj_bytes)?;
        Self::from_obj(id, device, obj, label)
    }

    /// Loads a mesh from an OBJ file on disk.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this mesh
    /// * `device` - WGPU device for creating GPU buffers
    /// * `obj_path` - File path to the OBJ file
    /// * `label` - Optional label for debugging GPU resources
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the OBJ data is malformed
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
    ///
    /// # Arguments
    /// * `device` - WGPU device for creating the instance buffer
    /// * `pass` - Render pass to record draw commands into
    /// * `primitive_type` - Type of primitives to draw (triangles, lines, or points)
    /// * `instance_transforms` - Pre-computed world transforms and normal matrices for each instance
    ///
    /// # Notes
    /// - Creates a new instance buffer each call (not cached)
    /// - Skips drawing if no primitives of the requested type exist
    /// - Uses indexed drawing with instancing for efficiency
    pub(crate) fn draw_instances(
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

    /// Extracts all triangle indices from the mesh.
    ///
    /// Collects indices from all triangle list primitives in the mesh into a single vector.
    /// Each group of 3 indices defines one triangle.
    ///
    /// # Returns
    /// A vector of indices for all triangles. Empty if the mesh contains no triangle primitives.
    pub fn triangle_indices(&self) -> Vec<MeshIndex> {
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

    /// Tests a convex volume against all triangles in the mesh.
    ///
    /// The volume should be in local mesh space. Returns information about which
    /// triangles intersect the volume and whether the entire mesh is contained.
    ///
    /// # Arguments
    /// * `volume` - The convex polyhedron to test against (in local mesh space)
    /// * `thorough` - If true, uses more accurate but slower edge-triangle intersection tests
    ///
    /// # Returns
    /// `Some(MeshVolumeHit)` if any triangles intersect the volume, `None` otherwise.
    pub fn intersect_volume(
        &self,
        volume: &ConvexPolyhedron,
        thorough: bool,
    ) -> Option<MeshVolumeHit> {
        let triangle_indices_data = self.triangle_indices();
        let num_triangles = triangle_indices_data.len() / 3;

        if num_triangles == 0 {
            return None;
        }

        let mut hit_indices = Vec::new();
        let mut all_fully_contained = true;

        for triangle_index in 0..num_triangles {
            let i0 = triangle_indices_data[triangle_index * 3] as usize;
            let i1 = triangle_indices_data[triangle_index * 3 + 1] as usize;
            let i2 = triangle_indices_data[triangle_index * 3 + 2] as usize;

            let v0 = Point3::from(self.vertices[i0].position);
            let v1 = Point3::from(self.vertices[i1].position);
            let v2 = Point3::from(self.vertices[i2].position);

            // Check if triangle is fully contained
            let fully_inside = volume.contains_triangle(v0, v1, v2);

            if fully_inside {
                hit_indices.push(triangle_index);
            } else if volume.intersects_triangle(v0, v1, v2, thorough) {
                // Triangle intersects but is not fully contained
                hit_indices.push(triangle_index);
                all_fully_contained = false;
            } else {
                // Triangle doesn't intersect at all - mesh is not fully contained
                all_fully_contained = false;
            }
        }

        if hit_indices.is_empty() {
            None
        } else {
            Some(MeshVolumeHit {
                triangle_indices: hit_indices,
                fully_contained: all_fully_contained,
            })
        }
    }
}