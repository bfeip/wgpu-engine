use std::{cell::Cell, fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use cgmath::Point3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    common::{Aabb, ConvexPolyhedron, Ray},
    drawstate::VertexShaderLocations,
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
    pub normal: [f32; 3],
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
                    offset: std::mem::size_of::<[f32; 3 * 2]>() as wgpu::BufferAddress,
                    shader_location: VertexShaderLocations::VertexNormal as u32,
                    format: wgpu::VertexFormat::Float32x3,
                },
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
    Path(&'a Path),
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

/// GPU resources for a mesh (vertex and index buffers).
///
/// These are created lazily when the mesh is first needed for rendering.
pub(crate) struct MeshGpuResources {
    /// GPU vertex buffer
    pub vertex_buffer: wgpu::Buffer,
    /// GPU index buffer for triangle primitives
    pub triangle_index_buffer: wgpu::Buffer,
    /// GPU index buffer for line primitives
    pub line_index_buffer: wgpu::Buffer,
    /// GPU index buffer for point primitives
    pub point_index_buffer: wgpu::Buffer,
}

/// Meshes store vertex data (positions, normals, texture coordinates) and primitives
/// (triangle lists, line lists, point lists). GPU buffers are created lazily when
/// the mesh is first rendered.
///
/// # Examples
///
/// ```ignore
/// // Create from raw data (no device needed)
/// let mesh = Mesh::from_raw(vertices, primitives);
///
/// // Add to scene
/// let mesh_id = scene.add_mesh(mesh);
///
/// // GPU resources are created automatically during rendering
/// ```
pub struct Mesh {
    /// Unique identifier for this mesh (assigned by Scene)
    pub id: MeshId,
    /// CPU-side vertex data
    vertices: Vec<Vertex>,
    /// CPU-side primitive data (index lists grouped by type)
    primitives: Vec<MeshPrimitive>,
    /// GPU resources (created lazily)
    gpu: Option<MeshGpuResources>,
    /// True if vertex/primitive data changed since last GPU upload
    dirty: bool,
    /// Cached local-space axis-aligned bounding box
    cached_bounding: Cell<Option<Aabb>>,
}

impl Mesh {
    /// Creates a new empty mesh with no vertices or primitives.
    ///
    /// No GPU resources are allocated until the mesh is rendered.
    pub fn new() -> Self {
        Self {
            id: 0, // Assigned by Scene
            vertices: Vec::new(),
            primitives: Vec::new(),
            gpu: None,
            dirty: true,
            cached_bounding: Cell::new(None),
        }
    }

    /// Creates a mesh from raw vertex and primitive data.
    ///
    /// No GPU resources are allocated until the mesh is rendered.
    ///
    /// # Arguments
    /// * `vertices` - Vertex data (positions, normals, texture coordinates)
    /// * `primitives` - Primitive data (index lists grouped by type)
    pub fn from_raw(vertices: Vec<Vertex>, primitives: Vec<MeshPrimitive>) -> Self {
        Self {
            id: 0, // Assigned by Scene
            vertices,
            primitives,
            gpu: None,
            dirty: true,
            cached_bounding: Cell::new(None),
        }
    }

    /// Creates a mesh from a descriptor.
    ///
    /// No GPU resources are allocated until the mesh is rendered.
    ///
    /// # Arguments
    /// * `descriptor` - Source data for the mesh (empty, OBJ file, or raw data)
    ///
    /// # Errors
    /// Returns an error if loading from OBJ fails.
    pub fn from_descriptor(descriptor: MeshDescriptor) -> Result<Self> {
        match descriptor {
            MeshDescriptor::Empty => Ok(Self::new()),
            MeshDescriptor::Obj(obj_desc) => match obj_desc {
                ObjMesh::Bytes(bytes) => Self::from_obj_bytes(bytes),
                ObjMesh::Path(path) => Self::from_obj_path(path),
            },
            MeshDescriptor::Raw { vertices, primitives } => Ok(Self::from_raw(vertices, primitives)),
        }
    }

    /// Loads a mesh from OBJ data in a byte slice.
    ///
    /// # Arguments
    /// * `obj_bytes` - OBJ file data as bytes
    ///
    /// # Errors
    /// Returns an error if the OBJ data is malformed or cannot be parsed
    pub fn from_obj_bytes(obj_bytes: &[u8]) -> Result<Self> {
        let obj: obj::Obj<obj::TexturedVertex> =
            obj::load_obj(obj_bytes).context("Failed to parse OBJ data")?;
        Self::from_obj(obj)
    }

    /// Loads a mesh from an OBJ file on disk.
    ///
    /// # Arguments
    /// * `obj_path` - File path to the OBJ file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the OBJ data is malformed
    pub fn from_obj_path<P: AsRef<Path>>(obj_path: P) -> Result<Self> {
        let path = obj_path.as_ref();
        let obj_file =
            File::open(path).with_context(|| format!("Failed to open OBJ file: {:?}", path))?;
        let obj_reader = BufReader::new(obj_file);
        let obj: obj::Obj<obj::TexturedVertex> =
            obj::load_obj(obj_reader).with_context(|| format!("Failed to parse OBJ file: {:?}", path))?;
        Self::from_obj(obj)
    }

    /// Creates a mesh from a parsed OBJ object.
    fn from_obj(obj: obj::Obj<obj::TexturedVertex>) -> Result<Self> {
        let vertices: Vec<Vertex> = obj
            .vertices
            .iter()
            .map(|v: &obj::TexturedVertex| Vertex {
                position: v.position,
                tex_coords: v.texture,
                normal: v.normal,
            })
            .collect();

        // OBJ files contain triangles, so create a triangle primitive
        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: obj.indices,
        }];

        Ok(Self::from_raw(vertices, primitives))
    }

    // ========== Mutation methods (set dirty flag) ==========

    /// Set the mesh's vertex data, marking it as dirty.
    pub fn set_vertices(&mut self, vertices: Vec<Vertex>) {
        self.vertices = vertices;
        self.dirty = true;
        self.cached_bounding.set(None);
    }

    /// Set the mesh's primitive data, marking it as dirty.
    pub fn set_primitives(&mut self, primitives: Vec<MeshPrimitive>) {
        self.primitives = primitives;
        self.dirty = true;
    }

    /// Add vertices to the mesh, marking it as dirty.
    pub fn add_vertices(&mut self, vertices: &[Vertex]) {
        self.vertices.extend_from_slice(vertices);
        self.dirty = true;
        self.cached_bounding.set(None);
    }

    /// Add a primitive to the mesh, marking it as dirty.
    pub fn add_primitive(&mut self, primitive: MeshPrimitive) {
        self.primitives.push(primitive);
        self.dirty = true;
    }

    /// Mark the mesh as dirty, requiring GPU resource update.
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    // ========== GPU resource management ==========

    /// Check if GPU resources need to be created or updated.
    pub(crate) fn needs_gpu_upload(&self) -> bool {
        self.gpu.is_none() || self.dirty
    }

    /// Check if this mesh has GPU resources initialized.
    pub(crate) fn has_gpu_resources(&self) -> bool {
        self.gpu.is_some()
    }

    /// Create or update GPU resources for this mesh.
    ///
    /// This method is called automatically by `DrawState::prepare_scene()` before rendering.
    /// After this call, `gpu()` can be used to access the GPU resources.
    pub(crate) fn ensure_gpu_resources(&mut self, device: &wgpu::Device) {
        if !self.needs_gpu_upload() {
            return;
        }

        // Create vertex buffer
        let vertex_buffer = if self.vertices.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh Vertex Buffer"),
                size: 0,
                usage: wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
        };

        // Helper to create index buffer for a specific primitive type
        let create_index_buffer = |prim_type: PrimitiveType, label: &str| -> wgpu::Buffer {
            let indices: Vec<MeshIndex> = self
                .primitives
                .iter()
                .filter(|p| p.primitive_type == prim_type)
                .flat_map(|p| p.indices.iter().copied())
                .collect();

            if indices.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 0,
                    usage: wgpu::BufferUsages::INDEX,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
            }
        };

        let triangle_index_buffer = create_index_buffer(PrimitiveType::TriangleList, "Triangle Index Buffer");
        let line_index_buffer = create_index_buffer(PrimitiveType::LineList, "Line Index Buffer");
        let point_index_buffer = create_index_buffer(PrimitiveType::PointList, "Point Index Buffer");

        self.gpu = Some(MeshGpuResources {
            vertex_buffer,
            triangle_index_buffer,
            line_index_buffer,
            point_index_buffer,
        });
        self.dirty = false;
    }

    /// Get the GPU resources for this mesh.
    ///
    /// # Panics
    /// Panics if GPU resources haven't been initialized yet.
    /// Call `ensure_gpu_resources()` first, or use `has_gpu_resources()` to check.
    pub(crate) fn gpu(&self) -> &MeshGpuResources {
        self.gpu
            .as_ref()
            .expect("Mesh GPU resources not initialized. Call ensure_gpu_resources() first.")
    }

    // ========== Query methods ==========

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
        self.primitives
            .iter()
            .any(|p| p.primitive_type == primitive_type)
    }

    /// Extracts all triangle indices from the mesh.
    ///
    /// Collects indices from all triangle list primitives in the mesh into a single vector.
    /// Each group of 3 indices defines one triangle.
    ///
    /// # Returns
    /// A vector of indices for all triangles. Empty if the mesh contains no triangle primitives.
    pub fn triangle_indices(&self) -> Vec<MeshIndex> {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::TriangleList)
            .flat_map(|p| p.indices.iter().copied())
            .collect()
    }

    /// Get the count of indices for a primitive type.
    pub fn index_count(&self, primitive_type: PrimitiveType) -> u32 {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == primitive_type)
            .map(|p| p.indices.len())
            .sum::<usize>() as u32
    }

    /// Computes the local-space axis-aligned bounding box for a mesh.
    /// Returns None if the mesh has no vertices.
    pub fn bounding(&self) -> Option<Aabb> {
        let cached_bounding = self.cached_bounding.get();
        if cached_bounding.is_some() {
            return cached_bounding;
        }

        if self.vertices.is_empty() {
            return None;
        }

        // Extract positions from vertices
        let positions: Vec<Point3<f32>> = self
            .vertices
            .iter()
            .map(|v| Point3::new(v.position[0], v.position[1], v.position[2]))
            .collect();

        let bounding = Aabb::from_points(&positions);
        self.cached_bounding.set(bounding);
        bounding
    }

    // ========== Ray/Volume intersection ==========

    /// Tests a ray against all triangles in the mesh.
    ///
    /// The ray should be in local mesh space. Returns all intersections found,
    /// unsorted (caller can sort by distance if needed).
    pub fn intersect_ray(&self, ray: &Ray) -> Vec<MeshHit> {
        let mut hits = Vec::new();

        let triangle_indices = self.triangle_indices();

        for triangle_index in 0..(triangle_indices.len() / 3) {
            let i0 = triangle_indices[triangle_index * 3] as usize;
            let i1 = triangle_indices[triangle_index * 3 + 1] as usize;
            let i2 = triangle_indices[triangle_index * 3 + 2] as usize;

            let v0 = Point3::from(self.vertices[i0].position);
            let v1 = Point3::from(self.vertices[i1].position);
            let v2 = Point3::from(self.vertices[i2].position);

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

            let fully_inside = volume.contains_triangle(v0, v1, v2);

            if fully_inside {
                hit_indices.push(triangle_index);
            } else if volume.intersects_triangle(v0, v1, v2, thorough) {
                hit_indices.push(triangle_index);
                all_fully_contained = false;
            } else {
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

    // ========== Drawing ==========

    /// Draws instances of this mesh using pre-computed transforms.
    ///
    /// # Panics
    /// Debug assert if GPU resources are out of date or uninitialized
    pub(crate) fn draw_instances(
        &self,
        device: &wgpu::Device,
        pass: &mut wgpu::RenderPass,
        primitive_type: PrimitiveType,
        instance_transforms: &[super::tree::InstanceTransform],
    ) {
        use super::InstanceRaw;

        debug_assert!(!self.needs_gpu_upload(), "Mesh not up to date");
        let gpu = self.gpu();

        // Convert InstanceTransforms to InstanceRaw for the GPU
        let instance_raws: Vec<InstanceRaw> = instance_transforms
            .iter()
            .map(|inst_transform| InstanceRaw {
                transform: inst_transform.world_transform.into(),
                normal_mat: inst_transform.normal_matrix.into(),
            })
            .collect();

        // Create instance buffer
        // TODO: Very wasteful to re-create this every time.
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_raws),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Select the appropriate index buffer based on primitive type
        let (index_buffer, n_indices) = match primitive_type {
            PrimitiveType::TriangleList => (&gpu.triangle_index_buffer, self.index_count(PrimitiveType::TriangleList)),
            PrimitiveType::LineList => (&gpu.line_index_buffer, self.index_count(PrimitiveType::LineList)),
            PrimitiveType::PointList => (&gpu.point_index_buffer, self.index_count(PrimitiveType::PointList)),
        };

        // Skip drawing if there are no indices for this primitive type
        if n_indices == 0 {
            return;
        }

        // Draw
        let n_instances = instance_transforms.len() as u32;
        pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..n_indices, 0, 0..n_instances);
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}
