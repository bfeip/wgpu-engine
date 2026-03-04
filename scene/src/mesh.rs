use std::{cell::Cell, collections::HashSet, fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use cgmath::{InnerSpace, Matrix4, Point3, Transform, Vector3};

mod primitives;

use crate::common::{Aabb, ConvexPolyhedron, Ray};

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

impl MeshPrimitive {
    /// Converts a TriangleList to a LineList with deduplicated edges.
    ///
    /// Each triangle `[a, b, c]` produces edges `[a,b], [b,c], [c,a]`. Shared
    /// edges between adjacent triangles appear only once in the output.
    ///
    /// Returns `None` if this primitive is not a TriangleList.
    pub fn to_line_list(&self) -> Option<MeshPrimitive> {
        if self.primitive_type != PrimitiveType::TriangleList {
            return None;
        }

        let mut edge_set: HashSet<(MeshIndex, MeshIndex)> = HashSet::new();
        let mut line_indices: Vec<MeshIndex> = Vec::new();

        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            for (v0, v1) in [(a, b), (b, c), (c, a)] {
                let edge = if v0 <= v1 { (v0, v1) } else { (v1, v0) };
                if edge_set.insert(edge) {
                    line_indices.push(v0);
                    line_indices.push(v1);
                }
            }
        }

        Some(MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: line_indices,
        })
    }

    /// Converts a TriangleList or LineList to a PointList with unique vertex indices.
    ///
    /// Returns `None` if this primitive is already a PointList.
    pub fn to_point_list(&self) -> Option<MeshPrimitive> {
        if self.primitive_type == PrimitiveType::PointList {
            return None;
        }

        let mut seen: HashSet<MeshIndex> = HashSet::new();
        let mut point_indices: Vec<MeshIndex> = Vec::new();

        for &idx in &self.indices {
            if seen.insert(idx) {
                point_indices.push(idx);
            }
        }

        Some(MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices: point_indices,
        })
    }
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

/// A mesh composed of vertices and primitives
///
/// Meshes store vertex data (positions, normals, texture coordinates) and primitives
/// (triangle lists, line lists, point lists).
///
/// # Examples
///
/// ```
/// use wgpu_engine::scene::{Mesh, MeshPrimitive, Vertex, Scene, PrimitiveType};
///
/// // Create from raw data (no device needed)
/// let vertices = vec![
///     Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [0.5, 1.0, 0.0], tex_coords: [0.5, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
/// ];
/// let primitives = vec![MeshPrimitive {
///     primitive_type: PrimitiveType::TriangleList,
///     indices: vec![0, 1, 2],
/// }];
/// let mesh = Mesh::from_raw(vertices, primitives);
///
/// // Add to scene
/// let mut scene = Scene::new();
/// let mesh_id = scene.add_mesh(mesh);
///
/// // GPU resources are created automatically during rendering
/// ```
#[derive(Clone)]
pub struct Mesh {
    /// Unique identifier for this mesh (assigned by Scene)
    pub id: MeshId,
    /// CPU-side vertex data
    vertices: Vec<Vertex>,
    /// CPU-side primitive data (index lists grouped by type)
    primitives: Vec<MeshPrimitive>,
    /// Generation counter - increments on any mutation (for GPU sync tracking)
    generation: u64,
    /// Cached local-space axis-aligned bounding box
    cached_bounding: Cell<Option<Aabb>>,
}

impl Mesh {
    /// Creates a new empty mesh with no vertices or primitives.
    pub fn new() -> Self {
        Self {
            id: 0, // Assigned by Scene
            vertices: Vec::new(),
            primitives: Vec::new(),
            generation: 1, // Start at 1 so initial sync triggers upload
            cached_bounding: Cell::new(None),
        }
    }

    /// Creates a mesh from raw vertex and primitive data.
    ///
    /// # Arguments
    /// * `vertices` - Vertex data (positions, normals, texture coordinates)
    /// * `primitives` - Primitive data (index lists grouped by type)
    pub fn from_raw(vertices: Vec<Vertex>, primitives: Vec<MeshPrimitive>) -> Self {
        Self {
            id: 0, // Assigned by Scene
            vertices,
            primitives,
            generation: 1,
            cached_bounding: Cell::new(None),
        }
    }

    /// Creates a mesh from a descriptor.
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

    // ========== Mutation methods (increment generation) ==========

    /// Set the mesh's vertex data, incrementing the generation counter.
    pub fn set_vertices(&mut self, vertices: Vec<Vertex>) {
        self.vertices = vertices;
        self.generation += 1;
        self.cached_bounding.set(None);
    }

    /// Set the mesh's primitive data, incrementing the generation counter.
    pub fn set_primitives(&mut self, primitives: Vec<MeshPrimitive>) {
        self.primitives = primitives;
        self.generation += 1;
    }

    /// Add vertices to the mesh, incrementing the generation counter.
    pub fn add_vertices(&mut self, vertices: &[Vertex]) {
        self.vertices.extend_from_slice(vertices);
        self.generation += 1;
        self.cached_bounding.set(None);
    }

    /// Add a primitive to the mesh, incrementing the generation counter.
    pub fn add_primitive(&mut self, primitive: MeshPrimitive) {
        self.primitives.push(primitive);
        self.generation += 1;
    }

    /// Translates all vertex positions by the given offset.
    pub fn translate(&mut self, offset: Vector3<f32>) {
        for v in &mut self.vertices {
            v.position[0] += offset.x;
            v.position[1] += offset.y;
            v.position[2] += offset.z;
        }
        self.generation += 1;
        self.cached_bounding.set(None);
    }

    /// Translates all vertex positions by the given offset (consuming variant).
    pub fn translated(mut self, offset: Vector3<f32>) -> Self {
        self.translate(offset);
        self
    }

    /// Transforms all vertex positions and normals by a 4x4 matrix.
    ///
    /// Positions are transformed as points (affected by translation).
    /// Normals are transformed by the inverse-transpose of the upper 3x3
    /// to handle non-uniform scaling correctly.
    pub fn transform(&mut self, matrix: &Matrix4<f32>) {
        let normal_matrix = crate::common::compute_normal_matrix(matrix);

        for v in &mut self.vertices {
            // Transform position as a point
            let pos = Point3::new(v.position[0], v.position[1], v.position[2]);
            let transformed = matrix.transform_point(pos);
            v.position = [transformed.x, transformed.y, transformed.z];

            // Transform normal by inverse-transpose
            let normal = Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
            let transformed_normal = (normal_matrix * normal).normalize();
            v.normal = [transformed_normal.x, transformed_normal.y, transformed_normal.z];
        }
        self.generation += 1;
        self.cached_bounding.set(None);
    }

    /// Transforms all vertex positions and normals by a 4x4 matrix (consuming variant).
    pub fn transformed(mut self, matrix: &Matrix4<f32>) -> Self {
        self.transform(matrix);
        self
    }

    /// Returns the current generation counter.
    ///
    /// This value increments on any mutation to the mesh data.
    /// Used by renderers to track when GPU resources need updating.
    pub fn generation(&self) -> u64 {
        self.generation
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
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}
