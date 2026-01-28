use std::{cell::Cell, fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use cgmath::Point3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{
    common::{Aabb, ConvexPolyhedron, Ray},
    renderer::VertexShaderLocations,
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

    // ========== Primitive geometry constructors ==========

    /// Creates a box (cuboid) mesh centered at the origin.
    ///
    /// # Arguments
    /// * `width` - Size along the X axis
    /// * `height` - Size along the Y axis
    /// * `depth` - Size along the Z axis
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let cube = Mesh::box_mesh(1.0, 1.0, 1.0);
    /// let rectangular = Mesh::box_mesh(2.0, 1.0, 0.5);
    /// ```
    pub fn box_mesh(width: f32, height: f32, depth: f32) -> Self {
        struct Face {
            normal: [f32; 3],
            corners: [[f32; 3]; 4],
        }

        let hw = width / 2.0;
        let hh = height / 2.0;
        let hd = depth / 2.0;

        let faces = [
            Face {
                normal: [0.0, 0.0, 1.0],
                corners: [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]],
            },
            Face {
                normal: [0.0, 0.0, -1.0],
                corners: [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]],
            },
            Face {
                normal: [0.0, 1.0, 0.0],
                corners: [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]],
            },
            Face {
                normal: [0.0, -1.0, 0.0],
                corners: [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]],
            },
            Face {
                normal: [1.0, 0.0, 0.0],
                corners: [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]],
            },
            Face {
                normal: [-1.0, 0.0, 0.0],
                corners: [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]],
            },
        ];

        // UV coordinates for each face corner
        let uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for face in &faces {
            let base_index = vertices.len() as MeshIndex;

            for (i, pos) in face.corners.iter().enumerate() {
                vertices.push(Vertex {
                    position: *pos,
                    tex_coords: [uvs[i][0], uvs[i][1], 0.0],
                    normal: face.normal,
                });
            }

            // Two triangles per face
            indices.extend_from_slice(&[
                base_index,
                base_index + 1,
                base_index + 2,
                base_index,
                base_index + 2,
                base_index + 3,
            ]);
        }

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
    }

    /// Creates a cube mesh centered at the origin.
    ///
    /// Convenience method equivalent to `Mesh::box_mesh(size, size, size)`.
    ///
    /// # Arguments
    /// * `size` - The length of each edge
    pub fn cube(size: f32) -> Self {
        Self::box_mesh(size, size, size)
    }

    /// Creates a UV sphere mesh centered at the origin.
    ///
    /// # Arguments
    /// * `radius` - Radius of the sphere
    /// * `segments` - Number of longitudinal segments (minimum 3)
    /// * `rings` - Number of latitudinal rings (minimum 2)
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let sphere = Mesh::sphere(1.0, 32, 16);
    /// ```
    pub fn sphere(radius: f32, segments: u32, rings: u32) -> Self {
        use std::f32::consts::PI;

        let segments = segments.max(3);
        let rings = rings.max(2);

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Generate vertices
        for ring in 0..=rings {
            let phi = PI * ring as f32 / rings as f32; // 0 to PI (top to bottom)
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            let v = ring as f32 / rings as f32;

            for seg in 0..=segments {
                let theta = 2.0 * PI * seg as f32 / segments as f32; // 0 to 2PI
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();
                let u = seg as f32 / segments as f32;

                let x = sin_phi * cos_theta;
                let y = cos_phi;
                let z = sin_phi * sin_theta;

                vertices.push(Vertex {
                    position: [x * radius, y * radius, z * radius],
                    tex_coords: [u, v, 0.0],
                    normal: [x, y, z],
                });
            }
        }

        // Generate indices
        let verts_per_ring = segments + 1;
        for ring in 0..rings {
            for seg in 0..segments {
                let current = ring * verts_per_ring + seg;
                let next = current + verts_per_ring;

                // Skip degenerate triangles at poles
                if ring != 0 {
                    indices.push(current as MeshIndex);
                    indices.push(next as MeshIndex);
                    indices.push((current + 1) as MeshIndex);
                }
                if ring != rings - 1 {
                    indices.push((current + 1) as MeshIndex);
                    indices.push(next as MeshIndex);
                    indices.push((next + 1) as MeshIndex);
                }
            }
        }

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
    }

    /// Creates a cylinder mesh centered at the origin, extending along the Y axis.
    ///
    /// # Arguments
    /// * `radius` - Radius of the cylinder
    /// * `height` - Height of the cylinder
    /// * `segments` - Number of segments around the circumference (minimum 3)
    /// * `capped` - Whether to include top and bottom cap faces
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let cylinder = Mesh::cylinder(0.5, 2.0, 32, true);
    /// ```
    pub fn cylinder(radius: f32, height: f32, segments: u32, capped: bool) -> Self {
        use std::f32::consts::PI;

        let segments = segments.max(3);
        let half_height = height / 2.0;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Side vertices (two rings)
        for i in 0..=segments {
            let theta = 2.0 * PI * i as f32 / segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let u = i as f32 / segments as f32;

            let x = radius * cos_theta;
            let z = radius * sin_theta;
            let normal = [cos_theta, 0.0, sin_theta];

            // Bottom vertex
            vertices.push(Vertex {
                position: [x, -half_height, z],
                tex_coords: [u, 1.0, 0.0],
                normal,
            });
            // Top vertex
            vertices.push(Vertex {
                position: [x, half_height, z],
                tex_coords: [u, 0.0, 0.0],
                normal,
            });
        }

        // Side indices
        for i in 0..segments {
            let base = i * 2;
            indices.extend_from_slice(&[
                base as MeshIndex,
                (base + 1) as MeshIndex,
                (base + 3) as MeshIndex,
                base as MeshIndex,
                (base + 3) as MeshIndex,
                (base + 2) as MeshIndex,
            ]);
        }

        // Caps
        if capped {
            // Top cap center
            let top_center_idx = vertices.len() as MeshIndex;
            vertices.push(Vertex {
                position: [0.0, half_height, 0.0],
                tex_coords: [0.5, 0.5, 0.0],
                normal: [0.0, 1.0, 0.0],
            });

            // Top cap ring
            for i in 0..=segments {
                let theta = 2.0 * PI * i as f32 / segments as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                vertices.push(Vertex {
                    position: [radius * cos_theta, half_height, radius * sin_theta],
                    tex_coords: [(cos_theta + 1.0) / 2.0, (sin_theta + 1.0) / 2.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                });
            }

            // Top cap indices
            let top_ring_start = top_center_idx + 1;
            for i in 0..segments as MeshIndex {
                indices.extend_from_slice(&[
                    top_center_idx,
                    top_ring_start + i,
                    top_ring_start + i + 1,
                ]);
            }

            // Bottom cap center
            let bottom_center_idx = vertices.len() as MeshIndex;
            vertices.push(Vertex {
                position: [0.0, -half_height, 0.0],
                tex_coords: [0.5, 0.5, 0.0],
                normal: [0.0, -1.0, 0.0],
            });

            // Bottom cap ring
            for i in 0..=segments {
                let theta = 2.0 * PI * i as f32 / segments as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                vertices.push(Vertex {
                    position: [radius * cos_theta, -half_height, radius * sin_theta],
                    tex_coords: [(cos_theta + 1.0) / 2.0, (1.0 - sin_theta) / 2.0, 0.0],
                    normal: [0.0, -1.0, 0.0],
                });
            }

            // Bottom cap indices (winding reversed)
            let bottom_ring_start = bottom_center_idx + 1;
            for i in 0..segments as MeshIndex {
                indices.extend_from_slice(&[
                    bottom_center_idx,
                    bottom_ring_start + i + 1,
                    bottom_ring_start + i,
                ]);
            }
        }

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
    }

    /// Creates a cone mesh centered at the origin, with the apex pointing up (+Y).
    ///
    /// # Arguments
    /// * `radius` - Radius of the base
    /// * `height` - Height of the cone
    /// * `segments` - Number of segments around the circumference (minimum 3)
    /// * `capped` - Whether to include the bottom cap face
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let cone = Mesh::cone(0.5, 1.0, 32, true);
    /// ```
    pub fn cone(radius: f32, height: f32, segments: u32, capped: bool) -> Self {
        use std::f32::consts::PI;

        let segments = segments.max(3);
        let half_height = height / 2.0;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Calculate the normal slope for the cone sides
        let slope = radius / height;
        let normal_y = slope / (1.0 + slope * slope).sqrt();
        let normal_xz = 1.0 / (1.0 + slope * slope).sqrt();

        // Apex vertex (duplicated for each segment for proper normals)
        let apex_y = half_height;

        // Side faces
        for i in 0..=segments {
            let theta = 2.0 * PI * i as f32 / segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let nx = normal_xz * cos_theta;
            let nz = normal_xz * sin_theta;

            // Base vertex
            vertices.push(Vertex {
                position: [radius * cos_theta, -half_height, radius * sin_theta],
                tex_coords: [i as f32 / segments as f32, 1.0, 0.0],
                normal: [nx, normal_y, nz],
            });

            // Apex vertex (with matching normal for this segment)
            vertices.push(Vertex {
                position: [0.0, apex_y, 0.0],
                tex_coords: [i as f32 / segments as f32, 0.0, 0.0],
                normal: [nx, normal_y, nz],
            });
        }

        // Side indices
        for i in 0..segments {
            let base = i * 2;
            indices.extend_from_slice(&[
                base as MeshIndex,
                (base + 1) as MeshIndex,
                (base + 2) as MeshIndex,
            ]);
        }

        // Bottom cap
        if capped {
            let cap_center_idx = vertices.len() as MeshIndex;
            vertices.push(Vertex {
                position: [0.0, -half_height, 0.0],
                tex_coords: [0.5, 0.5, 0.0],
                normal: [0.0, -1.0, 0.0],
            });

            // Cap ring
            for i in 0..=segments {
                let theta = 2.0 * PI * i as f32 / segments as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                vertices.push(Vertex {
                    position: [radius * cos_theta, -half_height, radius * sin_theta],
                    tex_coords: [(cos_theta + 1.0) / 2.0, (1.0 - sin_theta) / 2.0, 0.0],
                    normal: [0.0, -1.0, 0.0],
                });
            }

            // Cap indices (winding reversed)
            let cap_ring_start = cap_center_idx + 1;
            for i in 0..segments as MeshIndex {
                indices.extend_from_slice(&[
                    cap_center_idx,
                    cap_ring_start + i + 1,
                    cap_ring_start + i,
                ]);
            }
        }

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
    }

    /// Creates a torus mesh centered at the origin, lying in the XZ plane.
    ///
    /// # Arguments
    /// * `major_radius` - Distance from the center of the torus to the center of the tube
    /// * `minor_radius` - Radius of the tube
    /// * `major_segments` - Number of segments around the main ring (minimum 3)
    /// * `minor_segments` - Number of segments around the tube cross-section (minimum 3)
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let torus = Mesh::torus(1.0, 0.3, 32, 16);
    /// ```
    pub fn torus(major_radius: f32, minor_radius: f32, major_segments: u32, minor_segments: u32) -> Self {
        use std::f32::consts::PI;

        let major_segments = major_segments.max(3);
        let minor_segments = minor_segments.max(3);

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for i in 0..=major_segments {
            let theta = 2.0 * PI * i as f32 / major_segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let u = i as f32 / major_segments as f32;

            for j in 0..=minor_segments {
                let phi = 2.0 * PI * j as f32 / minor_segments as f32;
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();
                let v = j as f32 / minor_segments as f32;

                // Position on the tube surface
                let x = (major_radius + minor_radius * cos_phi) * cos_theta;
                let y = minor_radius * sin_phi;
                let z = (major_radius + minor_radius * cos_phi) * sin_theta;

                // Normal vector (points from tube center to surface)
                let nx = cos_phi * cos_theta;
                let ny = sin_phi;
                let nz = cos_phi * sin_theta;

                vertices.push(Vertex {
                    position: [x, y, z],
                    tex_coords: [u, v, 0.0],
                    normal: [nx, ny, nz],
                });
            }
        }

        // Generate indices
        let verts_per_ring = minor_segments + 1;
        for i in 0..major_segments {
            for j in 0..minor_segments {
                let current = i * verts_per_ring + j;
                let next = (i + 1) * verts_per_ring + j;

                indices.extend_from_slice(&[
                    current as MeshIndex,
                    next as MeshIndex,
                    (current + 1) as MeshIndex,
                    (current + 1) as MeshIndex,
                    next as MeshIndex,
                    (next + 1) as MeshIndex,
                ]);
            }
        }

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
    }

    /// Creates a flat plane mesh in the XZ plane, centered at the origin.
    ///
    /// # Arguments
    /// * `width` - Size along the X axis
    /// * `depth` - Size along the Z axis
    /// * `width_segments` - Number of segments along the width (minimum 1)
    /// * `depth_segments` - Number of segments along the depth (minimum 1)
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let plane = Mesh::plane(10.0, 10.0, 1, 1);
    /// let detailed_plane = Mesh::plane(10.0, 10.0, 10, 10);
    /// ```
    pub fn plane(width: f32, depth: f32, width_segments: u32, depth_segments: u32) -> Self {
        let width_segments = width_segments.max(1);
        let depth_segments = depth_segments.max(1);

        let hw = width / 2.0;
        let hd = depth / 2.0;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for zi in 0..=depth_segments {
            let v = zi as f32 / depth_segments as f32;
            let z = -hd + v * depth;

            for xi in 0..=width_segments {
                let u = xi as f32 / width_segments as f32;
                let x = -hw + u * width;

                vertices.push(Vertex {
                    position: [x, 0.0, z],
                    tex_coords: [u, v, 0.0],
                    normal: [0.0, 1.0, 0.0],
                });
            }
        }

        let verts_per_row = width_segments + 1;
        for zi in 0..depth_segments {
            for xi in 0..width_segments {
                let current = zi * verts_per_row + xi;
                let next = current + verts_per_row;

                indices.extend_from_slice(&[
                    current as MeshIndex,
                    next as MeshIndex,
                    (current + 1) as MeshIndex,
                    (current + 1) as MeshIndex,
                    next as MeshIndex,
                    (next + 1) as MeshIndex,
                ]);
            }
        }

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
    }

    /// Creates a simple quad (two triangles) in the XY plane, facing +Z.
    ///
    /// # Arguments
    /// * `width` - Size along the X axis
    /// * `height` - Size along the Y axis
    ///
    /// # Example
    /// ```
    /// use wgpu_engine::scene::Mesh;
    /// let quad = Mesh::quad(2.0, 1.0);
    /// ```
    pub fn quad(width: f32, height: f32) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;

        let vertices = vec![
            Vertex {
                position: [-hw, -hh, 0.0],
                tex_coords: [0.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [hw, -hh, 0.0],
                tex_coords: [1.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [hw, hh, 0.0],
                tex_coords: [1.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-hw, hh, 0.0],
                tex_coords: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        ];

        let indices = vec![0, 1, 2, 0, 2, 3];

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices,
            }],
        )
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

    // ========== GPU resource management ==========

    /// Check if GPU resources need to be created or updated.
    pub(crate) fn needs_gpu_upload(&self) -> bool {
        self.gpu.is_none() || self.dirty
    }

    /// Create or update GPU resources for this mesh.
    ///
    /// This method is called automatically by `Renderer::prepare_scene()` before rendering.
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
