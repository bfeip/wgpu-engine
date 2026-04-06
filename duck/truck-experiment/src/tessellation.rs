use std::collections::HashMap;

use truck_meshalgo::tessellation::{MeshableShape, MeshedShape};
use truck_modeling::{self as truck, ParametricCurve};
use duck_engine_scene::{Mesh, MeshPrimitive, PrimitiveType, Vertex};

use crate::body::Body;
use crate::edge::EdgeId;
use crate::face::FaceId;

/// Controls the quality of NURBS-to-mesh tessellation.
pub struct TessellationOptions {
    /// Chord tolerance — maximum allowed distance between the tessellated
    /// surface and the true NURBS surface. Smaller values produce finer meshes.
    pub tolerance: f64,
}

impl TessellationOptions {
    pub fn very_low_quality() -> Self {
        Self { tolerance: 0.1 }
    }

    pub fn low_quality() -> Self {
        Self { tolerance: 0.01 }
    }

    pub fn medium_quality() -> Self {
        Self { tolerance: 0.001 }
    }

    pub fn high_quality() -> Self {
        Self { tolerance: 0.0001 }
    }

    pub fn very_high_quality() -> Self {
        Self { tolerance: 0.00001 }
    }
}

impl Default for TessellationOptions {
    fn default() -> Self {
        Self::medium_quality()
    }
}

/// The result of tessellating a [`Body`] into renderable meshes.
pub struct TessellatedBody {
    /// One triangle mesh per face.
    pub face_meshes: HashMap<FaceId, Mesh>,
    /// One line-list mesh per edge (polyline approximation of the curve).
    pub edge_meshes: HashMap<EdgeId, Mesh>,
}

/// Tessellate a body into triangle meshes (faces) and line meshes (edges).
pub fn tessellate_body(body: &Body, options: &TessellationOptions) -> TessellatedBody {
    let face_meshes = tessellate_faces(body, options);
    let edge_meshes = tessellate_edges(body, options);

    TessellatedBody {
        face_meshes,
        edge_meshes,
    }
}

fn tessellate_faces(body: &Body, options: &TessellationOptions) -> HashMap<FaceId, Mesh> {
    let mut result = HashMap::new();

    for (&face_id, face) in body.faces() {
        // Create a single-face shell for tessellation
        let shell = truck::Shell::from(vec![face.inner().clone()]);
        let polygon = shell.triangulation(options.tolerance).to_polygon();

        let mesh = polygon_mesh_to_scene_mesh(&polygon);
        result.insert(face_id, mesh);
    }

    result
}

fn tessellate_edges(body: &Body, options: &TessellationOptions) -> HashMap<EdgeId, Mesh> {
    let mut result = HashMap::new();

    for (&edge_id, edge) in body.edges() {
        let truck_edge = edge.inner();
        let curve = truck_edge.oriented_curve();
        let (t0, t1) = match curve.parameter_range() {
            (std::ops::Bound::Included(a), std::ops::Bound::Included(b)) => (a, b),
            (std::ops::Bound::Included(a), std::ops::Bound::Excluded(b)) => (a, b),
            (std::ops::Bound::Excluded(a), std::ops::Bound::Included(b)) => (a, b),
            _ => (0.0, 1.0),
        };

        // Approximate the curve with line segments.
        let n_segments = ((t1 - t0) / options.tolerance).ceil().max(2.0) as usize;
        let n_segments = n_segments.min(1000);

        let mut vertices = Vec::with_capacity(n_segments + 1);
        let mut indices = Vec::with_capacity(n_segments * 2);

        for i in 0..=n_segments {
            let t = t0 + (t1 - t0) * (i as f64 / n_segments as f64);
            let pt = curve.subs(t);
            vertices.push(Vertex {
                position: [pt.x as f32, pt.y as f32, pt.z as f32],
                tex_coords: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
            });
        }

        for i in 0..n_segments {
            indices.push(i as u32);
            indices.push((i + 1) as u32);
        }

        let primitive = MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        };

        let mesh = Mesh::from_raw(vertices, vec![primitive]);
        result.insert(edge_id, mesh);
    }

    result
}

/// Convert a truck PolygonMesh into a scene Mesh.
fn polygon_mesh_to_scene_mesh(polygon: &truck_polymesh::PolygonMesh) -> Mesh {
    let positions = polygon.positions();
    let normals = polygon.normals();

    // Build a vertex for each unique (position, normal) combination.
    // Truck uses indexed attributes; we flatten to per-vertex data.
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut vertex_map: HashMap<(usize, usize), u32> = HashMap::new();

    for tri in polygon.tri_faces() {
        for sv in tri {
            let pos_idx = sv.pos;
            let nor_idx = sv.nor.unwrap_or(0);
            let key = (pos_idx, nor_idx);

            let vertex_idx = if let Some(&idx) = vertex_map.get(&key) {
                idx
            } else {
                let idx = vertices.len() as u32;
                let pos = positions[pos_idx];
                let nor = if sv.nor.is_some() && nor_idx < normals.len() {
                    normals[nor_idx]
                } else {
                    cgmath::Vector3::new(0.0, 1.0, 0.0)
                };

                vertices.push(Vertex {
                    position: [pos.x as f32, pos.y as f32, pos.z as f32],
                    tex_coords: [0.0, 0.0, 0.0],
                    normal: [nor.x as f32, nor.y as f32, nor.z as f32],
                });
                vertex_map.insert(key, idx);
                idx
            };

            indices.push(vertex_idx);
        }
    }

    let primitive = MeshPrimitive {
        primitive_type: PrimitiveType::TriangleList,
        indices,
    };

    Mesh::from_raw(vertices, vec![primitive])
}
