use anyhow::{Context, Result};
use duck_engine_scene::{Mesh, MeshPrimitive, PrimitiveType, SubMeshRange, Topology, Vertex};
use opencascade::primitives::{EdgeType, Shape};

/// Tessellates an OpenCASCADE B-Rep shape into a `Mesh` containing face triangles and,
/// optionally, wireframe edge line segments.
///
/// This is the shared tessellation kernel used by both the XCAF import path and the
/// programmatic authoring path (`CadShape::tessellate_into`).
pub(crate) fn tessellate_occ_shape(
    shape: &Shape,
    tolerance: f64,
    scale_factor: f32,
    include_edges: bool,
) -> Result<Mesh> {
    let s = scale_factor;

    // --- Faces ---
    let (occt_mesh, occt_face_ranges) = shape
        .mesh_with_tolerance_and_ranges(tolerance)
        .context("OCCT tessellation failed")?;

    let mut vertices: Vec<Vertex> = (0..occt_mesh.vertices.len())
        .map(|i| {
            let pos = occt_mesh.vertices[i];
            let norm = occt_mesh.normals.get(i).copied().unwrap_or_default();
            let uv = occt_mesh.uvs.get(i).copied().unwrap_or_default();
            Vertex {
                position: [pos.x as f32 * s, pos.y as f32 * s, pos.z as f32 * s],
                normal: [norm.x as f32, norm.y as f32, norm.z as f32],
                tex_coords: [uv.x as f32, uv.y as f32, 0.0],
            }
        })
        .collect();

    let face_indices: Vec<u32> = occt_mesh.indices.iter().map(|&i| i as u32).collect();
    let face_ranges: Vec<SubMeshRange> = occt_face_ranges
        .iter()
        .map(|r| SubMeshRange { start: r.start, count: r.count })
        .collect();

    // --- Edges ---
    // Edge vertices are appended after face vertices; absolute vertex indices are used
    // so the LineList primitive correctly references into the combined vertex buffer.
    let mut edge_indices: Vec<u32> = Vec::new();
    let mut edge_ranges: Vec<SubMeshRange> = Vec::new();

    if include_edges {
        for edge in shape.edges() {
            let points: Vec<_> = match edge.edge_type() {
                EdgeType::Line => vec![edge.start_point(), edge.end_point()],
                _ => edge.approximation_segments().collect(),
            };

            if points.len() < 2 {
                continue;
            }

            let seg_start = (edge_indices.len() / 2) as u32;
            let mut seg_count = 0u32;

            for window in points.windows(2) {
                let base = vertices.len() as u32;
                for p in window {
                    vertices.push(Vertex {
                        position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0, 0.0],
                    });
                }
                edge_indices.push(base);
                edge_indices.push(base + 1);
                seg_count += 1;
            }

            if seg_count > 0 {
                edge_ranges.push(SubMeshRange { start: seg_start, count: seg_count });
            }
        }
    }

    // --- Assemble mesh ---
    let mut primitives = Vec::new();
    if !face_indices.is_empty() {
        primitives.push(MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: face_indices,
        });
    }
    if !edge_indices.is_empty() {
        primitives.push(MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: edge_indices,
        });
    }

    let mut mesh = Mesh::from_raw(vertices, primitives);
    mesh.set_topology(Topology { face_ranges, edge_ranges, point_ranges: Vec::new() });

    Ok(mesh)
}
