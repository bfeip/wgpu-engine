//! Shared mesh utilities for import/export operations.
//!
//! Provides mesh splitting for large meshes that exceed the u16 index limit (65,535 vertices).

use std::collections::HashMap;

use crate::{MeshIndex, MeshPrimitive, PrimitiveType, Vertex};

/// Split a mesh with >65535 vertices into multiple chunks that each fit in u16 indices.
///
/// Walks triangles one at a time, maintaining a vertex remap. When adding a triangle
/// would overflow the current chunk, the chunk is finalized and a new one starts.
pub fn split_mesh(vertices: &[Vertex], indices: &[u32]) -> Vec<(Vec<Vertex>, MeshPrimitive)> {
    let max_verts = MeshIndex::MAX as usize;
    let mut chunks: Vec<(Vec<Vertex>, MeshPrimitive)> = Vec::new();

    let mut chunk_verts: Vec<Vertex> = Vec::new();
    let mut chunk_indices: Vec<MeshIndex> = Vec::new();
    let mut remap: HashMap<u32, MeshIndex> = HashMap::new();

    for triangle in indices.chunks(3) {
        if triangle.len() < 3 {
            break;
        }

        // Check if adding this triangle would overflow the chunk
        let new_verts_needed = triangle
            .iter()
            .filter(|&&idx| !remap.contains_key(&idx))
            .count();

        if chunk_verts.len() + new_verts_needed > max_verts {
            // Finalize current chunk
            if !chunk_indices.is_empty() {
                chunks.push((
                    std::mem::take(&mut chunk_verts),
                    MeshPrimitive {
                        primitive_type: PrimitiveType::TriangleList,
                        indices: std::mem::take(&mut chunk_indices),
                    },
                ));
            }
            remap.clear();
        }

        // Add triangle to current chunk
        for &orig_idx in triangle {
            let local_idx = *remap.entry(orig_idx).or_insert_with(|| {
                let idx = chunk_verts.len() as MeshIndex;
                chunk_verts.push(vertices[orig_idx as usize]);
                idx
            });
            chunk_indices.push(local_idx);
        }
    }

    // Finalize last chunk
    if !chunk_indices.is_empty() {
        chunks.push((
            chunk_verts,
            MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: chunk_indices,
            },
        ));
    }

    chunks
}

/// Convert u32 indices to u16, splitting the mesh if it exceeds the u16 vertex limit.
///
/// Returns one or more `(vertices, MeshPrimitive)` pairs ready for `Mesh::from_raw()`.
/// If the mesh fits within the u16 limit, a single pair is returned without splitting.
pub fn to_u16_primitives(
    vertices: &[Vertex],
    indices: &[u32],
    primitive_type: PrimitiveType,
) -> Vec<(Vec<Vertex>, MeshPrimitive)> {
    if vertices.len() <= MeshIndex::MAX as usize {
        // Simple case: fits in u16
        let u16_indices: Vec<MeshIndex> = indices.iter().map(|&i| i as MeshIndex).collect();
        let primitive = MeshPrimitive {
            primitive_type,
            indices: u16_indices,
        };
        vec![(vertices.to_vec(), primitive)]
    } else {
        // Need to split into chunks
        split_mesh(vertices, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vertex(x: f32) -> Vertex {
        Vertex {
            position: [x, 0.0, 0.0],
            tex_coords: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn test_split_mesh_no_split_needed() {
        let vertices: Vec<Vertex> = (0..100).map(|i| make_vertex(i as f32)).collect();
        let indices: Vec<u32> = (0..99).collect();
        // 33 triangles
        let mut tri_indices = Vec::new();
        for i in (0..99).step_by(3) {
            tri_indices.push(i);
            tri_indices.push(i + 1);
            tri_indices.push(i + 2);
        }

        let chunks = split_mesh(&vertices, &tri_indices);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0.len(), 99);
    }

    #[test]
    fn test_split_mesh_splits_large_mesh() {
        // Create a mesh that needs splitting: vertices > 65535
        let n = 70_000u32;
        let vertices: Vec<Vertex> = (0..n).map(|i| make_vertex(i as f32)).collect();
        // Create triangles: each uses 3 unique vertices
        let mut indices = Vec::new();
        for i in (0..n - 2).step_by(3) {
            indices.push(i);
            indices.push(i + 1);
            indices.push(i + 2);
        }

        let chunks = split_mesh(&vertices, &indices);
        assert!(chunks.len() >= 2, "Should split into at least 2 chunks");

        // Verify each chunk respects the u16 limit
        for (verts, prim) in &chunks {
            assert!(verts.len() <= MeshIndex::MAX as usize);
            assert_eq!(prim.primitive_type, PrimitiveType::TriangleList);
            assert_eq!(prim.indices.len() % 3, 0);
        }

        // Verify total triangle count is preserved
        let total_indices: usize = chunks.iter().map(|(_, p)| p.indices.len()).sum();
        assert_eq!(total_indices, indices.len());
    }

    #[test]
    fn test_split_mesh_empty() {
        let vertices: Vec<Vertex> = Vec::new();
        let indices: Vec<u32> = Vec::new();
        let chunks = split_mesh(&vertices, &indices);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_to_u16_primitives_small_mesh() {
        let vertices: Vec<Vertex> = (0..100).map(|i| make_vertex(i as f32)).collect();
        let indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5];

        let chunks = to_u16_primitives(&vertices, &indices, PrimitiveType::TriangleList);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0.len(), 100);
        assert_eq!(chunks[0].1.indices.len(), 6);
        assert_eq!(chunks[0].1.primitive_type, PrimitiveType::TriangleList);
    }

    #[test]
    fn test_to_u16_primitives_large_mesh() {
        let n = 70_000u32;
        let vertices: Vec<Vertex> = (0..n).map(|i| make_vertex(i as f32)).collect();
        let mut indices = Vec::new();
        for i in (0..n - 2).step_by(3) {
            indices.push(i);
            indices.push(i + 1);
            indices.push(i + 2);
        }

        let chunks = to_u16_primitives(&vertices, &indices, PrimitiveType::TriangleList);
        assert!(chunks.len() >= 2);
        for (verts, _) in &chunks {
            assert!(verts.len() <= MeshIndex::MAX as usize);
        }
    }
}
