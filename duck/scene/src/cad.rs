use anyhow::{Context, Result};
use opencascade::primitives::{EdgeType, Shape};

use crate::common::{RgbaColor, Transform};
use crate::{
    FaceMaterial, Instance, LineMaterial, Mesh, MeshPrimitive, NodeFlags, NodeId, NodePayload,
    PrimitiveType, Scene, SubMeshRange, Topology, Vertex,
};

/// Options controlling tessellation and presentation when producing scene geometry from CAD data.
///
/// Used for both file import and programmatic authoring via [`tessellate_into`].
#[derive(Clone)]
pub struct CadTessellationOptions {
    /// Tolerance used for OCCT incremental mesh tessellation. Lower values
    /// produce finer meshes. Units match the file's unit system (typically mm).
    pub tessellation_tolerance: f64,
    /// Uniform scale applied to all vertex positions. Use `0.001` to convert
    /// from millimeters (STEP default) to metres.
    pub scale_factor: f32,
    /// Color applied to triangle faces.
    pub face_color: RgbaColor,
    /// Color applied to wireframe edges.
    pub edge_color: RgbaColor,
    /// Whether to include wireframe edges as `LineList` meshes.
    pub include_edges: bool,
}

impl Default for CadTessellationOptions {
    fn default() -> Self {
        Self {
            tessellation_tolerance: 0.01,
            scale_factor: 1.0,
            face_color: RgbaColor { r: 0.8, g: 0.8, b: 0.8, a: 1.0 },
            edge_color: RgbaColor { r: 0.15, g: 0.15, b: 0.15, a: 1.0 },
            include_edges: true,
        }
    }
}

/// Tessellates an OpenCASCADE B-Rep shape into a [`Mesh`] containing face triangles
/// and, optionally, wireframe edge line segments.
///
/// This is the shared tessellation kernel used by both the XCAF import path
/// and the interactive authoring path ([`tessellate_into`]).
pub fn tessellate_occ_shape(
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

/// Tessellates `shape` and wires it into `scene` as a mesh + material + instance + node.
///
/// Returns the [`NodeId`] of the created node.
pub fn tessellate_into(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadTessellationOptions,
    parent: Option<NodeId>,
    name: Option<&str>,
) -> Result<NodeId> {
    let mesh = tessellate_occ_shape(
        shape,
        options.tessellation_tolerance,
        options.scale_factor,
        options.include_edges,
    )?;
    let face_mat =
        scene.add_face_material(FaceMaterial::new().with_base_color_factor(options.face_color));
    let line_mat = scene.add_line_material(LineMaterial::new(options.edge_color));
    let mesh_id = scene.add_mesh(mesh);
    let instance_id = scene.add_instance(
        Instance::new(mesh_id)
            .with_face_material(face_mat)
            .with_line_material(line_mat),
    );

    let node_name = name.map(|s| s.to_string());
    let node = scene
        .add_node(parent, node_name, Transform::IDENTITY, NodeFlags::NONE)
        .context("Failed to add shape node")?;
    scene.set_node_payload(node, NodePayload::Instance(instance_id));

    Ok(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_options() -> CadTessellationOptions {
        CadTessellationOptions::default()
    }

    #[test]
    fn sphere_tessellates_to_nonempty_mesh() {
        let shape = opencascade::primitives::Shape::sphere(1.0).build();
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, Some("sphere"))
            .expect("tessellation failed");
        assert!(scene.mesh_count() > 0);
        assert!(scene.node_count() > 0);
    }

    #[test]
    fn cuboid_tessellates_to_nonempty_mesh() {
        let shape = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, Some("box"))
            .expect("tessellation failed");
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn union_of_cuboid_and_sphere_tessellates() {
        let a = opencascade::primitives::Shape::box_centered(2.0, 2.0, 2.0);
        let b = opencascade::primitives::Shape::sphere(1.5).build();
        let combined = a.union(&b).shape;
        let mut scene = Scene::new();
        tessellate_into(&combined, &mut scene, &default_options(), None, None)
            .expect("union tessellation failed");
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn cylinder_tessellates() {
        let shape = opencascade::primitives::Shape::cylinder_radius_height(0.5, 2.0);
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, None).unwrap();
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn torus_tessellates() {
        let shape = opencascade::primitives::Shape::torus().radius_1(2.0).radius_2(0.5).build();
        let mut scene = Scene::new();
        tessellate_into(&shape, &mut scene, &default_options(), None, None).unwrap();
        assert!(scene.mesh_count() > 0);
    }
}
