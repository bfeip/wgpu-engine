use std::path::Path;

use anyhow::{Context, Result};
use opencascade::primitives::Shape;
use wgpu_engine_scene::{Mesh, Material, MeshPrimitive, NodeId, PrimitiveType, Scene, Vertex};
use wgpu_engine_scene::common::{RgbaColor, Transform};

/// Options controlling how a CAD file is imported.
pub struct CadImportOptions {
    /// Tolerance used for OCCT incremental mesh tessellation. Lower values
    /// produce finer meshes. Units match the file's unit system (typically mm).
    pub tessellation_tolerance: f64,
    /// Uniform scale applied to all vertex positions. Use `0.001` to convert
    /// from millimetres (STEP default) to metres.
    pub scale_factor: f32,
    /// Color applied to triangle faces.
    pub face_color: RgbaColor,
    /// Color applied to wireframe edges.
    pub edge_color: RgbaColor,
    /// Whether to import wireframe edges as `LineList` meshes.
    pub include_edges: bool,
}

impl Default for CadImportOptions {
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

/// Import a STEP file into `scene`, returning the root [`NodeId`].
///
/// All geometry is flattened under a single root node. When a tessellated mesh
/// exceeds the u16 vertex limit, it is split into multiple child nodes.
pub fn load_step(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<NodeId> {
    let shape = Shape::read_step(path.as_ref())
        .with_context(|| format!("Failed to read STEP file: {}", path.as_ref().display()))?;
    import_shape(shape, scene, options)
}

/// Import an IGES file into `scene`, returning the root [`NodeId`].
///
/// All geometry is flattened under a single root node. When a tessellated mesh
/// exceeds the u16 vertex limit, it is split into multiple child nodes.
pub fn load_iges(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<NodeId> {
    let shape = Shape::read_iges(path.as_ref())
        .with_context(|| format!("Failed to read IGES file: {}", path.as_ref().display()))?;
    import_shape(shape, scene, options)
}

fn import_shape(shape: Shape, scene: &mut Scene, options: &CadImportOptions) -> Result<NodeId> {
    let root = scene
        .add_default_node(None, Some("cad_import".to_string()))
        .context("Failed to add root CAD node")?;

    import_faces(&shape, scene, options, root)?;

    if options.include_edges {
        import_edges(&shape, scene, options, root)?;
    }

    Ok(root)
}

fn import_faces(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: NodeId,
) -> Result<()> {
    let occt_mesh = shape
        .mesh_with_tolerance(options.tessellation_tolerance)
        .context("OCCT tessellation failed")?;

    if occt_mesh.vertices.is_empty() {
        return Ok(());
    }

    let s = options.scale_factor;

    let vertices: Vec<Vertex> = (0..occt_mesh.vertices.len())
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

    let indices: Vec<u32> = occt_mesh.indices.iter().map(|&i| i as u32).collect();
    let primitive = MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices };

    let face_mat = scene.add_material(Material::new().with_base_color_factor(options.face_color));

    let mesh_id = scene.add_mesh(Mesh::from_raw(vertices, vec![primitive]));
    scene
        .add_instance_node(Some(parent), mesh_id, face_mat, None, Transform::IDENTITY)
        .context("Failed to add face instance node")?;

    Ok(())
}

fn import_edges(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: NodeId,
) -> Result<()> {
    use opencascade::primitives::EdgeType;

    let s = options.scale_factor;
    let mut edge_verts: Vec<Vertex> = Vec::new();
    let mut edge_indices: Vec<u32> = Vec::new();

    for edge in shape.edges() {
        let points: Vec<_> = match edge.edge_type() {
            EdgeType::Line => vec![edge.start_point(), edge.end_point()],
            _ => edge.approximation_segments().collect(),
        };

        if points.len() < 2 {
            continue;
        }

        for window in points.windows(2) {
            let base = edge_verts.len() as u32;
            for p in window {
                edge_verts.push(Vertex {
                    position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                    normal: [0.0, 0.0, 0.0],
                    tex_coords: [0.0, 0.0, 0.0],
                });
            }
            edge_indices.push(base);
            edge_indices.push(base + 1);
        }
    }

    if edge_verts.is_empty() {
        return Ok(());
    }

    let edge_mat = scene
        .add_material(Material::new().with_line_color(options.edge_color));

    let primitive = MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: edge_indices };
    let mesh_id = scene.add_mesh(Mesh::from_raw(edge_verts, vec![primitive]));
    scene
        .add_instance_node(Some(parent), mesh_id, edge_mat, None, Transform::IDENTITY)
        .context("Failed to add edge instance node")?;

    Ok(())
}
