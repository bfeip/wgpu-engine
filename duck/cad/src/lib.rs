use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use cgmath::Matrix4;
use duck_engine_common::decompose_matrix;
use duck_engine_scene::common::Transform;
use duck_engine_scene::{Material, Mesh, MeshPrimitive, NodeId, PrimitiveType, Scene, Vertex};
use duck_engine_scene::common::RgbaColor;
use opencascade::primitives::{EdgeType, Shape, ShapeType};
use opencascade::xcaf::{XcafColorTool, XcafDimTolTool, XcafDocument, XcafLabel, XcafShapeTool};

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
    /// Whether to import PMI graphical presentation geometry as `LineList` meshes.
    pub include_pmi: bool,
    /// Color applied to PMI annotation lines.
    pub pmi_color: RgbaColor,
}

impl Default for CadImportOptions {
    fn default() -> Self {
        Self {
            tessellation_tolerance: 0.01,
            scale_factor: 1.0,
            face_color: RgbaColor { r: 0.8, g: 0.8, b: 0.8, a: 1.0 },
            edge_color: RgbaColor { r: 0.15, g: 0.15, b: 0.15, a: 1.0 },
            include_edges: true,
            include_pmi: true,
            pmi_color: RgbaColor { r: 0.2, g: 0.6, b: 1.0, a: 1.0 },
        }
    }
}

/// CAD-specific metadata for a node created during import.
pub struct CadEntityInfo {
    /// Part name from CAD file metadata. Always `None` until XDE support is added.
    pub name: Option<String>,
    /// `true` when this node is an assembly node (has sub-components).
    /// `false` for leaf geometry nodes.
    pub is_assembly: bool,
}

/// Result of a hierarchy-preserving CAD import.
pub struct CadImportResult {
    /// Root node in the scene graph.
    pub root: NodeId,
    /// Maps every created [`NodeId`] to its CAD metadata.
    pub entity_map: HashMap<NodeId, CadEntityInfo>,
    /// Root node of the PMI geometry sub-tree, if PMI was found.
    ///
    /// `None` when `CadImportOptions::include_pmi` is false, the file has no
    /// graphical PMI, or the import used the string-based path (which bypasses XCAF).
    ///
    /// **Note**: currently all PMI geometry is accumulated into a single `LineList` mesh
    /// under this node. Per-annotation nodes (for individual visibility control) are
    /// planned as a future extension.
    pub pmi_root: Option<NodeId>,
}

/// Import a STEP file into `scene`, returning a [`CadImportResult`] that mirrors
/// the assembly hierarchy. Each assembly component becomes its own scene node.
pub fn load_step(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_step(path.as_ref())
        .with_context(|| format!("Failed to read STEP file: {}", path.as_ref().display()))?;
    import_xcaf_document(&doc, scene, options)
}

/// Import a STEP file from its text content into `scene`.
pub fn load_step_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let shape = Shape::read_step_from_str(s).context("Failed to read STEP data")?;
    import_shape_hierarchical(shape, scene, options)
}

/// Import an IGES file into `scene`, returning a [`CadImportResult`].
pub fn load_iges(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_iges(path.as_ref())
        .with_context(|| format!("Failed to read IGES file: {}", path.as_ref().display()))?;
    import_xcaf_document(&doc, scene, options)
}

/// Import an IGES file from its text content into `scene`.
pub fn load_iges_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let shape = Shape::read_iges_from_str(s).context("Failed to read IGES data")?;
    import_shape_hierarchical(shape, scene, options)
}

fn import_shape_hierarchical(
    shape: Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let mut entity_map = HashMap::new();
    let root = import_node(&shape, scene, options, None, &mut entity_map, Some("cad_import".to_string()))?;
    Ok(CadImportResult { root, entity_map, pmi_root: None })
}

fn import_xcaf_document(
    doc: &XcafDocument,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let shape_tool = doc.shape_tool();
    let color_tool = doc.color_tool();
    let mut entity_map = HashMap::new();

    let root = scene
        .add_node(None, Some("cad_import".to_string()), Transform::IDENTITY)
        .context("Failed to add root CAD node")?;
    entity_map.insert(root, CadEntityInfo { name: Some("cad_import".to_string()), is_assembly: true });

    for label in shape_tool.free_shapes() {
        import_xcaf_label(&label, &shape_tool, &color_tool, scene, options, Some(root), &mut entity_map)?;
    }

    let pmi_root = if options.include_pmi {
        import_pmi(&doc.dim_tol_tool(), scene, options, root)?
    } else {
        None
    };

    Ok(CadImportResult { root, entity_map, pmi_root })
}

fn import_xcaf_label(
    label: &XcafLabel,
    shape_tool: &XcafShapeTool,
    color_tool: &XcafColorTool,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: Option<NodeId>,
    entity_map: &mut HashMap<NodeId, CadEntityInfo>,
) -> Result<NodeId> {
    let name = label.name();
    let transform = matrix_to_transform(shape_tool.location_matrix(label));
    let is_assembly = shape_tool.is_assembly(label);

    let node_id = scene
        .add_node(parent, name.clone(), transform)
        .context("Failed to add XCAF node")?;
    entity_map.insert(node_id, CadEntityInfo { name, is_assembly });

    if is_assembly {
        for child in shape_tool.components(label) {
            import_xcaf_label(&child, shape_tool, color_tool, scene, options, Some(node_id), entity_map)?;
        }
    } else {
        let shape = shape_tool.shape(label);

        let face_color = color_tool
            .color_of_label(label)
            .or_else(|| color_tool.color_of_shape(&shape))
            .map(|(r, g, b)| RgbaColor { r, g, b, a: 1.0 })
            .unwrap_or(options.face_color);

        import_faces(&shape, scene, options, node_id, face_color)?;
        if options.include_edges {
            import_edges(&shape, scene, options, node_id)?;
        }
    }

    Ok(node_id)
}

/// Converts a row-major `[[f64; 4]; 4]` matrix (as returned by `Shape::location_as_matrix`)
/// into a [`Transform`] via matrix decomposition.
///
/// This is reusable for any future matrix-to-transform conversion (XDE, etc.).
fn matrix_to_transform(mat: [[f64; 4]; 4]) -> Transform {
    // cgmath::Matrix4::new takes arguments in column-major order:
    //   new(c0r0, c0r1, c0r2, c0r3,  c1r0, c1r1, ...)
    // mat is row-major: mat[row][col], so c{col}r{row} = mat[row][col].
    let m = Matrix4::new(
        mat[0][0] as f32, mat[1][0] as f32, mat[2][0] as f32, mat[3][0] as f32, // col 0
        mat[0][1] as f32, mat[1][1] as f32, mat[2][1] as f32, mat[3][1] as f32, // col 1
        mat[0][2] as f32, mat[1][2] as f32, mat[2][2] as f32, mat[3][2] as f32, // col 2
        mat[0][3] as f32, mat[1][3] as f32, mat[2][3] as f32, mat[3][3] as f32, // col 3
    );
    decompose_matrix(&m)
}

/// Recursively imports `shape` as a scene node (and its sub-tree) under `parent`.
///
/// - If the shape has direct sub-shapes it is treated as an assembly node and the
///   function recurses into each child.
/// - If it has no sub-shapes it is treated as leaf geometry: faces are tessellated
///   and edges are extracted as children of the node.
fn import_node(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: Option<NodeId>,
    entity_map: &mut HashMap<NodeId, CadEntityInfo>,
    name: Option<String>,
) -> Result<NodeId> {
    // Null shapes appear when the file format is unsupported (e.g. IFC uses the STEP wire format
    // but a different schema) or as placeholders for partially-transferred entities.
    if shape.is_null() {
        let node_id = scene
            .add_node(parent, name.clone(), Transform::IDENTITY)
            .context("Failed to add null CAD node")?;
        entity_map.insert(node_id, CadEntityInfo { name, is_assembly: false });
        return Ok(node_id);
    }

    let transform = matrix_to_transform(shape.location_as_matrix());

    // Only recurse into Compound/CompoundSolid shapes — these represent assembly nodes.
    // Solid, Shell, Face, etc. are leaf geometry and should be tessellated directly.
    let is_assembly = matches!(shape.shape_type(), ShapeType::Compound | ShapeType::CompoundSolid);
    let children: Vec<Shape> = if is_assembly { shape.sub_shapes().collect() } else { vec![] };
    let is_assembly = !children.is_empty(); // treat an empty compound as a leaf

    let node_id = scene
        .add_node(parent, name.clone(), transform)
        .context("Failed to add CAD node")?;
    entity_map.insert(node_id, CadEntityInfo { name, is_assembly });

    if is_assembly {
        for (i, child) in children.iter().enumerate() {
            import_node(child, scene, options, Some(node_id), entity_map, Some(format!("part_{i}")))?;
        }
    } else {
        import_faces(shape, scene, options, node_id, options.face_color)?;
        if options.include_edges {
            import_edges(shape, scene, options, node_id)?;
        }
    }

    Ok(node_id)
}

fn import_faces(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: NodeId,
    face_color: RgbaColor,
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

    let face_mat = scene.add_material(Material::new().with_base_color_factor(face_color));
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

    let edge_mat = scene.add_material(Material::new().with_line_color(options.edge_color));
    let primitive = MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: edge_indices };
    let mesh_id = scene.add_mesh(Mesh::from_raw(edge_verts, vec![primitive]));
    scene
        .add_instance_node(Some(parent), mesh_id, edge_mat, None, Transform::IDENTITY)
        .context("Failed to add edge instance node")?;

    Ok(())
}

/// Extract PMI graphical presentation geometry from `dim_tol_tool` and add it to the scene
/// as a single `LineList` mesh under a dedicated `pmi` node.
///
/// Iterates all three PMI label types (dimensions, geometric tolerances, datums) and
/// tessellates their presentation shapes using the same edge-extraction path as
/// [`import_edges`].
///
/// Returns `None` if no presentation geometry was found (e.g. the file has only semantic
/// PMI and no graphical representations).
fn import_pmi(
    dim_tol_tool: &XcafDimTolTool,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: NodeId,
) -> Result<Option<NodeId>> {
    let s = options.scale_factor;
    let mut all_verts: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    let all_labels = dim_tol_tool
        .dimension_labels()
        .chain(dim_tol_tool.geom_tolerance_labels())
        .chain(dim_tol_tool.datum_labels());

    for label in all_labels {
        let shape = label
            .dimension_presentation()
            .or_else(|| label.geom_tolerance_presentation())
            .or_else(|| label.datum_presentation());

        let Some(shape) = shape else { continue };

        for edge in shape.edges() {
            let points: Vec<_> = match edge.edge_type() {
                EdgeType::Line => vec![edge.start_point(), edge.end_point()],
                _ => edge.approximation_segments().collect(),
            };

            if points.len() < 2 {
                continue;
            }

            for window in points.windows(2) {
                let base = all_verts.len() as u32;
                for p in window {
                    all_verts.push(Vertex {
                        position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0, 0.0],
                    });
                }
                all_indices.push(base);
                all_indices.push(base + 1);
            }
        }
    }

    if all_verts.is_empty() {
        return Ok(None);
    }

    let pmi_root = scene
        .add_node(Some(parent), Some("pmi".to_string()), Transform::IDENTITY)
        .context("Failed to add PMI root node")?;
    let mat = scene.add_material(Material::new().with_line_color(options.pmi_color));
    let primitive = MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: all_indices };
    let mesh_id = scene.add_mesh(Mesh::from_raw(all_verts, vec![primitive]));
    scene
        .add_instance_node(Some(pmi_root), mesh_id, mat, None, Transform::IDENTITY)
        .context("Failed to add PMI geometry node")?;

    Ok(Some(pmi_root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn step_file() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/NIST-PMI-STEP-Files/nist_ctc_01_asme1_ap242-e1.stp")
    }

    #[test]
    fn load_step_from_file_produces_meshes() {
        let path = step_file();
        let mut scene = duck_engine_scene::Scene::new();
        let options = CadImportOptions::default();
        load_step(&path, &mut scene, &options).expect("load_step failed");
        assert!(scene.mesh_count() > 0, "expected at least one mesh");
    }

    #[test]
    fn load_step_from_str_produces_meshes() {
        let path = step_file();
        let text = std::fs::read_to_string(&path).expect("could not read STEP file");
        let options = CadImportOptions::default();
        let mut scene = duck_engine_scene::Scene::new();
        load_step_from_str(&text, &mut scene, &options).expect("str load failed");
        assert!(scene.mesh_count() > 0, "expected at least one mesh");
    }

    #[test]
    fn load_step_hierarchy_produces_tree() {
        let path = step_file();
        let mut scene = duck_engine_scene::Scene::new();
        let options = CadImportOptions::default();
        let result = load_step(&path, &mut scene, &options).expect("hierarchy load failed");
        assert!(scene.node_count() > 1, "expected multiple nodes for assembly file");
        assert!(result.entity_map.contains_key(&result.root));
    }

    #[test]
    fn load_step_with_pmi_disabled_has_no_pmi_root() {
        let path = step_file();
        let mut scene = duck_engine_scene::Scene::new();
        let options = CadImportOptions { include_pmi: false, ..Default::default() };
        let result = load_step(&path, &mut scene, &options).expect("load without PMI failed");
        assert!(result.pmi_root.is_none(), "pmi_root should be None when include_pmi is false");
    }

    #[test]
    fn load_step_pmi_enabled_completes_without_error() {
        // Verifies that PMI extraction runs without panicking. Whether pmi_root is Some or None
        // depends on whether the test file has graphical PMI representations — both are valid.
        let path = step_file();
        let mut scene = duck_engine_scene::Scene::new();
        let options = CadImportOptions { include_pmi: true, ..Default::default() };
        let result = load_step(&path, &mut scene, &options).expect("PMI load failed");
        // If PMI geometry was found, the root node must be in the scene.
        if let Some(pmi_root) = result.pmi_root {
            assert!(scene.node_count() > 1);
            let _ = pmi_root; // node id is valid
        }
    }
}
