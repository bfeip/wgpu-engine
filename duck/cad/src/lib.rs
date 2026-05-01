use std::path::Path;

use anyhow::{Context, Result};
use cgmath::Matrix4;
use duck_engine_common::decompose_matrix;
use duck_engine_scene::common::Transform;
use duck_engine_scene::{
    Camera, InstanceId, Material, Mesh, MeshPrimitive, NodeId, NodePayload, PrimitiveType, Scene,
    SubMeshRange, Topology, Vertex,
};
use duck_engine_scene::common::RgbaColor;
use opencascade::primitives::{EdgeType, Shape};
use opencascade::xcaf::{
    XcafColorTool, XcafDimTolTool, XcafDocument, XcafLabel, XcafShapeTool, ViewProjection,
};

/// Options controlling how a CAD file is imported.
pub struct CadImportOptions {
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
            pmi_color: RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
        }
    }
}

/// Result of a hierarchy-preserving CAD import.
pub struct CadImportResult {
    /// Root node in the scene graph.
    pub root: NodeId,
    /// Root node of the PMI geometry sub-tree, if PMI was found.
    ///
    /// `None` when `CadImportOptions::include_pmi` is false, the file has no
    /// graphical PMI, or the import used the string-based path (which bypasses XCAF).
    ///
    /// Each annotation (dimension, geometric tolerance, datum) is a named child node
    /// under this root, with its own mesh and instance.
    pub pmi_root: Option<NodeId>,
    /// Camera nodes for named views imported from the CAD file.
    ///
    /// Empty when the file contains no view definitions.
    /// TODO(cad-views): implement once camera/view node representation is finalized.
    pub views: Vec<NodeId>,
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

/// Import STEP data from a string into `scene`.
pub fn load_step_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_step_from_str(s).context("Failed to read STEP data")?;
    import_xcaf_document(&doc, scene, options)
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

/// Import IGES data from a string into `scene`.
pub fn load_iges_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = XcafDocument::read_iges_from_str(s).context("Failed to read IGES data")?;
    import_xcaf_document(&doc, scene, options)
}

fn import_xcaf_document(
    doc: &XcafDocument,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let shape_tool = doc.shape_tool();
    let color_tool = doc.color_tool();

    let root = scene
        .add_node(None, Some("cad_import".to_string()), Transform::IDENTITY)
        .context("Failed to add root CAD node")?;

    for label in shape_tool.free_shapes() {
        import_xcaf_label(&label, &shape_tool, &color_tool, scene, options, Some(root))?;
    }

    let pmi_root = if options.include_pmi {
        import_pmi(&doc.dim_tol_tool(), scene, options, root)?
    } else {
        None
    };

    let views = import_views(&doc.view_tool(), scene, options);

    Ok(CadImportResult { root, pmi_root, views })
}

fn import_xcaf_label(
    label: &XcafLabel,
    shape_tool: &XcafShapeTool,
    color_tool: &XcafColorTool,
    scene: &mut Scene,
    options: &CadImportOptions,
    parent: Option<NodeId>,
) -> Result<NodeId> {
    let name = label.name();
    let transform = matrix_to_transform(shape_tool.location_matrix(label));
    let is_assembly = shape_tool.is_assembly(label);

    let node_id = scene
        .add_node(parent, name.clone(), transform)
        .context("Failed to add XCAF node")?;
    if is_assembly {
        for child in shape_tool.components(label) {
            import_xcaf_label(&child, shape_tool, color_tool, scene, options, Some(node_id))?;
        }
    } else {
        let shape = shape_tool.shape(label);

        let face_color = color_tool
            .color_of_label(label)
            .or_else(|| color_tool.color_of_shape(&shape))
            .map(|(r, g, b)| RgbaColor { r, g, b, a: 1.0 })
            .unwrap_or(options.face_color);

        import_leaf_part(&shape, scene, options, node_id, face_color)?;
    };

    Ok(node_id)
}

/// Converts a row-major `[[f64; 4]; 4]` matrix into a [`Transform`] via matrix decomposition.
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

/// Tessellates a leaf B-Rep shape into a single mesh containing both face triangles and
/// edge lines, sets sub-geometry topology on the mesh, and attaches it directly to `node`.
///
/// Returns the [`InstanceId`] of the created instance so callers can record it in the
/// entity map.
fn import_leaf_part(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    node: NodeId,
    face_color: RgbaColor,
) -> Result<InstanceId> {
    let s = options.scale_factor;

    // --- Faces ---
    let (occt_mesh, occt_face_ranges) = shape
        .mesh_with_tolerance_and_ranges(options.tessellation_tolerance)
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
    // TODO: de-duplicate edge vertices that coincide with existing face vertices.
    // This requires a spatial lookup (e.g. a HashMap keyed on quantized position)
    // since OCCT's face and edge tessellations are independent.
    let mut edge_indices: Vec<u32> = Vec::new();
    let mut edge_ranges: Vec<SubMeshRange> = Vec::new();

    if options.include_edges {
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
        primitives.push(MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: face_indices });
    }
    if !edge_indices.is_empty() {
        primitives.push(MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: edge_indices });
    }

    let mut mesh = Mesh::from_raw(vertices, primitives);
    mesh.set_topology(Topology { face_ranges, edge_ranges, point_ranges: Vec::new() });

    let mat = scene.add_material(
        Material::new()
            .with_base_color_factor(face_color)
            .with_line_color(options.edge_color),
    );
    let mesh_id = scene.add_mesh(mesh);
    let instance_id = scene.add_instance(mesh_id, mat);
    scene.set_node_payload(node, NodePayload::Instance(instance_id));

    Ok(instance_id)
}

/// Extract PMI graphical presentation geometry from `dim_tol_tool` and add it to the scene.
///
/// Each annotation label (dimension, geometric tolerance, datum) becomes its own named
/// child node under a shared `pmi` root, with its own mesh and instance. The node name
/// is taken from `label.name()` when available; otherwise a numbered fallback such as
/// `"dimension 1"` or `"datum 3"` is used.
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

    let pmi_root = scene
        .add_node(Some(parent), Some("pmi".to_string()), Transform::IDENTITY)
        .context("Failed to add PMI root node")?;
    let mat = scene.add_material(Material::new().with_line_color(options.pmi_color));

    let all_labels = dim_tol_tool
        .dimension_labels()
        .chain(dim_tol_tool.geom_tolerance_labels())
        .chain(dim_tol_tool.datum_labels());

    let mut any_annotation = false;
    let (mut dim_count, mut tol_count, mut datum_count) = (0u32, 0u32, 0u32);

    for label in all_labels {
        let shape = label
            .dimension_presentation()
            .or_else(|| label.geom_tolerance_presentation())
            .or_else(|| label.datum_presentation());
        let Some(shape) = shape else { continue };

        let fallback_name = if label.is_dimension() {
            dim_count += 1;
            format!("dimension {dim_count}")
        } else if label.is_geom_tolerance() {
            tol_count += 1;
            format!("geom_tolerance {tol_count}")
        } else {
            datum_count += 1;
            format!("datum {datum_count}")
        };

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for edge in shape.edges() {
            let points: Vec<_> = match edge.edge_type() {
                EdgeType::Line => vec![edge.start_point(), edge.end_point()],
                _ => edge.approximation_segments().collect(),
            };

            if points.len() < 2 {
                continue;
            }

            for window in points.windows(2) {
                let base = verts.len() as u32;
                for p in window {
                    verts.push(Vertex {
                        position: [p.x as f32 * s, p.y as f32 * s, p.z as f32 * s],
                        normal: [0.0, 0.0, 0.0],
                        tex_coords: [0.0, 0.0, 0.0],
                    });
                }
                indices.push(base);
                indices.push(base + 1);
            }
        }

        if indices.is_empty() {
            continue;
        }

        let name = label.name().unwrap_or(fallback_name);
        let primitive = MeshPrimitive { primitive_type: PrimitiveType::LineList, indices };
        let mesh_id = scene.add_mesh(Mesh::from_raw(verts, vec![primitive]));
        scene
            .add_instance_node(Some(pmi_root), mesh_id, mat, Some(name), Transform::IDENTITY)
            .context("Failed to add PMI annotation node")?;
        any_annotation = true;
    }

    if !any_annotation {
        return Ok(None);
    }

    Ok(Some(pmi_root))
}

/// Extract named views from `view_tool` and add them to the scene.
///
/// Only views with camera data (parallel or perspective projection) are imported.
/// Views with `NoCamera` projection are skipped — these typically encode clipping
/// plane configurations without a viewpoint. When clipping plane support is added,
/// this function should be extended to handle them.
///
/// TODO(cad-views): Import CAD views as camera nodes. Currently stubbed out pending
/// finalization of the camera/view node representation in Phase 1.
fn import_views(
    _view_tool: &opencascade::xcaf::XcafViewTool,
    _scene: &mut Scene,
    _options: &CadImportOptions,
) -> Vec<NodeId> {
    vec![]
}

/// Convert XCAF [`ViewData`] to a Duck [`Camera`].
///
/// All positions are scaled from OCCT model units to scene units via `scale`
/// (e.g. 0.001 for mm→m), matching how vertex positions are scaled in [`import_leaf_part`].
///
/// **Perspective views**: `projection_point` is the eye position; `target` is placed
/// ahead of it along `view_direction`.
///
/// **Orthographic views**: CAD software treats `projection_point` as the center of
/// interest (our `target`), not the camera origin — parallel projection has no
/// meaningful eye position. The eye is placed behind the target along `-view_direction`
/// at a distance derived from `window_vertical_size` so the zoom level matches the
/// saved view. The fallback when `window_vertical_size` is zero is the target's
/// distance from the scene origin.
///
/// Clipping planes and fov are rough defaults; callers should use [`View::apply_to`] with
/// the active camera when they need a properly calibrated result for rendering.
#[allow(dead_code)] // TODO(cad-views): remove when import_views is implemented
fn view_data_to_camera(data: &opencascade::xcaf::ViewData, scale: f32) -> Camera {
    use cgmath::{EuclideanSpace, InnerSpace, MetricSpace, Point3, Vector3};

    let s = scale as f64;
    let pp = data.projection_point;
    let pt = Point3::new((pp[0] * s) as f32, (pp[1] * s) as f32, (pp[2] * s) as f32);

    let vd = data.view_direction;
    let dir = Vector3::new(vd[0] as f32, vd[1] as f32, vd[2] as f32);
    let dir = if dir.magnitude2() > 1e-12 { dir.normalize() } else { Vector3::new(0.0, 0.0, -1.0) };

    let ud = data.up_direction;
    let up = Vector3::new(ud[0] as f32, ud[1] as f32, ud[2] as f32);
    let up = if up.magnitude2() > 1e-12 { up.normalize() } else { Vector3::new(0.0, 1.0, 0.0) };

    let ortho = data.projection() == ViewProjection::Parallel;

    const FOVY: f32 = 45.0;

    let (eye, target) = if ortho {
        // projection_point is the center of interest; place eye behind it.
        let target = pt;
        let half_height = (data.window_vertical_size * s) as f32 / 2.0;
        let orbit_dist = if half_height > 1e-6 {
            half_height / (FOVY.to_radians() / 2.0).tan()
        } else {
            target.distance(Point3::origin()).max(1.0)
        };
        let eye = target - dir * orbit_dist;
        (eye, target)
    } else {
        // projection_point is the eye; target is placed ahead of it.
        let eye = pt;
        let orbit_dist = eye.distance(Point3::origin()).max(1.0);
        let target = eye + dir * orbit_dist;
        (eye, target)
    };

    Camera { eye, target, up, aspect: 1.0, fovy: FOVY, znear: 0.1, zfar: 100_000.0, ortho }
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
        assert!(scene.node_count() > 1, "expected hierarchy preserved via XCAF");
    }

    #[test]
    fn load_step_hierarchy_produces_tree() {
        let path = step_file();
        let mut scene = duck_engine_scene::Scene::new();
        let options = CadImportOptions::default();
        load_step(&path, &mut scene, &options).expect("hierarchy load failed");
        assert!(scene.node_count() > 1, "expected multiple nodes for assembly file");
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
