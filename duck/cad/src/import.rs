use anyhow::{Context, Result};
use cgmath::Matrix4;
use duck_engine_common::decompose_matrix;
use duck_engine_scene::common::{RgbaColor, Transform};
use duck_engine_scene::{
    Instance, Material, Mesh, MeshPrimitive, NodeId, NodePayload, PositionedCamera, PrimitiveType,
    Scene, Vertex,
};
use opencascade::primitives::{EdgeType, Shape};
use opencascade::xcaf::{
    ViewProjection, XcafColorTool, XcafDimTolTool, XcafDocument, XcafLabel, XcafShapeTool,
};

use crate::tessellate::tessellate_occ_shape;
use crate::{CadImportOptions, CadImportResult};

pub(crate) fn import_xcaf_document(
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

fn import_leaf_part(
    shape: &Shape,
    scene: &mut Scene,
    options: &CadImportOptions,
    node: NodeId,
    face_color: RgbaColor,
) -> Result<()> {
    let mesh =
        tessellate_occ_shape(shape, options.tessellation_tolerance, options.scale_factor, options.include_edges)?;
    let mat = scene.add_material(
        Material::new()
            .with_base_color_factor(face_color)
            .with_line_color(options.edge_color),
    );
    let mesh_id = scene.add_mesh(mesh);
    let instance_id = scene.add_instance(Instance::new(mesh_id, mat));
    scene.set_node_payload(node, NodePayload::Instance(instance_id));
    Ok(())
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

/// TODO(cad-views): Import CAD views as camera nodes. Currently stubbed out pending
/// finalization of the camera/view node representation.
pub(crate) fn import_views(
    _view_tool: &opencascade::xcaf::XcafViewTool,
    _scene: &mut Scene,
    _options: &CadImportOptions,
) -> Vec<NodeId> {
    vec![]
}

/// Convert XCAF [`ViewData`] to a Duck [`PositionedCamera`].
#[allow(dead_code)] // TODO(cad-views): remove when import_views is implemented
fn view_data_to_camera(data: &opencascade::xcaf::ViewData, scale: f32) -> PositionedCamera {
    use cgmath::{EuclideanSpace, InnerSpace, MetricSpace, Point3, Vector3};

    let s = scale as f64;
    let pp = data.projection_point;
    let pt = Point3::new((pp[0] * s) as f32, (pp[1] * s) as f32, (pp[2] * s) as f32);

    let vd = data.view_direction;
    let dir = Vector3::new(vd[0] as f32, vd[1] as f32, vd[2] as f32);
    let dir =
        if dir.magnitude2() > 1e-12 { dir.normalize() } else { Vector3::new(0.0, 0.0, -1.0) };

    let ud = data.up_direction;
    let up = Vector3::new(ud[0] as f32, ud[1] as f32, ud[2] as f32);
    let up = if up.magnitude2() > 1e-12 { up.normalize() } else { Vector3::new(0.0, 1.0, 0.0) };

    let ortho = data.projection() == ViewProjection::Parallel;

    const FOVY: f32 = 45.0;

    let (eye, target) = if ortho {
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
        let eye = pt;
        let orbit_dist = eye.distance(Point3::origin()).max(1.0);
        let target = eye + dir * orbit_dist;
        (eye, target)
    };

    PositionedCamera { eye, target, up, aspect: 1.0, fovy: FOVY, znear: 0.1, zfar: 100_000.0, ortho }
}
