use std::path::Path;

use anyhow::{Context, Result};
use duck_engine_scene::common::RgbaColor;
use duck_engine_scene::{NodeId, Scene};

mod import;
mod tessellate;
pub mod shape;
pub mod wire;
pub mod model;

pub use model::{CadModel, CadShapeId};
pub use shape::CadShape;
pub use wire::{CadEdge, CadWire};

/// Options controlling tessellation and presentation when producing scene geometry from CAD data.
///
/// Used for both file import (`load_step`, `load_iges`) and programmatic authoring
/// (`CadShape::tessellate_into`, `CadModel::tessellate`).
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
    /// Whether to include wireframe edges as `LineList` meshes.
    pub include_edges: bool,
    /// Whether to import PMI graphical presentation geometry as `LineList` meshes.
    /// Only applies to file imports; ignored for programmatic shapes.
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
    pub pmi_root: Option<NodeId>,
    /// Camera nodes for named views imported from the CAD file.
    ///
    /// TODO(cad-views): implement once camera/view node representation is finalized.
    pub views: Vec<NodeId>,
}

/// Import a STEP file into `scene`, returning a [`CadImportResult`] that mirrors
/// the assembly hierarchy.
pub fn load_step(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = opencascade::xcaf::XcafDocument::read_step(path.as_ref())
        .with_context(|| format!("Failed to read STEP file: {}", path.as_ref().display()))?;
    import::import_xcaf_document(&doc, scene, options)
}

/// Import STEP data from a string into `scene`.
pub fn load_step_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = opencascade::xcaf::XcafDocument::read_step_from_str(s)
        .context("Failed to read STEP data")?;
    import::import_xcaf_document(&doc, scene, options)
}

/// Import an IGES file into `scene`, returning a [`CadImportResult`].
pub fn load_iges(
    path: impl AsRef<Path>,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = opencascade::xcaf::XcafDocument::read_iges(path.as_ref())
        .with_context(|| format!("Failed to read IGES file: {}", path.as_ref().display()))?;
    import::import_xcaf_document(&doc, scene, options)
}

/// Import IGES data from a string into `scene`.
pub fn load_iges_from_str(
    s: &str,
    scene: &mut Scene,
    options: &CadImportOptions,
) -> Result<CadImportResult> {
    let doc = opencascade::xcaf::XcafDocument::read_iges_from_str(s)
        .context("Failed to read IGES data")?;
    import::import_xcaf_document(&doc, scene, options)
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
        let path = step_file();
        let mut scene = duck_engine_scene::Scene::new();
        let options = CadImportOptions { include_pmi: true, ..Default::default() };
        let result = load_step(&path, &mut scene, &options).expect("PMI load failed");
        if let Some(pmi_root) = result.pmi_root {
            assert!(scene.node_count() > 1);
            let _ = pmi_root;
        }
    }
}
