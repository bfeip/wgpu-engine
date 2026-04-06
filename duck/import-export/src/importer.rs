//! Trait-based importer system for scene file formats.
//!
//! Implement [`Importer`] to add support for a new file format. The built-in
//! importers ([`DuckImporter`], [`GltfImporter`], [`UsdImporter`], [`AssimpImporter`])
//! cover the standard formats; custom importers can be passed to
//! [`load_sync_with`](crate::load_sync_with) / [`load_async_with`](crate::load_async_with).

use std::path::Path;

use crate::{
    DetectedFormat, LoadError, LoadOptions, LoadPhase, LoadProgress, SceneLoadResult,
};

// ============================================================================
// PhaseWeights
// ============================================================================

/// Describes how overall progress is distributed across loading phases.
///
/// Weights must sum to 100. Each entry maps a [`LoadPhase`] to a relative
/// weight that determines what fraction of the progress bar that phase occupies.
#[derive(Debug, Clone)]
pub struct PhaseWeights(pub(crate) [u8; LoadPhase::PHASE_COUNT]);

impl PhaseWeights {
    /// Create an empty weight table (all zeros). Used as the initial state
    /// before an importer binds its weights.
    pub(crate) fn empty() -> Self {
        Self([0u8; LoadPhase::PHASE_COUNT])
    }

    /// Create from phase-weight pairs.
    ///
    /// # Panics
    /// Panics if the weights do not sum to 100.
    pub fn new(weights: &[(LoadPhase, u8)]) -> Self {
        let mut arr = [0u8; LoadPhase::PHASE_COUNT];
        let mut sum = 0u16;
        for &(phase, weight) in weights {
            arr[phase as u8 as usize] = weight;
            sum += weight as u16;
        }
        assert_eq!(sum, 100, "Phase weights must sum to 100, got {sum}");
        Self(arr)
    }

    /// Get the weight for a given phase.
    pub fn get(&self, phase: LoadPhase) -> u8 {
        self.0[phase as u8 as usize]
    }

    /// Sum of all phase weights.
    pub fn total_weight(&self) -> u16 {
        self.0.iter().map(|&x| x as u16).sum()
    }

    /// Sum of weights for all phases with discriminants below `phase`.
    pub fn completed_weight(&self, phase: LoadPhase) -> u16 {
        self.0.iter().take(phase as u8 as usize).map(|&x| x as u16).sum()
    }
}

// ============================================================================
// Importer Trait
// ============================================================================

/// Trait for scene file format importers.
///
/// Implement this trait to add support for loading a new file format.
/// The importer is responsible for format detection, progress reporting,
/// and converting file data into a [`SceneLoadResult`].
///
/// # Format Detection
///
/// When multiple importers are registered, detection proceeds in list order:
/// 1. [`detect_from_bytes`](Importer::detect_from_bytes) is tried on each importer; first `true` wins.
/// 2. If no byte-level match, [`detect_from_extension`](Importer::detect_from_extension) is tried; first `true` wins.
///
/// Use [`default_importers`] to get the built-in set, or supply your own list
/// to [`load_sync_with`](crate::load_sync_with) / [`load_async_with`](crate::load_async_with).
pub trait Importer: Send + Sync {
    /// Human-readable format name (e.g., `"glTF"`, `"Duck"`).
    fn name(&self) -> &str;

    /// Try to detect this format from the first bytes of a file.
    ///
    /// When multiple importers match, the first one in the importer list wins.
    fn detect_from_bytes(&self, bytes: &[u8]) -> bool;

    /// Check if this importer handles the given file extension (without dot, lowercase).
    fn detect_from_extension(&self, ext: &str) -> bool;

    /// Phase weights for progress reporting.
    fn phase_weights(&self) -> PhaseWeights;

    /// Load a scene from bytes.
    ///
    /// `path_hint` is provided when the source was a file path, allowing the
    /// importer to resolve external resources (textures, .bin files, etc.)
    /// relative to it.
    ///
    /// The importer should call [`LoadProgress::enter_phase`] and
    /// [`LoadProgress::complete_item`] to drive progress reporting. The weights
    /// are already set before this method is called.
    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        options: &LoadOptions,
        progress: &LoadProgress,
    ) -> Result<SceneLoadResult, LoadError>;
}

// ============================================================================
// Built-in Importers
// ============================================================================

/// Importer for the Duck binary scene format.
pub struct DuckImporter;

impl Importer for DuckImporter {
    fn name(&self) -> &str {
        "Duck"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        bytes.len() >= 4 && bytes.starts_with(&crate::format::MAGIC)
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("duck")
    }

    fn phase_weights(&self) -> PhaseWeights {
        PhaseWeights::new(&[
            (LoadPhase::Reading, 10),
            (LoadPhase::Parsing, 10),
            (LoadPhase::DecodingTextures, 60),
            (LoadPhase::Assembling, 20),
        ])
    }

    fn load(
        &self,
        bytes: &[u8],
        _path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &LoadProgress,
    ) -> Result<SceneLoadResult, LoadError> {
        use crate::format::{assemble_duck_scene, decode_duck_texture, parse_duck};

        progress.enter_phase(LoadPhase::Parsing);
        let sections = parse_duck(bytes)?;

        progress.enter_phase(LoadPhase::DecodingTextures);
        progress.set_item_count(sections.textures.len() as u32);

        let mut decoded = Vec::with_capacity(sections.textures.len());
        for st in &sections.textures {
            decoded.push(decode_duck_texture(st)?);
            progress.complete_item();
        }

        progress.enter_phase(LoadPhase::Assembling);
        let scene = assemble_duck_scene(sections, decoded)?;

        progress.enter_phase(LoadPhase::Complete);
        Ok(SceneLoadResult {
            scene,
            camera: None,
            format: DetectedFormat::Duck,
        })
    }
}

/// Importer for glTF / GLB files.
pub struct GltfImporter;

impl Importer for GltfImporter {
    fn name(&self) -> &str {
        "glTF"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        if bytes.len() < 4 {
            return false;
        }
        // GLB binary magic
        if bytes.starts_with(b"glTF") {
            return true;
        }
        // glTF JSON: starts with '{' after optional whitespace
        let first_non_ws = bytes.iter().position(|&b| !b.is_ascii_whitespace());
        matches!(first_non_ws, Some(pos) if bytes[pos] == b'{')
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        ext.eq_ignore_ascii_case("glb") || ext.eq_ignore_ascii_case("gltf")
    }

    fn phase_weights(&self) -> PhaseWeights {
        PhaseWeights::new(&[
            (LoadPhase::Reading, 10),
            (LoadPhase::Parsing, 10),
            (LoadPhase::DecodingTextures, 50),
            (LoadPhase::Assembling, 30),
        ])
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        options: &LoadOptions,
        progress: &LoadProgress,
    ) -> Result<SceneLoadResult, LoadError> {
        use crate::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf, parse_gltf_from_path};
        use duck_engine_scene::Scene;

        progress.enter_phase(LoadPhase::Parsing);
        let parsed = if let Some(path) = path_hint {
            parse_gltf_from_path(path)
        } else {
            parse_gltf(bytes)
        }
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

        progress.enter_phase(LoadPhase::DecodingTextures);
        let mut scene = Scene::new();
        let (_material_map, mesh_map) = load_gltf_assets(&parsed, &mut scene)
            .map_err(|e| LoadError::Gltf(e.to_string()))?;

        progress.enter_phase(LoadPhase::Assembling);
        let camera = build_gltf_scene(&parsed, &mut scene, &mesh_map, options.aspect)
            .map_err(|e| LoadError::Gltf(e.to_string()))?;

        progress.enter_phase(LoadPhase::Complete);
        Ok(SceneLoadResult {
            scene,
            camera,
            format: DetectedFormat::Gltf,
        })
    }
}

/// Importer for USD files (USDC, USDA, USDZ).
#[cfg(feature = "usd")]
pub struct UsdImporter;

#[cfg(feature = "usd")]
impl Importer for UsdImporter {
    fn name(&self) -> &str {
        "USD"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        // USDC binary starts with "PXR-USDC"
        if bytes.len() >= 8 && bytes.starts_with(b"PXR-USDC") {
            return true;
        }
        // USDZ is a ZIP archive starting with "PK"
        bytes.len() >= 2 && bytes[0] == b'P' && bytes[1] == b'K'
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        crate::usd::is_usd_extension(ext)
    }

    fn phase_weights(&self) -> PhaseWeights {
        PhaseWeights::new(&[
            (LoadPhase::Reading, 10),
            (LoadPhase::Parsing, 30),
            (LoadPhase::BuildingMeshes, 40),
            (LoadPhase::Assembling, 20),
        ])
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &LoadProgress,
    ) -> Result<SceneLoadResult, LoadError> {
        progress.enter_phase(LoadPhase::Parsing);

        let result = if let Some(path) = path_hint {
            crate::usd::load_usd_scene_from_path(path)
        } else {
            crate::usd::load_usd_scene_from_bytes(bytes, "scene.usd")
        };

        let usd_result = result.map_err(|e| LoadError::Usd(e.to_string()))?;

        progress.enter_phase(LoadPhase::Complete);
        Ok(SceneLoadResult {
            scene: usd_result.scene,
            camera: usd_result.camera,
            format: DetectedFormat::Usd,
        })
    }
}

/// Importer for Assimp-supported formats (OBJ, FBX, DAE, etc.).
#[cfg(feature = "assimp")]
pub struct AssimpImporter;

#[cfg(feature = "assimp")]
impl Importer for AssimpImporter {
    fn name(&self) -> &str {
        "Assimp"
    }

    fn detect_from_bytes(&self, _bytes: &[u8]) -> bool {
        // Assimp formats have no single magic byte sequence we can check
        false
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        crate::assimp::is_assimp_extension(ext)
    }

    fn phase_weights(&self) -> PhaseWeights {
        PhaseWeights::new(&[
            (LoadPhase::Reading, 10),
            (LoadPhase::Parsing, 30),
            (LoadPhase::BuildingMeshes, 40),
            (LoadPhase::Assembling, 20),
        ])
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &LoadProgress,
    ) -> Result<SceneLoadResult, LoadError> {
        progress.enter_phase(LoadPhase::Parsing);

        let result = if let Some(path) = path_hint {
            crate::assimp::load_assimp_scene_from_path(path)
        } else {
            crate::assimp::load_assimp_scene_from_bytes(bytes, "")
        };

        let assimp_result = result.map_err(|e| LoadError::Assimp(e.to_string()))?;

        progress.enter_phase(LoadPhase::Complete);
        Ok(SceneLoadResult {
            scene: assimp_result.scene,
            camera: assimp_result.camera,
            format: DetectedFormat::Assimp,
        })
    }
}

// ============================================================================
// Importer Registry
// ============================================================================

/// Importer for STEP and IGES CAD files.
///
/// Uses OpenCASCADE via the `duck-engine-cad` crate for tessellation.
/// Requires a filesystem path (`path_hint`); loading from raw bytes is not
/// supported by the underlying `opencascade-rs` bindings.
#[cfg(feature = "cad")]
pub struct CadImporter;

#[cfg(feature = "cad")]
impl Importer for CadImporter {
    fn name(&self) -> &str {
        "CAD"
    }

    fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
        // STEP files begin with the ISO 10303-21 exchange format header.
        // IGES has no reliable magic bytes; it falls back to extension detection.
        bytes.starts_with(b"ISO-10303-21")
    }

    fn detect_from_extension(&self, ext: &str) -> bool {
        crate::cad::is_cad_extension(ext)
    }

    fn phase_weights(&self) -> PhaseWeights {
        // OCCT tessellation is a monolithic step with no item-level progress,
        // so we give it the bulk of the weight under Parsing.
        PhaseWeights::new(&[
            (LoadPhase::Reading, 10),
            (LoadPhase::Parsing, 80),
            (LoadPhase::Assembling, 10),
        ])
    }

    fn load(
        &self,
        bytes: &[u8],
        path_hint: Option<&Path>,
        _options: &LoadOptions,
        progress: &LoadProgress,
    ) -> Result<SceneLoadResult, LoadError> {
        let path = path_hint.ok_or_else(|| {
            LoadError::UnsupportedPlatform(
                "CAD loading requires a file path (in-memory loading is not supported by opencascade-rs)".into(),
            )
        })?;

        progress.enter_phase(LoadPhase::Parsing);

        let is_step = bytes.starts_with(b"ISO-10303-21")
            || path
                .extension()
                .and_then(|e| e.to_str())
                .map(crate::cad::is_step_extension)
                .unwrap_or(false);

        let mut scene = duck_engine_scene::Scene::new();
        let options = cad::CadImportOptions::default();

        let result = if is_step {
            cad::load_step(path, &mut scene, &options)
                .map_err(|e| LoadError::Cad(e.to_string()))
        } else {
            cad::load_iges(path, &mut scene, &options)
                .map_err(|e| LoadError::Cad(e.to_string()))
        };
        result?;

        progress.enter_phase(LoadPhase::Assembling);
        progress.enter_phase(LoadPhase::Complete);

        let format = if is_step {
            DetectedFormat::Step
        } else {
            DetectedFormat::Iges
        };

        Ok(SceneLoadResult {
            scene,
            camera: None,
            format,
        })
    }
}

/// Returns the default set of built-in importers.
///
/// The order determines detection priority (first match wins):
/// Duck, glTF, CAD (if enabled), USD (if enabled), Assimp (if enabled).
pub fn default_importers() -> Vec<Box<dyn Importer>> {
    #[allow(unused_mut)]
    let mut importers: Vec<Box<dyn Importer>> = vec![
        Box::new(DuckImporter),
        Box::new(GltfImporter),
    ];
    #[cfg(feature = "cad")]
    importers.push(Box::new(CadImporter));
    #[cfg(feature = "usd")]
    importers.push(Box::new(UsdImporter));
    #[cfg(feature = "assimp")]
    importers.push(Box::new(AssimpImporter));
    importers
}

/// Find the first importer that recognizes the given data.
///
/// Tries magic-byte detection first (in list order), then falls back to
/// file-extension detection if a path hint is available.
pub fn detect_importer<'a>(
    bytes: &[u8],
    path_hint: Option<&Path>,
    importers: &'a [Box<dyn Importer>],
) -> Result<&'a dyn Importer, LoadError> {
    // First pass: magic bytes
    for imp in importers {
        if imp.detect_from_bytes(bytes) {
            return Ok(imp.as_ref());
        }
    }

    // Second pass: file extension fallback
    if let Some(path) = path_hint {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            for imp in importers {
                if imp.detect_from_extension(ext) {
                    return Ok(imp.as_ref());
                }
            }
        }
    }

    Err(LoadError::UnknownFormat)
}
