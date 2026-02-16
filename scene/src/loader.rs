//! Async scene import/export library.
//!
//! Provides format-agnostic loading and saving of scenes with progress reporting.
//! Supports glTF (.glb/.gltf) and WGSC (.wgsc) formats with automatic detection.
//!
//! # Native
//! Loading runs on a background thread via [`std::thread::spawn`]. The caller
//! polls [`LoadHandle`] each frame for completion and progress.
//!
//! # WASM
//! Loading uses chunked yielding via `wasm_bindgen_futures::spawn_local`,
//! yielding to the browser event loop between loading phases so the page
//! stays responsive.
//!
//! # Examples
//!
//! ```no_run
//! use wgpu_engine_scene::loader::{load_async, LoadOptions, SceneSource};
//!
//! let handle = load_async(SceneSource::Path("model.glb".into()), LoadOptions::default());
//!
//! // In your render loop:
//! // if let Some(result) = handle.try_recv() { ... }
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use thiserror::Error;

use crate::camera::Camera;
use crate::format::FormatError;
use crate::Scene;

// ============================================================================
// Type Aliases
// ============================================================================

/// The result type produced by a completed load operation.
type LoadResult = Result<SceneLoadResult, LoadError>;

/// Receiver for the native async load result.
#[cfg(not(target_arch = "wasm32"))]
type LoadReceiver = std::sync::mpsc::Receiver<LoadResult>;

/// Shared cell for the WASM async load result.
#[cfg(target_arch = "wasm32")]
type LoadResultCell = std::rc::Rc<std::cell::RefCell<Option<LoadResult>>>;

// ============================================================================
// Types
// ============================================================================

/// Where scene data comes from.
pub enum SceneSource {
    /// Load from a filesystem path. Format is auto-detected from magic bytes,
    /// falling back to file extension.
    Path(PathBuf),
    /// Load from bytes already in memory. Format is auto-detected from magic bytes.
    Bytes(Vec<u8>),
}

/// Options that control scene loading behavior.
pub struct LoadOptions {
    /// Aspect ratio for cameras embedded in glTF files. Ignored for other formats.
    /// Default: 16.0 / 9.0.
    pub aspect: f32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            aspect: 16.0 / 9.0,
        }
    }
}

/// The result of a successful scene load.
pub struct SceneLoadResult {
    /// The loaded scene.
    pub scene: Scene,
    /// Camera extracted from the file, if present (glTF only).
    pub camera: Option<Camera>,
    /// Which format was detected and loaded.
    pub format: DetectedFormat,
}

/// The file format that was detected and used for loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedFormat {
    Wgsc,
    Gltf,
}

/// Coarse loading phases for progress display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LoadPhase {
    Pending = 0,
    Reading = 1,
    Parsing = 2,
    DecodingTextures = 3,
    BuildingMeshes = 4,
    Assembling = 5,
    Complete = 6,
    Failed = 7,
}

impl LoadPhase {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Pending,
            1 => Self::Reading,
            2 => Self::Parsing,
            3 => Self::DecodingTextures,
            4 => Self::BuildingMeshes,
            5 => Self::Assembling,
            6 => Self::Complete,
            7 => Self::Failed,
            _ => Self::Failed,
        }
    }
}

/// Errors that can occur during scene loading.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Format error: {0}")]
    Format(#[from] FormatError),

    #[error("glTF error: {0}")]
    Gltf(String),

    #[error("Unknown file format")]
    UnknownFormat,
}

/// Shared progress state, readable from any thread.
pub struct LoadProgress {
    phase: Arc<AtomicU8>,
    progress_pct: Arc<AtomicU8>,
    items_total: Arc<AtomicU32>,
    items_complete: Arc<AtomicU32>,
}

impl LoadProgress {
    fn new() -> Self {
        Self {
            phase: Arc::new(AtomicU8::new(LoadPhase::Pending as u8)),
            progress_pct: Arc::new(AtomicU8::new(0)),
            items_total: Arc::new(AtomicU32::new(0)),
            items_complete: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Current loading phase.
    pub fn phase(&self) -> LoadPhase {
        LoadPhase::from_u8(self.phase.load(Ordering::Relaxed))
    }

    /// Overall progress percentage (0-100).
    pub fn progress_pct(&self) -> u8 {
        self.progress_pct.load(Ordering::Relaxed)
    }

    /// Total number of items in the current phase (e.g., textures to decode).
    pub fn items_total(&self) -> u32 {
        self.items_total.load(Ordering::Relaxed)
    }

    /// Number of items completed in the current phase.
    pub fn items_complete(&self) -> u32 {
        self.items_complete.load(Ordering::Relaxed)
    }

    fn set_phase(&self, phase: LoadPhase, pct: u8) {
        self.phase.store(phase as u8, Ordering::Relaxed);
        self.progress_pct.store(pct, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    fn set_items(&self, total: u32, complete: u32) {
        self.items_total.store(total, Ordering::Relaxed);
        self.items_complete.store(complete, Ordering::Relaxed);
    }

    fn clone_arcs(&self) -> Self {
        Self {
            phase: Arc::clone(&self.phase),
            progress_pct: Arc::clone(&self.progress_pct),
            items_total: Arc::clone(&self.items_total),
            items_complete: Arc::clone(&self.items_complete),
        }
    }
}

/// Handle to a loading operation in progress.
///
/// Poll this each frame with [`try_recv`](LoadHandle::try_recv) to check for
/// completion, and read [`progress`](LoadHandle::progress) to display a
/// progress bar.
pub struct LoadHandle {
    progress: LoadProgress,
    done: Arc<AtomicBool>,
    #[cfg(not(target_arch = "wasm32"))]
    receiver: LoadReceiver,
    #[cfg(target_arch = "wasm32")]
    result: LoadResultCell,
}

impl LoadHandle {
    /// Returns `Some(result)` if loading has completed, `None` if still in progress.
    ///
    /// Consumes the result on first successful call. Subsequent calls return `None`.
    pub fn try_recv(&self) -> Option<LoadResult> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.receiver.try_recv().ok()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.result.borrow_mut().take()
        }
    }

    /// Returns the shared progress state.
    pub fn progress(&self) -> &LoadProgress {
        &self.progress
    }

    /// Returns true if loading has completed (success or failure).
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

// ============================================================================
// Format Detection
// ============================================================================

/// Detect format from magic bytes.
fn detect_format_from_bytes(bytes: &[u8]) -> Result<DetectedFormat, LoadError> {
    if bytes.len() < 4 {
        return Err(LoadError::UnknownFormat);
    }

    if bytes.starts_with(b"WGSC") {
        return Ok(DetectedFormat::Wgsc);
    }

    // glTF binary (.glb) starts with "glTF"
    if bytes.starts_with(b"glTF") {
        return Ok(DetectedFormat::Gltf);
    }

    // glTF JSON starts with '{' (possibly with leading whitespace/BOM)
    let trimmed = bytes.iter().position(|&b| !b.is_ascii_whitespace());
    if let Some(pos) = trimmed {
        if bytes[pos] == b'{' {
            return Ok(DetectedFormat::Gltf);
        }
    }

    Err(LoadError::UnknownFormat)
}

/// Detect format from file extension as a fallback.
fn detect_format_from_extension(path: &std::path::Path) -> Result<DetectedFormat, LoadError> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("wgsc") => Ok(DetectedFormat::Wgsc),
        Some("glb") | Some("gltf") => Ok(DetectedFormat::Gltf),
        _ => Err(LoadError::UnknownFormat),
    }
}

// ============================================================================
// Sync Loading (core logic)
// ============================================================================

/// Load a scene synchronously. Useful for CLI tools and tests.
pub fn load_sync(source: SceneSource, options: LoadOptions) -> LoadResult {
    let progress = LoadProgress::new();
    load_sync_with_progress(source, &options, &progress)
}

fn load_sync_with_progress(
    source: SceneSource,
    options: &LoadOptions,
    progress: &LoadProgress,
) -> LoadResult {
    progress.set_phase(LoadPhase::Reading, 0);

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    progress.set_phase(LoadPhase::Parsing, 5);

    let format = detect_format_from_bytes(&bytes).or_else(|_| {
        path_hint
            .as_ref()
            .map(|p| detect_format_from_extension(p))
            .unwrap_or(Err(LoadError::UnknownFormat))
    })?;

    match format {
        DetectedFormat::Wgsc => load_wgsc_phased(&bytes, progress),
        DetectedFormat::Gltf => load_gltf_phased(&bytes, options, progress),
    }
}

fn load_wgsc_phased(bytes: &[u8], progress: &LoadProgress) -> LoadResult {
    use crate::format::{assemble_wgsc_scene, decode_wgsc_texture, parse_wgsc};

    progress.set_phase(LoadPhase::Parsing, 10);
    let sections = parse_wgsc(bytes)?;

    progress.set_phase(LoadPhase::DecodingTextures, 20);
    let total = sections.textures.len() as u32;
    progress.set_items(total, 0);

    let mut decoded = Vec::with_capacity(total as usize);
    for (i, st) in sections.textures.iter().enumerate() {
        decoded.push(decode_wgsc_texture(st)?);
        let complete = (i + 1) as u32;
        progress.set_items(total, complete);
        // Scale progress: textures span 20%–80%
        let pct = 20 + (60 * complete / total.max(1)) as u8;
        progress.progress_pct.store(pct, Ordering::Relaxed);
    }

    progress.set_phase(LoadPhase::Assembling, 80);
    let scene = assemble_wgsc_scene(sections, decoded)?;

    progress.set_phase(LoadPhase::Complete, 100);
    Ok(SceneLoadResult {
        scene,
        camera: None,
        format: DetectedFormat::Wgsc,
    })
}

fn load_gltf_phased(bytes: &[u8], options: &LoadOptions, progress: &LoadProgress) -> LoadResult {
    use crate::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf};

    progress.set_phase(LoadPhase::Parsing, 10);
    let parsed = parse_gltf(bytes).map_err(|e| LoadError::Gltf(e.to_string()))?;

    progress.set_phase(LoadPhase::DecodingTextures, 30);
    let mut scene = Scene::new();
    let (_material_map, mesh_map) = load_gltf_assets(&parsed, &mut scene)
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

    progress.set_phase(LoadPhase::Assembling, 80);
    let camera = build_gltf_scene(&parsed, &mut scene, &mesh_map, options.aspect)
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

    progress.set_phase(LoadPhase::Complete, 100);
    Ok(SceneLoadResult {
        scene,
        camera,
        format: DetectedFormat::Gltf,
    })
}

// ============================================================================
// Async Loading — Native
// ============================================================================

/// Start loading a scene asynchronously.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between loading phases.
///
/// Returns a [`LoadHandle`] to poll for completion and progress.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_async(source: SceneSource, options: LoadOptions) -> LoadHandle {
    let progress = LoadProgress::new();
    let progress_clone = progress.clone_arcs();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let result = load_sync_with_progress(source, &options, &progress_clone);
        if result.is_err() {
            progress_clone.set_phase(LoadPhase::Failed, 0);
        }
        done_clone.store(true, Ordering::Release);
        let _ = tx.send(result);
    });

    LoadHandle {
        progress,
        done,
        receiver: rx,
    }
}

// ============================================================================
// Async Loading — WASM
// ============================================================================

/// Start loading a scene asynchronously.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between loading phases.
///
/// Returns a [`LoadHandle`] to poll for completion and progress.
#[cfg(target_arch = "wasm32")]
pub fn load_async(source: SceneSource, options: LoadOptions) -> LoadHandle {
    use std::cell::RefCell;
    use std::rc::Rc;

    let progress = LoadProgress::new();
    let progress_clone = progress.clone_arcs();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let result_cell: LoadResultCell = Rc::new(RefCell::new(None));
    let result_clone = Rc::clone(&result_cell);

    wasm_bindgen_futures::spawn_local(async move {
        let result = load_chunked_wasm(source, &options, &progress_clone).await;
        if result.is_err() {
            progress_clone.set_phase(LoadPhase::Failed, 0);
        }
        done_clone.store(true, Ordering::Release);
        *result_clone.borrow_mut() = Some(result);
    });

    LoadHandle {
        progress,
        done,
        result: result_cell,
    }
}

/// Yield to the browser event loop. Allows the page to process events
/// (rendering, input) between expensive loading phases.
///
/// Uses `setTimeout(0)` to schedule a macrotask, ensuring the browser gets
/// a chance to render and run `requestAnimationFrame` between yield points.
/// A resolved `Promise` alone only creates a microtask, which runs before
/// the browser paints.
#[cfg(target_arch = "wasm32")]
async fn yield_to_event_loop() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let set_timeout =
            js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout"))
                .expect("setTimeout not found on global");
        let set_timeout: js_sys::Function = set_timeout.into();
        let _ = set_timeout.call2(
            &wasm_bindgen::JsValue::NULL,
            &resolve,
            &wasm_bindgen::JsValue::from(0),
        );
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// WASM chunked loading: runs loading phases with yields between them.
#[cfg(target_arch = "wasm32")]
async fn load_chunked_wasm(
    source: SceneSource,
    options: &LoadOptions,
    progress: &LoadProgress,
) -> LoadResult {
    progress.set_phase(LoadPhase::Reading, 0);

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            // Note: std::fs::read doesn't work on WASM. Callers should use
            // SceneSource::Bytes. This arm exists for API uniformity.
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    progress.set_phase(LoadPhase::Parsing, 5);
    yield_to_event_loop().await;

    let format = detect_format_from_bytes(&bytes).or_else(|_| {
        path_hint
            .as_ref()
            .map(|p| detect_format_from_extension(p))
            .unwrap_or(Err(LoadError::UnknownFormat))
    })?;

    match format {
        DetectedFormat::Wgsc => load_wgsc_chunked(&bytes, progress).await,
        DetectedFormat::Gltf => load_gltf_chunked(&bytes, options, progress).await,
    }
}

#[cfg(target_arch = "wasm32")]
async fn load_wgsc_chunked(bytes: &[u8], progress: &LoadProgress) -> LoadResult {
    use crate::format::{assemble_wgsc_scene, decode_wgsc_texture, parse_wgsc};

    progress.set_phase(LoadPhase::Parsing, 10);
    let sections = parse_wgsc(bytes)?;
    yield_to_event_loop().await;

    progress.set_phase(LoadPhase::DecodingTextures, 20);
    let total = sections.textures.len() as u32;
    progress.set_items(total, 0);

    let mut decoded = Vec::with_capacity(total as usize);
    for (i, st) in sections.textures.iter().enumerate() {
        decoded.push(decode_wgsc_texture(st)?);
        let complete = (i + 1) as u32;
        progress.set_items(total, complete);
        let pct = 20 + (60 * complete / total.max(1)) as u8;
        progress.progress_pct.store(pct, Ordering::Relaxed);
        yield_to_event_loop().await;
    }

    progress.set_phase(LoadPhase::Assembling, 80);
    yield_to_event_loop().await;

    let scene = assemble_wgsc_scene(sections, decoded)?;

    progress.set_phase(LoadPhase::Complete, 100);
    Ok(SceneLoadResult {
        scene,
        camera: None,
        format: DetectedFormat::Wgsc,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_gltf_chunked(
    bytes: &[u8],
    options: &LoadOptions,
    progress: &LoadProgress,
) -> LoadResult {
    use crate::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf};

    progress.set_phase(LoadPhase::Parsing, 10);
    let parsed = parse_gltf(bytes).map_err(|e| LoadError::Gltf(e.to_string()))?;
    yield_to_event_loop().await;

    progress.set_phase(LoadPhase::DecodingTextures, 30);
    let mut scene = Scene::new();
    let (_material_map, mesh_map) =
        load_gltf_assets(&parsed, &mut scene).map_err(|e| LoadError::Gltf(e.to_string()))?;
    yield_to_event_loop().await;

    progress.set_phase(LoadPhase::Assembling, 80);
    let camera = build_gltf_scene(&parsed, &mut scene, &mesh_map, options.aspect)
        .map_err(|e| LoadError::Gltf(e.to_string()))?;

    progress.set_phase(LoadPhase::Complete, 100);
    Ok(SceneLoadResult {
        scene,
        camera,
        format: DetectedFormat::Gltf,
    })
}

// ============================================================================
// Export Types
// ============================================================================

/// Where to write the exported scene.
pub enum SaveDestination {
    /// Write to a filesystem path.
    Path(PathBuf),
    /// Return the bytes in memory.
    Bytes,
}

/// The result type produced by a completed save operation.
type SaveResult = Result<Option<Vec<u8>>, LoadError>;

/// Receiver for the native async save result.
#[cfg(not(target_arch = "wasm32"))]
type SaveReceiver = std::sync::mpsc::Receiver<SaveResult>;

/// Shared cell for the WASM async save result.
#[cfg(target_arch = "wasm32")]
type SaveResultCell = std::rc::Rc<std::cell::RefCell<Option<SaveResult>>>;

/// Handle to a save operation in progress.
///
/// Poll this each frame with [`try_recv`](SaveHandle::try_recv) to check for
/// completion.
pub struct SaveHandle {
    progress: LoadProgress,
    done: Arc<AtomicBool>,
    #[cfg(not(target_arch = "wasm32"))]
    receiver: SaveReceiver,
    #[cfg(target_arch = "wasm32")]
    result: SaveResultCell,
}

impl SaveHandle {
    /// Returns `Some(result)` if saving has completed, `None` if still in progress.
    ///
    /// The inner `Option<Vec<u8>>` is `Some` when saving to bytes
    /// ([`SaveDestination::Bytes`]), `None` when saving to a file.
    pub fn try_recv(&self) -> Option<SaveResult> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.receiver.try_recv().ok()
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.result.borrow_mut().take()
        }
    }

    /// Returns the shared progress state.
    pub fn progress(&self) -> &LoadProgress {
        &self.progress
    }

    /// Returns true if saving has completed (success or failure).
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

// ============================================================================
// Sync Export
// ============================================================================

/// Save a scene synchronously. Useful for CLI tools and tests.
pub fn save_sync(
    scene: &mut Scene,
    dest: SaveDestination,
    options: crate::format::SaveOptions,
) -> SaveResult {
    let progress = LoadProgress::new();
    save_sync_with_progress(scene, dest, &options, &progress)
}

fn save_sync_with_progress(
    scene: &mut Scene,
    dest: SaveDestination,
    options: &crate::format::SaveOptions,
    progress: &LoadProgress,
) -> SaveResult {
    progress.set_phase(LoadPhase::Parsing, 10); // "Parsing" = serializing in export context
    let bytes = scene
        .to_bytes_with_options(options)
        .map_err(LoadError::Format)?;

    progress.set_phase(LoadPhase::Assembling, 80);
    match dest {
        SaveDestination::Path(path) => {
            std::fs::write(path, &bytes)?;
            progress.set_phase(LoadPhase::Complete, 100);
            Ok(None)
        }
        SaveDestination::Bytes => {
            progress.set_phase(LoadPhase::Complete, 100);
            Ok(Some(bytes))
        }
    }
}

// ============================================================================
// Async Export — Native
// ============================================================================

/// Start saving a scene asynchronously.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask.
///
/// Note: The scene is moved into the background task. If you need the scene
/// afterward, clone it first.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_async(
    mut scene: Scene,
    dest: SaveDestination,
    options: crate::format::SaveOptions,
) -> SaveHandle {
    let progress = LoadProgress::new();
    let progress_clone = progress.clone_arcs();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let result = save_sync_with_progress(&mut scene, dest, &options, &progress_clone);
        if result.is_err() {
            progress_clone.set_phase(LoadPhase::Failed, 0);
        }
        done_clone.store(true, Ordering::Release);
        let _ = tx.send(result);
    });

    SaveHandle {
        progress,
        done,
        receiver: rx,
    }
}

// ============================================================================
// Async Export — WASM
// ============================================================================

/// Start saving a scene asynchronously.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask.
///
/// Note: The scene is moved into the background task. If you need the scene
/// afterward, clone it first.
#[cfg(target_arch = "wasm32")]
pub fn save_async(
    mut scene: Scene,
    dest: SaveDestination,
    options: crate::format::SaveOptions,
) -> SaveHandle {
    use std::cell::RefCell;
    use std::rc::Rc;

    let progress = LoadProgress::new();
    let progress_clone = progress.clone_arcs();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let result_cell: SaveResultCell = Rc::new(RefCell::new(None));
    let result_clone = Rc::clone(&result_cell);

    wasm_bindgen_futures::spawn_local(async move {
        let result = save_sync_with_progress(&mut scene, dest, &options, &progress_clone);
        if result.is_err() {
            progress_clone.set_phase(LoadPhase::Failed, 0);
        }
        done_clone.store(true, Ordering::Release);
        *result_clone.borrow_mut() = Some(result);
    });

    SaveHandle {
        progress,
        done,
        result: result_cell,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Material, Mesh};
    use cgmath::{Point3, Quaternion, Vector3};

    fn create_test_scene() -> Scene {
        let mut scene = Scene::new();
        let mesh = Mesh::cube(1.0);
        let mesh_id = scene.add_mesh(mesh);
        let mat_id = scene.add_material(Material::new());
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some("TestNode".into()),
                Point3::new(0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Vector3::new(1.0, 1.0, 1.0),
            )
            .unwrap();
        scene
    }

    fn create_test_scene_bytes() -> Vec<u8> {
        create_test_scene().to_bytes().unwrap()
    }

    #[test]
    fn test_detect_wgsc() {
        let bytes = b"WGSC\x01\x00rest of file";
        assert_eq!(
            detect_format_from_bytes(bytes).unwrap(),
            DetectedFormat::Wgsc
        );
    }

    #[test]
    fn test_detect_glb() {
        let bytes = b"glTF\x02\x00\x00\x00rest";
        assert_eq!(
            detect_format_from_bytes(bytes).unwrap(),
            DetectedFormat::Gltf
        );
    }

    #[test]
    fn test_detect_gltf_json() {
        let bytes = b"  { \"asset\": {} }";
        assert_eq!(
            detect_format_from_bytes(bytes).unwrap(),
            DetectedFormat::Gltf
        );
    }

    #[test]
    fn test_detect_unknown() {
        let bytes = b"\x00\x00\x00\x00";
        assert!(detect_format_from_bytes(bytes).is_err());
    }

    #[test]
    fn test_detect_from_extension() {
        assert_eq!(
            detect_format_from_extension(std::path::Path::new("model.wgsc")).unwrap(),
            DetectedFormat::Wgsc
        );
        assert_eq!(
            detect_format_from_extension(std::path::Path::new("model.glb")).unwrap(),
            DetectedFormat::Gltf
        );
        assert_eq!(
            detect_format_from_extension(std::path::Path::new("model.gltf")).unwrap(),
            DetectedFormat::Gltf
        );
        assert!(detect_format_from_extension(std::path::Path::new("model.obj")).is_err());
    }

    #[test]
    fn test_load_sync_wgsc_from_bytes() {
        let bytes = create_test_scene_bytes();
        let result = load_sync(SceneSource::Bytes(bytes), LoadOptions::default()).unwrap();
        assert_eq!(result.format, DetectedFormat::Wgsc);
        assert!(result.camera.is_none());
        assert_eq!(result.scene.meshes.len(), 1);
        assert_eq!(result.scene.nodes.len(), 1);
    }

    #[test]
    fn test_load_progress_initial_state() {
        let progress = LoadProgress::new();
        assert_eq!(progress.phase(), LoadPhase::Pending);
        assert_eq!(progress.progress_pct(), 0);
        assert_eq!(progress.items_total(), 0);
        assert_eq!(progress.items_complete(), 0);
    }

    #[test]
    fn test_load_progress_updates() {
        let progress = LoadProgress::new();
        progress.set_phase(LoadPhase::DecodingTextures, 50);
        progress.set_items(10, 3);

        assert_eq!(progress.phase(), LoadPhase::DecodingTextures);
        assert_eq!(progress.progress_pct(), 50);
        assert_eq!(progress.items_total(), 10);
        assert_eq!(progress.items_complete(), 3);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_async_wgsc() {
        let bytes = create_test_scene_bytes();
        let handle = load_async(SceneSource::Bytes(bytes), LoadOptions::default());

        // Poll until done (in a real app this would be each frame)
        loop {
            if let Some(result) = handle.try_recv() {
                let result = result.unwrap();
                assert_eq!(result.format, DetectedFormat::Wgsc);
                assert_eq!(result.scene.meshes.len(), 1);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_async_unknown_format() {
        let bytes = vec![0u8; 100];
        let handle = load_async(SceneSource::Bytes(bytes), LoadOptions::default());

        loop {
            if let Some(result) = handle.try_recv() {
                assert!(result.is_err());
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }

    #[test]
    fn test_save_sync_to_bytes() {
        let mut scene = create_test_scene();
        let result = save_sync(
            &mut scene,
            SaveDestination::Bytes,
            crate::format::SaveOptions::default(),
        )
        .unwrap();
        let bytes = result.expect("Should return bytes");
        assert!(bytes.starts_with(b"WGSC"));

        // Verify round-trip
        let loaded = load_sync(SceneSource::Bytes(bytes), LoadOptions::default()).unwrap();
        assert_eq!(loaded.scene.meshes.len(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_save_async_to_bytes() {
        let scene = create_test_scene();
        let handle = save_async(
            scene,
            SaveDestination::Bytes,
            crate::format::SaveOptions::default(),
        );

        loop {
            if let Some(result) = handle.try_recv() {
                let bytes = result.unwrap().expect("Should return bytes");
                assert!(bytes.starts_with(b"WGSC"));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }
}
