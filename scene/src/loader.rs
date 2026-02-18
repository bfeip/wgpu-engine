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
use std::sync::{Arc, Mutex};

use thiserror::Error;

use crate::camera::Camera;
use crate::format::FormatError;
use crate::Scene;

// ============================================================================
// Type Aliases
// ============================================================================

/// The result type produced by a completed load operation.
type LoadResult = Result<SceneLoadResult, LoadError>;

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

/// Phase weight table: maps each `LoadPhase` discriminant to a relative weight.
///
/// Indexed by `LoadPhase as usize`. The sum of weights determines how overall
/// percentage is distributed across phases. Only phases with non-zero weights
/// contribute to progress.
type PhaseWeights = Vec<u8>;

/// Weights for WGSC loading: Reading(10) + Parsing(10) + DecodingTextures(60) + Assembling(20).
const WGSC_WEIGHTS: [u8; 8] = weights(&[
    (LoadPhase::Reading, 10),
    (LoadPhase::Parsing, 10),
    (LoadPhase::DecodingTextures, 60),
    (LoadPhase::Assembling, 20),
]);

/// Weights for glTF loading: Reading(10) + Parsing(10) + DecodingTextures(50) + Assembling(30).
const GLTF_WEIGHTS: [u8; 8] = weights(&[
    (LoadPhase::Reading, 10),
    (LoadPhase::Parsing, 10),
    (LoadPhase::DecodingTextures, 50),
    (LoadPhase::Assembling, 30),
]);

/// Weights for save operations: Parsing/serializing(60) + Assembling/writing(40).
const SAVE_WEIGHTS: [u8; 8] = weights(&[
    (LoadPhase::Parsing, 60),
    (LoadPhase::Assembling, 40),
]);

/// Build a weight table from (phase, weight) pairs at compile time.
const fn weights(pairs: &[(LoadPhase, u8)]) -> [u8; 8] {
    let mut w = [0u8; 8];
    let mut i = 0;
    while i < pairs.len() {
        w[pairs[i].0 as u8 as usize] = pairs[i].1;
        i += 1;
    }
    w
}

/// Shared progress state, readable from any thread.
///
/// Overall progress percentage is derived automatically from the current phase,
/// item progress within that phase, and a weight table. Callers just call
/// [`enter_phase`](Self::enter_phase) and [`complete_item`](Self::complete_item);
/// no manual percentage math is needed.
#[derive(Clone)]
pub struct LoadProgress {
    phase: Arc<AtomicU8>,
    items_total: Arc<AtomicU32>,
    items_complete: Arc<AtomicU32>,
    weights: Arc<Mutex<PhaseWeights>>,
}

impl LoadProgress {
    fn new() -> Self {
        Self {
            phase: Arc::new(AtomicU8::new(LoadPhase::Pending as u8)),
            items_total: Arc::new(AtomicU32::new(0)),
            items_complete: Arc::new(AtomicU32::new(0)),
            weights: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Current loading phase.
    pub fn phase(&self) -> LoadPhase {
        LoadPhase::from_u8(self.phase.load(Ordering::Relaxed))
    }

    /// Overall progress percentage (0–100), derived from the weight table.
    ///
    /// Completed phases contribute their full weight. The current phase
    /// contributes proportionally to item progress (or 0% if no items are set,
    /// since monolithic phases jump from 0% to 100% when the next phase starts).
    pub fn progress_pct(&self) -> u8 {
        let w = self.weights.lock().unwrap();
        let phase = self.phase.load(Ordering::Relaxed) as usize;

        // Terminal states
        if phase == LoadPhase::Complete as usize {
            return 100;
        }
        if phase == LoadPhase::Failed as usize {
            return 0;
        }

        let total_weight: u16 = w.iter().map(|&x| x as u16).sum();
        if total_weight == 0 {
            return 0;
        }

        // Sum weights of all completed phases (discriminants below current)
        let completed_weight: u16 = w.iter().take(phase).map(|&x| x as u16).sum();

        // Fractional progress within current phase from items
        let items_total = self.items_total.load(Ordering::Relaxed);
        let items_complete = self.items_complete.load(Ordering::Relaxed);
        let current_weight = w.get(phase).copied().unwrap_or(0) as f32;
        let phase_fraction = if items_total > 0 {
            items_complete as f32 / items_total as f32
        } else {
            0.0
        };

        let pct = (completed_weight as f32 + current_weight * phase_fraction) * 100.0
            / total_weight as f32;
        (pct as u8).min(100)
    }

    /// Total number of items in the current phase (e.g., textures to decode).
    pub fn items_total(&self) -> u32 {
        self.items_total.load(Ordering::Relaxed)
    }

    /// Number of items completed in the current phase.
    pub fn items_complete(&self) -> u32 {
        self.items_complete.load(Ordering::Relaxed)
    }

    /// Bind a weight table for this operation.
    fn set_weights(&self, weights: &[u8]) {
        *self.weights.lock().unwrap() = weights.to_vec();
    }

    /// Enter a new phase. Resets item counters.
    fn enter_phase(&self, phase: LoadPhase) {
        self.items_total.store(0, Ordering::Relaxed);
        self.items_complete.store(0, Ordering::Relaxed);
        self.phase.store(phase as u8, Ordering::Relaxed);
    }

    /// Set the total number of items for the current phase.
    fn set_item_count(&self, total: u32) {
        self.items_total.store(total, Ordering::Relaxed);
        self.items_complete.store(0, Ordering::Relaxed);
    }

    /// Mark one more item complete in the current phase.
    fn complete_item(&self) {
        self.items_complete.fetch_add(1, Ordering::Relaxed);
    }
}

/// Handle to an async operation in progress.
///
/// Poll each frame with [`try_get`](AsyncHandle::try_get) to check for
/// completion, and read [`progress`](AsyncHandle::progress) to display a
/// progress bar.
pub struct AsyncHandle<T> {
    progress: LoadProgress,
    done: Arc<AtomicBool>,
    #[cfg(not(target_arch = "wasm32"))]
    receiver: std::sync::mpsc::Receiver<T>,
    #[cfg(target_arch = "wasm32")]
    result: std::rc::Rc<std::cell::RefCell<Option<T>>>,
}

impl<T> AsyncHandle<T> {
    /// Returns `Some(result)` if the operation has completed, `None` if still in progress.
    ///
    /// Consumes the result on first successful call. Subsequent calls return `None`.
    pub fn try_get(&self) -> Option<T> {
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

    /// Returns true if the operation has completed (success or failure).
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

/// Handle to a loading operation in progress.
pub type LoadHandle = AsyncHandle<LoadResult>;

/// Handle to a save operation in progress.
pub type SaveHandle = AsyncHandle<SaveResult>;

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
    progress.enter_phase(LoadPhase::Reading);

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    let format = detect_format_from_bytes(&bytes).or_else(|_| {
        path_hint
            .as_ref()
            .map(|p| detect_format_from_extension(p))
            .unwrap_or(Err(LoadError::UnknownFormat))
    })?;

    match format {
        DetectedFormat::Wgsc => {
            progress.set_weights(&WGSC_WEIGHTS);
            load_wgsc_phased(&bytes, progress)
        }
        DetectedFormat::Gltf => {
            progress.set_weights(&GLTF_WEIGHTS);
            load_gltf_phased(&bytes, options, progress)
        }
    }
}

fn load_wgsc_phased(bytes: &[u8], progress: &LoadProgress) -> LoadResult {
    use crate::format::{assemble_wgsc_scene, decode_wgsc_texture, parse_wgsc};

    progress.enter_phase(LoadPhase::Parsing);
    let sections = parse_wgsc(bytes)?;

    progress.enter_phase(LoadPhase::DecodingTextures);
    progress.set_item_count(sections.textures.len() as u32);

    let mut decoded = Vec::with_capacity(sections.textures.len());
    for st in &sections.textures {
        decoded.push(decode_wgsc_texture(st)?);
        progress.complete_item();
    }

    progress.enter_phase(LoadPhase::Assembling);
    let scene = assemble_wgsc_scene(sections, decoded)?;

    progress.enter_phase(LoadPhase::Complete);
    Ok(SceneLoadResult {
        scene,
        camera: None,
        format: DetectedFormat::Wgsc,
    })
}

fn load_gltf_phased(bytes: &[u8], options: &LoadOptions, progress: &LoadProgress) -> LoadResult {
    use crate::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf};

    progress.enter_phase(LoadPhase::Parsing);
    let parsed = parse_gltf(bytes).map_err(|e| LoadError::Gltf(e.to_string()))?;

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

// ============================================================================
// Async Helpers
// ============================================================================

/// Spawn a sync closure on a background thread, returning a handle to poll for the result.
///
/// Sets the progress phase to [`LoadPhase::Failed`] automatically if the closure
/// returns an `Err`.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_async<T, E>(
    progress: LoadProgress,
    f: impl FnOnce(&LoadProgress) -> Result<T, E> + Send + 'static,
) -> AsyncHandle<Result<T, E>>
where
    T: Send + 'static,
    E: Send + 'static,
{
    let progress_clone = progress.clone();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let result = f(&progress_clone);
        if result.is_err() {
            progress_clone.enter_phase(LoadPhase::Failed);
        }
        done_clone.store(true, Ordering::Release);
        let _ = tx.send(result);
    });

    AsyncHandle {
        progress,
        done,
        receiver: rx,
    }
}

/// Spawn a future as a WASM microtask, returning a handle to poll for the result.
///
/// Sets the progress phase to [`LoadPhase::Failed`] automatically if the future
/// resolves to an `Err`.
#[cfg(target_arch = "wasm32")]
fn spawn_async_wasm<T, E>(
    progress: LoadProgress,
    fut: impl std::future::Future<Output = Result<T, E>> + 'static,
) -> AsyncHandle<Result<T, E>>
where
    T: 'static,
    E: 'static,
{
    use std::cell::RefCell;
    use std::rc::Rc;

    let progress_clone = progress.clone();
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let result_cell: Rc<RefCell<Option<Result<T, E>>>> = Rc::new(RefCell::new(None));
    let result_clone = Rc::clone(&result_cell);

    wasm_bindgen_futures::spawn_local(async move {
        let result = fut.await;
        if result.is_err() {
            progress_clone.enter_phase(LoadPhase::Failed);
        }
        done_clone.store(true, Ordering::Release);
        *result_clone.borrow_mut() = Some(result);
    });

    AsyncHandle {
        progress,
        done,
        result: result_cell,
    }
}

// ============================================================================
// Async Loading
// ============================================================================

/// Start loading a scene asynchronously.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between loading phases.
///
/// Returns a [`LoadHandle`] to poll for completion and progress.
pub fn load_async(source: SceneSource, options: LoadOptions) -> LoadHandle {
    #[cfg(not(target_arch = "wasm32"))]
    {
        spawn_async(LoadProgress::new(), move |progress| {
            load_sync_with_progress(source, &options, progress)
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let progress = LoadProgress::new();
        let p = progress.clone();
        spawn_async_wasm(progress, async move {
            load_chunked_wasm(source, &options, &p).await
        })
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
    progress.enter_phase(LoadPhase::Reading);

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            // Note: std::fs::read doesn't work on WASM. Callers should use
            // SceneSource::Bytes. This arm exists for API uniformity.
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    yield_to_event_loop().await;

    let format = detect_format_from_bytes(&bytes).or_else(|_| {
        path_hint
            .as_ref()
            .map(|p| detect_format_from_extension(p))
            .unwrap_or(Err(LoadError::UnknownFormat))
    })?;

    match format {
        DetectedFormat::Wgsc => {
            progress.set_weights(&WGSC_WEIGHTS);
            load_wgsc_chunked(&bytes, progress).await
        }
        DetectedFormat::Gltf => {
            progress.set_weights(&GLTF_WEIGHTS);
            load_gltf_chunked(&bytes, options, progress).await
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn load_wgsc_chunked(bytes: &[u8], progress: &LoadProgress) -> LoadResult {
    use crate::format::{assemble_wgsc_scene, decode_wgsc_texture, parse_wgsc};

    progress.enter_phase(LoadPhase::Parsing);
    let sections = parse_wgsc(bytes)?;
    yield_to_event_loop().await;

    progress.enter_phase(LoadPhase::DecodingTextures);
    progress.set_item_count(sections.textures.len() as u32);

    let mut decoded = Vec::with_capacity(sections.textures.len());
    for st in &sections.textures {
        decoded.push(decode_wgsc_texture(st)?);
        progress.complete_item();
        yield_to_event_loop().await;
    }

    progress.enter_phase(LoadPhase::Assembling);
    yield_to_event_loop().await;

    let scene = assemble_wgsc_scene(sections, decoded)?;

    progress.enter_phase(LoadPhase::Complete);
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

    progress.enter_phase(LoadPhase::Parsing);
    let parsed = parse_gltf(bytes).map_err(|e| LoadError::Gltf(e.to_string()))?;
    yield_to_event_loop().await;

    progress.enter_phase(LoadPhase::DecodingTextures);
    let mut scene = Scene::new();
    let (_material_map, mesh_map) =
        load_gltf_assets(&parsed, &mut scene).map_err(|e| LoadError::Gltf(e.to_string()))?;
    yield_to_event_loop().await;

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

// ============================================================================
// Sync Export
// ============================================================================

/// Save a scene synchronously. Useful for CLI tools and tests.
pub fn save_sync(
    scene: &Scene,
    dest: SaveDestination,
    options: crate::format::SaveOptions,
) -> SaveResult {
    let progress = LoadProgress::new();
    save_sync_with_progress(scene, dest, &options, &progress)
}

fn save_sync_with_progress(
    scene: &Scene,
    dest: SaveDestination,
    options: &crate::format::SaveOptions,
    progress: &LoadProgress,
) -> SaveResult {
    progress.set_weights(&SAVE_WEIGHTS);
    progress.enter_phase(LoadPhase::Parsing); // "Parsing" = serializing in export context
    let bytes = scene
        .to_bytes_with_options(options)
        .map_err(LoadError::Format)?;

    progress.enter_phase(LoadPhase::Assembling);
    match dest {
        SaveDestination::Path(path) => {
            std::fs::write(path, &bytes)?;
            progress.enter_phase(LoadPhase::Complete);
            Ok(None)
        }
        SaveDestination::Bytes => {
            progress.enter_phase(LoadPhase::Complete);
            Ok(Some(bytes))
        }
    }
}

// ============================================================================
// Async Export
// ============================================================================

/// Start saving a scene asynchronously.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask.
///
/// The scene is cloned internally for the background task, so the caller
/// retains full ownership. Note: this is currently a deep clone. A future
/// optimization can Arc-wrap large data (textures, meshes) for cheap clones.
pub fn save_async(
    scene: &Scene,
    dest: SaveDestination,
    options: crate::format::SaveOptions,
) -> SaveHandle {
    let scene = scene.clone();
    #[cfg(not(target_arch = "wasm32"))]
    {
        spawn_async(LoadProgress::new(), move |progress| {
            save_sync_with_progress(&scene, dest, &options, progress)
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let progress = LoadProgress::new();
        let p = progress.clone();
        spawn_async_wasm(progress, async move {
            save_sync_with_progress(&scene, dest, &options, &p)
        })
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
        // Simulate a WGSC load: Reading(10) + Parsing(10) + DecodingTextures(60) + Assembling(20)
        progress.set_weights(&WGSC_WEIGHTS);

        progress.enter_phase(LoadPhase::Reading);
        assert_eq!(progress.phase(), LoadPhase::Reading);
        assert_eq!(progress.progress_pct(), 0);

        progress.enter_phase(LoadPhase::Parsing);
        assert_eq!(progress.phase(), LoadPhase::Parsing);
        // Reading(10) complete out of total 100 → 10%
        assert_eq!(progress.progress_pct(), 10);

        progress.enter_phase(LoadPhase::DecodingTextures);
        progress.set_item_count(10);
        assert_eq!(progress.items_total(), 10);
        assert_eq!(progress.items_complete(), 0);
        // Reading(10) + Parsing(10) complete → 20%
        assert_eq!(progress.progress_pct(), 20);

        // Complete 3 of 10 items → 20% + 60% * 3/10 = 38%
        for _ in 0..3 {
            progress.complete_item();
        }
        assert_eq!(progress.items_complete(), 3);
        assert_eq!(progress.progress_pct(), 38);

        progress.enter_phase(LoadPhase::Assembling);
        // Reading(10) + Parsing(10) + DecodingTextures(60) complete → 80%
        assert_eq!(progress.progress_pct(), 80);

        progress.enter_phase(LoadPhase::Complete);
        assert_eq!(progress.progress_pct(), 100);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_async_wgsc() {
        let bytes = create_test_scene_bytes();
        let handle = load_async(SceneSource::Bytes(bytes), LoadOptions::default());

        // Poll until done (in a real app this would be each frame)
        loop {
            if let Some(result) = handle.try_get() {
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
            if let Some(result) = handle.try_get() {
                assert!(result.is_err());
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }

    #[test]
    fn test_save_sync_to_bytes() {
        let scene = create_test_scene();
        let result = save_sync(
            &scene,
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
            &scene,
            SaveDestination::Bytes,
            crate::format::SaveOptions::default(),
        );

        loop {
            if let Some(result) = handle.try_get() {
                let bytes = result.unwrap().expect("Should return bytes");
                assert!(bytes.starts_with(b"WGSC"));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }
}
