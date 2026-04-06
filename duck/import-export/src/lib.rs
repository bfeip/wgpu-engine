//! Scene import/export library.
//!
//! Provides format-agnostic loading and saving of scenes with progress reporting.
//! Supports glTF (.glb/.gltf), USD (.usdc/.usda/.usdz), Duck (.duck), and
//! optionally assimp-based formats.
//!
//! # Submodules
//!
//! - [`mod@format`] — Binary scene serialization (.duck)
//! - [`gltf`] — glTF loading
//! - [`usd`] — USD loading (USDC, USDA, USDZ)
//! - [`assimp`] — Assimp-based loading (feature-gated)
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
//! use duck_engine_import_export::{load_async, LoadOptions, SceneSource};
//!
//! let handle = load_async(SceneSource::Path("model.glb".into()), LoadOptions::default());
//!
//! // In your render loop:
//! // if let Some(result) = handle.try_recv() { ... }
//! ```

#[cfg(feature = "assimp")]
pub mod assimp;
#[cfg(feature = "cad")]
pub mod cad;
pub mod format;
pub mod gltf;
pub mod importer;
#[cfg(feature = "usd")]
pub mod usd;

pub use importer::{default_importers, detect_importer, Importer, PhaseWeights};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use thiserror::Error;

use duck_engine_scene::Camera;
use self::format::FormatError;
use duck_engine_scene::Scene;

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectedFormat {
    Duck,
    Gltf,
    #[cfg(feature = "assimp")]
    Assimp,
    #[cfg(feature = "cad")]
    Step,
    #[cfg(feature = "cad")]
    Iges,
    #[cfg(feature = "usd")]
    Usd,
    /// A custom format provided by a user-defined [`Importer`].
    Other(String),
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
    // NOTE: `std::mem::variant_count` could be used here to get the number of variants
    // but it's nightly right now (March 2026)
    pub const PHASE_COUNT: usize = 8;

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

    #[cfg(feature = "assimp")]
    #[error("Assimp error: {0}")]
    Assimp(String),

    #[cfg(feature = "cad")]
    #[error("CAD error: {0}")]
    Cad(String),

    #[cfg(feature = "usd")]
    #[error("USD error: {0}")]
    Usd(String),

    #[error("Format '{0}' is not supported on this platform")]
    UnsupportedPlatform(String),

    #[error("Unknown file format")]
    UnknownFormat,
}

/// Weights for save operations.
fn save_weights() -> PhaseWeights {
    PhaseWeights::new(&[
        (LoadPhase::Parsing, 60),
        (LoadPhase::Assembling, 40),
    ])
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
            weights: Arc::new(Mutex::new(PhaseWeights::empty())),
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
        let weights = self.weights.lock().unwrap();
        let phase = LoadPhase::from_u8(self.phase.load(Ordering::Relaxed));

        // Terminal states
        if phase == LoadPhase::Complete {
            return 100;
        }
        if phase == LoadPhase::Failed {
            return 0;
        }

        let total_weight = weights.total_weight();
        if total_weight == 0 {
            return 0;
        }

        // Sum weights of all completed phases (discriminants below current)
        let completed_weight = weights.completed_weight(phase);

        // Fractional progress within current phase from items
        let items_total = self.items_total.load(Ordering::Relaxed);
        let items_complete = self.items_complete.load(Ordering::Relaxed);
        let current_weight = weights.get(phase) as f32;
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
    pub fn set_weights(&self, weights: &PhaseWeights) {
        *self.weights.lock().unwrap() = weights.clone();
    }

    /// Enter a new phase. Resets item counters.
    pub fn enter_phase(&self, phase: LoadPhase) {
        self.items_total.store(0, Ordering::Relaxed);
        self.items_complete.store(0, Ordering::Relaxed);
        self.phase.store(phase as u8, Ordering::Relaxed);
    }

    /// Set the total number of items for the current phase.
    pub fn set_item_count(&self, total: u32) {
        self.items_total.store(total, Ordering::Relaxed);
        self.items_complete.store(0, Ordering::Relaxed);
    }

    /// Mark one more item complete in the current phase.
    pub fn complete_item(&self) {
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
// Sync Loading (core logic)
// ============================================================================

/// Load a scene synchronously. Useful for CLI tools and tests.
///
/// Uses the [default importers](default_importers) for format detection.
pub fn load_sync(source: SceneSource, options: LoadOptions) -> LoadResult {
    let importers = default_importers();
    load_sync_with(source, options, &importers)
}

/// Load a scene synchronously using a custom set of importers.
///
/// The importer list order determines detection priority (first match wins).
pub fn load_sync_with(
    source: SceneSource,
    options: LoadOptions,
    importers: &[Box<dyn Importer>],
) -> LoadResult {
    let progress = LoadProgress::new();
    load_sync_with_progress(source, &options, &progress, importers)
}

fn load_sync_with_progress(
    source: SceneSource,
    options: &LoadOptions,
    progress: &LoadProgress,
    importers: &[Box<dyn Importer>],
) -> LoadResult {
    progress.enter_phase(LoadPhase::Reading);

    let (bytes, path_hint) = match source {
        SceneSource::Path(ref path) => {
            let bytes = std::fs::read(path)?;
            (bytes, Some(path.clone()))
        }
        SceneSource::Bytes(b) => (b, None),
    };

    let importer = detect_importer(&bytes, path_hint.as_deref(), importers)?;
    progress.set_weights(&importer.phase_weights());
    importer.load(&bytes, path_hint.as_deref(), options, progress)
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
/// Uses the [default importers](default_importers) for format detection.
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between loading phases.
///
/// Returns a [`LoadHandle`] to poll for completion and progress.
pub fn load_async(source: SceneSource, options: LoadOptions) -> LoadHandle {
    load_async_with(source, options, default_importers())
}

/// Start loading a scene asynchronously using a custom set of importers.
///
/// The importer list order determines detection priority (first match wins).
///
/// On native, spawns a background thread. On WASM, schedules as a microtask
/// with yield points between built-in loading phases; custom importers run
/// synchronously within a microtask.
pub fn load_async_with(
    source: SceneSource,
    options: LoadOptions,
    importers: Vec<Box<dyn Importer>>,
) -> LoadHandle {
    #[cfg(not(target_arch = "wasm32"))]
    {
        spawn_async(LoadProgress::new(), move |progress| {
            load_sync_with_progress(source, &options, progress, &importers)
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let progress = LoadProgress::new();
        let p = progress.clone();
        spawn_async_wasm(progress, async move {
            load_chunked_wasm(source, &options, &p, &importers).await
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
///
/// For built-in formats (Duck, glTF), uses optimized chunked loading with
/// yield points between phases. For custom importers, falls back to the
/// synchronous `Importer::load` within a single microtask.
#[cfg(target_arch = "wasm32")]
async fn load_chunked_wasm(
    source: SceneSource,
    options: &LoadOptions,
    progress: &LoadProgress,
    importers: &[Box<dyn Importer>],
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

    let importer = detect_importer(&bytes, path_hint.as_deref(), importers)?;
    let weights = importer.phase_weights();
    progress.set_weights(&weights);

    // Use optimized chunked paths for built-in formats
    match importer.name() {
        "Duck" => load_duck_chunked(&bytes, progress).await,
        "glTF" => {
            if !bytes.starts_with(b"glTF") {
                // Non-GLB glTF files reference external resources via filesystem
                // paths, which are not accessible on WASM.
                return Err(LoadError::UnsupportedPlatform(
                    "non-GLB glTF (requires filesystem)".into(),
                ));
            }
            load_gltf_chunked(&bytes, options, progress).await
        }
        _ => {
            // Custom/other importers: run synchronously within this microtask
            importer.load(&bytes, path_hint.as_deref(), options, progress)
        }
    }
}

#[cfg(target_arch = "wasm32")]
async fn load_duck_chunked(bytes: &[u8], progress: &LoadProgress) -> LoadResult {
    use self::format::{assemble_duck_scene, decode_duck_texture, parse_duck};

    progress.enter_phase(LoadPhase::Parsing);
    let sections = parse_duck(bytes)?;
    yield_to_event_loop().await;

    progress.enter_phase(LoadPhase::DecodingTextures);
    progress.set_item_count(sections.textures.len() as u32);

    let mut decoded = Vec::with_capacity(sections.textures.len());
    for st in &sections.textures {
        decoded.push(decode_duck_texture(st)?);
        progress.complete_item();
        yield_to_event_loop().await;
    }

    progress.enter_phase(LoadPhase::Assembling);
    yield_to_event_loop().await;

    let scene = assemble_duck_scene(sections, decoded)?;

    progress.enter_phase(LoadPhase::Complete);
    Ok(SceneLoadResult {
        scene,
        camera: None,
        format: DetectedFormat::Duck,
    })
}

#[cfg(target_arch = "wasm32")]
async fn load_gltf_chunked(
    bytes: &[u8],
    options: &LoadOptions,
    progress: &LoadProgress,
) -> LoadResult {
    use self::gltf::{build_gltf_scene, load_gltf_assets, parse_gltf};

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
    options: self::format::SaveOptions,
) -> SaveResult {
    let progress = LoadProgress::new();
    save_sync_with_progress(scene, dest, &options, &progress)
}

fn save_sync_with_progress(
    scene: &Scene,
    dest: SaveDestination,
    options: &self::format::SaveOptions,
    progress: &LoadProgress,
) -> SaveResult {
    progress.set_weights(&save_weights());
    progress.enter_phase(LoadPhase::Parsing); // "Parsing" = serializing in export context
    let bytes = format::to_bytes_with_options(scene, options)
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
    options: self::format::SaveOptions,
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
    use duck_engine_scene::{Material, Mesh, PrimitiveType};

    fn create_test_scene() -> Scene {
        let mut scene = Scene::new();
        let mesh = Mesh::cube(1.0, PrimitiveType::TriangleList);
        let mesh_id = scene.add_mesh(mesh);
        let mat_id = scene.add_material(Material::new());
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some("TestNode".into()),
                duck_engine_scene::common::Transform::IDENTITY,
            )
            .unwrap();
        scene
    }

    fn create_test_scene_bytes() -> Vec<u8> {
        format::to_bytes(&create_test_scene()).unwrap()
    }

    #[test]
    fn test_detect_duck() {
        let importers = default_importers();
        let bytes = b"DUCK\x01\x00rest of file";
        let imp = detect_importer(bytes, None, &importers).unwrap();
        assert_eq!(imp.name(), "Duck");
    }

    #[test]
    fn test_detect_glb() {
        let importers = default_importers();
        let bytes = b"glTF\x02\x00\x00\x00rest";
        let imp = detect_importer(bytes, None, &importers).unwrap();
        assert_eq!(imp.name(), "glTF");
    }

    #[test]
    fn test_detect_gltf_json() {
        let importers = default_importers();
        let bytes = b"  { \"asset\": {} }";
        let imp = detect_importer(bytes, None, &importers).unwrap();
        assert_eq!(imp.name(), "glTF");
    }

    #[test]
    fn test_detect_unknown() {
        let importers = default_importers();
        let bytes = b"\x00\x00\x00\x00";
        assert!(detect_importer(bytes, None, &importers).is_err());
    }

    #[test]
    fn test_detect_from_extension() {
        let importers = default_importers();
        let unknown = b"\x00\x00\x00\x00";

        let imp = detect_importer(unknown, Some(std::path::Path::new("model.duck")), &importers).unwrap();
        assert_eq!(imp.name(), "Duck");

        let imp = detect_importer(unknown, Some(std::path::Path::new("model.glb")), &importers).unwrap();
        assert_eq!(imp.name(), "glTF");

        let imp = detect_importer(unknown, Some(std::path::Path::new("model.gltf")), &importers).unwrap();
        assert_eq!(imp.name(), "glTF");

        #[cfg(feature = "assimp")]
        {
            let imp = detect_importer(unknown, Some(std::path::Path::new("model.obj")), &importers).unwrap();
            assert_eq!(imp.name(), "Assimp");
        }
        #[cfg(not(feature = "assimp"))]
        assert!(detect_importer(unknown, Some(std::path::Path::new("model.obj")), &importers).is_err());
    }

    #[test]
    fn test_custom_importer() {
        use std::path::Path;

        struct TestImporter;
        impl Importer for TestImporter {
            fn name(&self) -> &str { "Test" }
            fn detect_from_bytes(&self, bytes: &[u8]) -> bool {
                bytes.starts_with(b"TEST")
            }
            fn detect_from_extension(&self, ext: &str) -> bool {
                ext == "test"
            }
            fn phase_weights(&self) -> PhaseWeights {
                PhaseWeights::new(&[
                    (LoadPhase::Reading, 50),
                    (LoadPhase::Assembling, 50),
                ])
            }
            fn load(
                &self,
                _bytes: &[u8],
                _path_hint: Option<&Path>,
                _options: &LoadOptions,
                progress: &LoadProgress,
            ) -> Result<SceneLoadResult, LoadError> {
                progress.enter_phase(LoadPhase::Assembling);
                progress.enter_phase(LoadPhase::Complete);
                Ok(SceneLoadResult {
                    scene: Scene::new(),
                    camera: None,
                    format: DetectedFormat::Other("Test".into()),
                })
            }
        }

        let importers: Vec<Box<dyn Importer>> = vec![Box::new(TestImporter)];

        // Detect from bytes
        let imp = detect_importer(b"TEST data", None, &importers).unwrap();
        assert_eq!(imp.name(), "Test");

        // Detect from extension
        let imp = detect_importer(b"\x00\x00\x00\x00", Some(Path::new("file.test")), &importers).unwrap();
        assert_eq!(imp.name(), "Test");

        // Load via load_sync_with
        let result = load_sync_with(
            SceneSource::Bytes(b"TEST data".to_vec()),
            LoadOptions::default(),
            &importers,
        ).unwrap();
        assert_eq!(result.format, DetectedFormat::Other("Test".into()));
    }

    #[test]
    fn test_load_sync_duck_from_bytes() {
        let bytes = create_test_scene_bytes();
        let result = load_sync(SceneSource::Bytes(bytes), LoadOptions::default()).unwrap();
        assert_eq!(result.format, DetectedFormat::Duck);
        assert!(result.camera.is_none());
        assert_eq!(result.scene.mesh_count(), 1);
        assert_eq!(result.scene.node_count(), 1);
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
        // Simulate a Duck load: Reading(10) + Parsing(10) + DecodingTextures(60) + Assembling(20)
        let duck_weights = PhaseWeights::new(&[
            (LoadPhase::Reading, 10),
            (LoadPhase::Parsing, 10),
            (LoadPhase::DecodingTextures, 60),
            (LoadPhase::Assembling, 20),
        ]);
        progress.set_weights(&duck_weights);

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
    fn test_load_async_duck() {
        let bytes = create_test_scene_bytes();
        let handle = load_async(SceneSource::Bytes(bytes), LoadOptions::default());

        // Poll until done (in a real app this would be each frame)
        loop {
            if let Some(result) = handle.try_get() {
                let result = result.unwrap();
                assert_eq!(result.format, DetectedFormat::Duck);
                assert_eq!(result.scene.mesh_count(), 1);
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
            format::SaveOptions::default(),
        )
        .unwrap();
        let bytes = result.expect("Should return bytes");
        assert!(bytes.starts_with(b"DUCK"));

        // Verify round-trip
        let loaded = load_sync(SceneSource::Bytes(bytes), LoadOptions::default()).unwrap();
        assert_eq!(loaded.scene.mesh_count(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_save_async_to_bytes() {
        let scene = create_test_scene();
        let handle = save_async(
            &scene,
            SaveDestination::Bytes,
            format::SaveOptions::default(),
        );

        loop {
            if let Some(result) = handle.try_get() {
                let bytes = result.unwrap().expect("Should return bytes");
                assert!(bytes.starts_with(b"DUCK"));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(handle.is_done());
    }
}
