//! Platform-specific scene/HDR file loading and saving.
//!
//! Both sides expose the same inherent methods on [`crate::App`]
//! (`open_*_dialog`, `save_scene_file_dialog`, `process_pending_io`), so the
//! event loop calls them without any `cfg`. Native uses filesystem paths;
//! web uses the browser file picker + in-memory bytes (download on save).

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;
