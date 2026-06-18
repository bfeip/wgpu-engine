//! Web file I/O: `rfd` async browser dialogs + in-memory bytes.
//!
//! Dialogs run on the JS event loop via `spawn_local`; the resulting bytes are
//! stashed in the `App`'s shared buffers and consumed on the next frame.
//! Saving serializes the scene and triggers a browser download.

use std::sync::{Arc, Mutex};

use duck_engine_viewer::import_export;

use crate::App;

impl App<'_> {
    /// Consume any bytes delivered by the browser file dialogs.
    pub(crate) fn process_pending_io(&mut self) {
        if self.state.is_none() {
            return;
        }
        let scene_bytes = self.pending_scene_bytes.borrow_mut().take();
        if let Some(bytes) = scene_bytes {
            self.load_scene_bytes(bytes);
        }
        let hdr_bytes = self.pending_hdr_bytes.borrow_mut().take();
        if let Some(bytes) = hdr_bytes {
            self.load_hdr_bytes(bytes);
        }
    }

    pub(crate) fn open_hdr_file_dialog(&mut self) {
        let sink = self.pending_hdr_bytes.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("HDR", &["hdr"])
                .pick_file()
                .await
            {
                *sink.borrow_mut() = Some(file.read().await);
            }
        });
    }

    fn load_hdr_bytes(&mut self, bytes: Vec<u8>) {
        let Some(state) = self.state.as_mut() else { return };
        let scene_arc = state.viewer.scene();
        let mut scene = scene_arc.lock().unwrap();
        let env_id = scene.add_environment_map_from_hdr_data(bytes);
        scene.set_active_environment_map(Some(env_id));
        log::info!("Loaded HDR environment");
    }

    pub(crate) fn open_scene_file_dialog(&mut self) {
        let sink = self.pending_scene_bytes.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("3D Scenes", &["glb", "gltf", "duck"])
                .pick_file()
                .await
            {
                *sink.borrow_mut() = Some(file.read().await);
            }
        });
    }

    fn load_scene_bytes(&mut self, bytes: Vec<u8>) {
        use import_export::{load_sync, SceneSource, LoadOptions};
        let Some(state) = self.state.as_mut() else { return };
        match load_sync(SceneSource::Bytes(bytes), LoadOptions::default()) {
            Ok(result) => {
                let bounds = result.scene.bounding().bounds;
                state.viewer.set_scene(Arc::new(Mutex::new(result.scene)));
                if let Some(camera) = result.camera {
                    state.viewer.set_camera(camera);
                } else if let Some(bounds) = bounds {
                    state.viewer.with_camera_mut(|c| c.fit_to_bounds(&bounds));
                }
                log::info!("Loaded scene");
            }
            Err(e) => log::error!("Failed to load scene: {}", e),
        }
    }

    /// Serialize the scene and trigger a browser download via the save dialog.
    pub(crate) fn save_scene_file_dialog(&mut self) {
        use import_export::format::to_bytes;
        let Some(state) = self.state.as_ref() else { return };
        let scene_arc = state.viewer.scene();
        let bytes = {
            let scene = scene_arc.lock().unwrap();
            match to_bytes(&scene) {
                Ok(bytes) => bytes,
                Err(e) => {
                    log::error!("Failed to serialize scene: {}", e);
                    return;
                }
            }
        };
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = rfd::AsyncFileDialog::new()
                .set_file_name("scene.duck")
                .save_file()
                .await
            {
                if let Err(e) = file.write(&bytes).await {
                    log::error!("Failed to save scene: {:?}", e);
                } else {
                    log::info!("Saved scene");
                }
            }
        });
    }
}
