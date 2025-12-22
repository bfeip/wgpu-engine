use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use wgpu_engine::egui_support::EguiViewerApp;
#[cfg(not(target_arch = "wasm32"))]
use wgpu_engine::input::{ElementState, Key, NamedKey};
#[cfg(target_arch = "wasm32")]
use wgpu_engine::input::{ElementState, Key};
use wgpu_engine::scene::Scene;

use crate::ui;

/// Debug actions triggered by key presses
enum DebugAction {
    ToggleOrtho,
    #[cfg(not(target_arch = "wasm32"))]
    Exit,
}

/// Application state for the winit event loop with egui integration
pub struct App<'a> {
    viewer_app: Option<EguiViewerApp<'a>>,

    /// Pending file path to load (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pending_gltf_path: Option<std::path::PathBuf>,

    /// Pending file data to load (web only)
    #[cfg(target_arch = "wasm32")]
    pending_gltf_data: Option<Vec<u8>>,
}

impl<'a> App<'a> {
    pub fn new() -> Self {
        Self {
            viewer_app: None,
            #[cfg(not(target_arch = "wasm32"))]
            pending_gltf_path: None,
            #[cfg(target_arch = "wasm32")]
            pending_gltf_data: None,
        }
    }

    /// Handle the RedrawRequested event - build UI and render the frame
    fn handle_redraw_requested(&mut self) {
        // Process any pending glTF file load
        self.process_pending_load();

        // Build egui UI and render
        let mut ui_actions = ui::UiActions::default();
        {
            let viewer_app = self.viewer_app.as_mut().unwrap();
            if let Err(e) = viewer_app.render(|ctx, viewer| {
                ui_actions = ui::build(ctx, viewer);
            }) {
                log::error!("Render error: {}", e);
            }
        }

        // Handle UI actions (after releasing viewer_app borrow)
        if ui_actions.load_file {
            self.open_file_dialog();
        }
        if ui_actions.clear_scene {
            self.clear_scene();
        }

        // Request next frame
        self.viewer_app.as_ref().unwrap().request_redraw();
    }

    /// Process any pending glTF load
    fn process_pending_load(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(path) = self.pending_gltf_path.take() {
            self.load_gltf_from_path(&path);
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(data) = self.pending_gltf_data.take() {
            self.load_gltf_from_bytes(&data);
        }
    }

    /// Open platform-specific file dialog
    fn open_file_dialog(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = rfd::FileDialog::new()
                .add_filter("glTF", &["gltf", "glb"])
                .pick_file();

            if let Some(path) = file {
                self.pending_gltf_path = Some(path);
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            self.trigger_web_file_input();
        }
    }

    /// Load glTF from file path (native only)
    #[cfg(not(target_arch = "wasm32"))]
    fn load_gltf_from_path(&mut self, path: &std::path::Path) {
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let aspect = {
            let size = viewer.size();
            size.0 as f32 / size.1 as f32
        };

        let path_str = path.display().to_string();
        match wgpu_engine::load_gltf_scene_from_path(path, aspect) {
            Ok(result) => {
                viewer.set_scene(result.scene);
                if let Some(camera) = result.camera {
                    viewer.set_camera(camera);
                }
                log::info!("Loaded glTF: {}", path_str);
            }
            Err(e) => {
                log::error!("Failed to load glTF {}: {}", path_str, e);
            }
        }
    }

    /// Load glTF from bytes
    fn load_gltf_from_bytes(&mut self, data: &[u8]) {
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let aspect = {
            let size = viewer.size();
            size.0 as f32 / size.1 as f32
        };

        match wgpu_engine::load_gltf_scene_from_slice(data, aspect) {
            Ok(result) => {
                viewer.set_scene(result.scene);
                if let Some(camera) = result.camera {
                    viewer.set_camera(camera);
                }
                log::info!("Loaded glTF from bytes");
            }
            Err(e) => {
                log::error!("Failed to load glTF: {}", e);
            }
        }
    }

    /// Trigger the web file input element
    #[cfg(target_arch = "wasm32")]
    fn trigger_web_file_input(&self) {
        use wasm_bindgen::JsCast;
        use web_sys::HtmlInputElement;

        let window = web_sys::window().expect("no window");
        let document = window.document().expect("no document");

        // Check if input already exists
        if document.get_element_by_id("gltf-file-input").is_none() {
            self.setup_web_file_input(&document);
        }

        // Trigger the file input
        if let Some(element) = document.get_element_by_id("gltf-file-input") {
            let input: HtmlInputElement = element.dyn_into().unwrap();
            input.click();
        }
    }

    /// Setup the hidden file input element for web
    #[cfg(target_arch = "wasm32")]
    fn setup_web_file_input(&self, document: &web_sys::Document) {
        use wasm_bindgen::JsCast;
        use web_sys::HtmlInputElement;

        let input: HtmlInputElement = document
            .create_element("input")
            .unwrap()
            .dyn_into()
            .unwrap();
        input.set_type("file");
        input.set_accept(".gltf,.glb");
        input.set_id("gltf-file-input");
        let _ = input.style().set_property("display", "none");
        document.body().unwrap().append_child(&input).unwrap();

        // Note: File reading is handled via the change event, which we set up
        // but the actual reading would need to use a callback mechanism.
        // For simplicity, we log that the user needs to implement async file reading.
        log::info!("Web file input set up. File reading requires async handling.");
    }

    /// Clear the scene
    fn clear_scene(&mut self) {
        let viewer_app = self.viewer_app.as_mut().unwrap();
        viewer_app.viewer_mut().set_scene(Scene::new());
        log::info!("Scene cleared");
    }

    /// Handle debug key actions
    fn handle_debug_key_action(&mut self, action: DebugAction, event_loop: &ActiveEventLoop) {
        match action {
            DebugAction::ToggleOrtho => self.toggle_ortho(),
            #[cfg(not(target_arch = "wasm32"))]
            DebugAction::Exit => event_loop.exit(),
        }
        let _ = event_loop; // suppress unused warning on web
    }

    /// Check if event is a debug key press and return the action
    fn get_debug_key_action(event: &wgpu_engine::event::Event) -> Option<DebugAction> {
        let wgpu_engine::event::Event::KeyboardInput { event: key_event, .. } = event else {
            return None;
        };

        if key_event.state != ElementState::Pressed || key_event.repeat {
            return None;
        }

        match &key_event.logical_key {
            Key::Character('o') => Some(DebugAction::ToggleOrtho),
            #[cfg(not(target_arch = "wasm32"))]
            Key::Named(NamedKey::Escape) => Some(DebugAction::Exit),
            _ => None,
        }
    }

    /// Toggle between perspective and orthographic projection
    fn toggle_ortho(&mut self) {
        let viewer_app = self.viewer_app.as_mut().unwrap();
        let camera = viewer_app.viewer_mut().camera_mut();
        camera.ortho = !camera.ortho;
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer_app.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title("glTF Viewer")
            .with_min_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        #[cfg(not(target_arch = "wasm32"))]
        {
            let viewer_app = pollster::block_on(EguiViewerApp::with_window_attrs(
                event_loop,
                window_attrs,
            ));
            viewer_app.request_redraw();
            self.viewer_app = Some(viewer_app);
        }

        #[cfg(target_arch = "wasm32")]
        {
            use std::sync::Arc;
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            // Get the canvas element from the page
            let window = web_sys::window().expect("no window");
            let document = window.document().expect("no document");
            let canvas = document
                .get_element_by_id("gltf-viewer-canvas")
                .expect("no canvas with id 'gltf-viewer-canvas'");
            let canvas: web_sys::HtmlCanvasElement = canvas
                .dyn_into()
                .expect("element is not a canvas");

            let window_attrs = window_attrs.with_canvas(Some(canvas));

            // Spawn async initialization
            let event_loop_window = Arc::new(
                event_loop
                    .create_window(window_attrs)
                    .expect("Failed to create window"),
            );

            // For web, we need to handle async initialization differently.
            // The EguiViewerApp requires an async context for WGPU initialization.
            // We'll use wasm_bindgen_futures to spawn the async task.
            let _window = event_loop_window; // Keep window alive

            // Store a temporary placeholder - the actual viewer will be created async
            // This is a simplified approach; a production app would use channels or
            // shared state to communicate the created viewer back.
            log::info!("Web initialization started - WGPU async init required");

            // Note: Full web implementation would use EventLoopProxy with user events
            // to communicate the created viewer back to the App after async init.
            // For now, this is a placeholder showing the structure.
            wasm_bindgen_futures::spawn_local(async move {
                log::info!("Async WGPU initialization starting...");
                // A full implementation would:
                // 1. Create the viewer asynchronously
                // 2. Send it back via EventLoopProxy user events
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(viewer_app) = self.viewer_app.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.handle_redraw_requested();
            }
            _ => {
                viewer_app.handle_window_event(&event);

                if let Some(app_event) = wgpu_engine::winit_support::convert_window_event(event) {
                    if let Some(action) = Self::get_debug_key_action(&app_event) {
                        self.handle_debug_key_action(action, event_loop);
                    }
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(viewer_app) = &mut self.viewer_app {
            viewer_app.handle_device_event(&event);
        }
    }
}
