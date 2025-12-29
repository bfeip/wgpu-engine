use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use wgpu_engine::egui_support::EguiViewerApp;
#[cfg(not(target_arch = "wasm32"))]
use wgpu_engine::input::{ElementState, Key, NamedKey};
#[cfg(target_arch = "wasm32")]
use wgpu_engine::input::{ElementState, Key};
use wgpu_engine::scene::Scene;

use crate::ui;

/// Events sent from async tasks back to the main event loop
#[allow(dead_code)] // ViewerInitFailed reserved for future error handling
pub enum AppEvent {
    /// WGPU initialization completed successfully
    ViewerReady(Box<EguiViewerApp<'static>>),
    /// WGPU initialization failed (reserved for future use)
    ViewerInitFailed(String),
    /// File data loaded from web file input
    #[cfg(target_arch = "wasm32")]
    FileLoaded(Vec<u8>),
}

/// Application initialization state
enum InitState {
    /// Not yet started initialization
    Uninitialized,
    /// Async initialization in progress
    Initializing {
        /// Keep window alive during initialization
        #[allow(dead_code)]
        window: Arc<Window>,
    },
    /// Fully initialized and ready
    Ready(EguiViewerApp<'static>),
    /// Initialization failed
    #[allow(dead_code)]
    Failed(String),
}

/// Debug actions triggered by key presses
enum DebugAction {
    ToggleOrtho,
    #[cfg(not(target_arch = "wasm32"))]
    Exit,
}

/// Application state for the winit event loop with egui integration
pub struct App {
    state: InitState,
    proxy: EventLoopProxy<AppEvent>,

    /// Pending file path to load (native only)
    #[cfg(not(target_arch = "wasm32"))]
    pending_gltf_path: Option<std::path::PathBuf>,

    /// Pending file data to load (web only)
    #[cfg(target_arch = "wasm32")]
    pending_gltf_data: Option<Vec<u8>>,
}

impl App {
    pub fn new(proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            state: InitState::Uninitialized,
            proxy,
            #[cfg(not(target_arch = "wasm32"))]
            pending_gltf_path: None,
            #[cfg(target_arch = "wasm32")]
            pending_gltf_data: None,
        }
    }

    /// Start async initialization of the viewer
    fn start_initialization(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("glTF Viewer")
            .with_min_inner_size(winit::dpi::PhysicalSize::new(800, 600));

        // Platform-specific window setup
        #[cfg(target_arch = "wasm32")]
        let (window_attrs, initial_size) = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            let web_window = web_sys::window().expect("no window");
            let document = web_window.document().expect("no document");
            let canvas = document
                .get_element_by_id("gltf-viewer-canvas")
                .expect("no canvas with id 'gltf-viewer-canvas'")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("element is not a canvas");

            let dpr = web_window.device_pixel_ratio();
            let width = (canvas.client_width() as f64 * dpr) as u32;
            let height = (canvas.client_height() as f64 * dpr) as u32;
            let initial_size = Some(winit::dpi::PhysicalSize::new(width.max(1), height.max(1)));

            // Don't manually set canvas dimensions or inner_size - let winit's
            // ResizeObserver handle CSS-based sizing (per winit PR #2859).
            // This allows winit to detect and fire WindowEvent::Resized when
            // the browser window is resized.
            let window_attrs = window_attrs.with_canvas(Some(canvas));

            (window_attrs, initial_size)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let initial_size = None;

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        self.state = InitState::Initializing {
            window: Arc::clone(&window),
        };

        let proxy = self.proxy.clone();
        let init_future = async move {
            // Note: EguiViewerApp::from_window currently panics on failure.
            // For more robust error handling, the core library could be modified
            // to return Result<EguiViewerApp, Error> instead.
            let viewer_app = EguiViewerApp::from_window(window, initial_size).await;
            let _ = proxy.send_event(AppEvent::ViewerReady(Box::new(viewer_app)));
        };

        // Platform-specific async execution
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(init_future);

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(init_future);
    }

    /// Handle the RedrawRequested event - build UI and render the frame
    fn handle_redraw_requested(&mut self) {
        if !matches!(self.state, InitState::Ready(_)) {
            return;
        }

        // Process any pending glTF file load first
        self.process_pending_load();

        // Build egui UI and render
        let mut ui_actions = ui::UiActions::default();
        if let InitState::Ready(ref mut viewer_app) = self.state {
            if let Err(e) = viewer_app.render(|ctx, viewer| {
                ui_actions = ui::build(ctx, viewer);
            }) {
                log::error!("Render error: {}", e);
            }
        }

        // Handle UI actions (outside the borrow scope)
        if ui_actions.load_file {
            self.open_file_dialog();
        }
        if ui_actions.clear_scene {
            self.clear_scene();
        }

        // Request next frame
        if let InitState::Ready(ref app) = self.state {
            app.request_redraw();
        }
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
        let InitState::Ready(ref mut viewer_app) = self.state else {
            return;
        };

        let viewer = viewer_app.viewer_mut();
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

    /// Load glTF from bytes (used on web platform)
    #[allow(dead_code)] // Only used on wasm32
    fn load_gltf_from_bytes(&mut self, data: &[u8]) {
        let InitState::Ready(ref mut viewer_app) = self.state else {
            return;
        };

        let viewer = viewer_app.viewer_mut();
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

    /// Setup the hidden file input element for web with proper async callback
    #[cfg(target_arch = "wasm32")]
    fn setup_web_file_input(&self, document: &web_sys::Document) {
        use wasm_bindgen::{closure::Closure, JsCast};
        use web_sys::{FileReader, HtmlInputElement};

        let input: HtmlInputElement = document
            .create_element("input")
            .unwrap()
            .dyn_into()
            .unwrap();
        input.set_type("file");
        input.set_accept(".gltf,.glb");
        input.set_id("gltf-file-input");
        let _ = input.style().set_property("display", "none");

        // Clone proxy for the closure
        let proxy = self.proxy.clone();

        let onchange = Closure::<dyn Fn(_)>::new(move |event: web_sys::Event| {
            let input: HtmlInputElement = event.target().unwrap().dyn_into().unwrap();
            if let Some(files) = input.files() {
                if let Some(file) = files.get(0) {
                    let reader = FileReader::new().unwrap();
                    let proxy_clone = proxy.clone();

                    let onload = Closure::<dyn Fn(_)>::new(move |event: web_sys::ProgressEvent| {
                        let reader: FileReader = event.target().unwrap().dyn_into().unwrap();
                        if let Ok(result) = reader.result() {
                            let array = js_sys::Uint8Array::new(&result);
                            let data = array.to_vec();
                            let _ = proxy_clone.send_event(AppEvent::FileLoaded(data));
                        }
                    });

                    reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                    onload.forget(); // Prevent cleanup - closure must live forever
                    reader.read_as_array_buffer(&file).unwrap();
                }
            }
        });

        input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
        onchange.forget(); // Prevent cleanup - closure must live forever

        document.body().unwrap().append_child(&input).unwrap();
    }

    /// Clear the scene
    fn clear_scene(&mut self) {
        let InitState::Ready(ref mut viewer_app) = self.state else {
            return;
        };
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
        let InitState::Ready(ref mut viewer_app) = self.state else {
            return;
        };
        let camera = viewer_app.viewer_mut().camera_mut();
        camera.ortho = !camera.ortho;
    }
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !matches!(self.state, InitState::Uninitialized) {
            // Already initialized or initializing
            if let InitState::Ready(ref app) = self.state {
                app.request_redraw();
            }
            return;
        }
        self.start_initialization(event_loop);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::ViewerReady(viewer_app) => {
                log::info!("Viewer initialization complete");
                viewer_app.request_redraw();
                self.state = InitState::Ready(*viewer_app);

                // On web, the canvas size may not be known until after initialization.
                // Send a synthetic resize event to ensure the viewer has the correct size.
                #[cfg(target_arch = "wasm32")]
                if let InitState::Ready(ref mut app) = self.state {
                    let size = app.window().inner_size();
                    app.handle_window_event(&WindowEvent::Resized(size));
                }
            }
            AppEvent::ViewerInitFailed(error) => {
                log::error!("Viewer initialization failed: {}", error);
                self.state = InitState::Failed(error);
            }
            #[cfg(target_arch = "wasm32")]
            AppEvent::FileLoaded(data) => {
                self.pending_gltf_data = Some(data);
                if let InitState::Ready(ref app) = self.state {
                    app.request_redraw();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match &mut self.state {
            InitState::Ready(viewer_app) => {
                match event {
                    WindowEvent::CloseRequested => {
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        self.handle_redraw_requested();
                    }
                    _ => {
                        viewer_app.handle_window_event(&event);

                        if let Some(app_event) =
                            wgpu_engine::winit_support::convert_window_event(event)
                        {
                            if let Some(action) = Self::get_debug_key_action(&app_event) {
                                self.handle_debug_key_action(action, event_loop);
                            }
                        }
                    }
                }
            }
            InitState::Initializing { .. } => {
                // Minimal handling during initialization
                if let WindowEvent::CloseRequested = event {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let InitState::Ready(ref mut viewer_app) = self.state {
            viewer_app.handle_device_event(&event);
        }
    }
}
