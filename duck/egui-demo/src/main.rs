mod ui;

use std::path::PathBuf;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use cgmath::Vector3;
use duck_engine_viewer::common::RgbaColor;
use duck_engine_viewer::egui_support::EguiViewerApp;
use duck_engine_viewer::input::{ElementState, Key};
use duck_engine_viewer::operator::NavigationMode;
use duck_engine_viewer::import_export;
use duck_engine_viewer::scene::{Light, LightType, Scene, MAX_LIGHTS};
use duck_engine_viewer::winit_support;

/// Debug actions triggered by key presses
enum DebugAction {
    CycleOperator,
    ToggleOrtho,
    CycleWorkflow,
}

/// Application state for the winit event loop with egui integration
struct App<'a> {
    viewer_app: Option<EguiViewerApp<'a>>,
    ui: ui::UiState,
    /// Index of the currently active workflow (cycled by the W debug key).
    workflow_index: usize,
    /// Pending HDR environment path to load
    pending_hdr_path: Option<std::path::PathBuf>,
    /// Pending scene file path to load
    pending_scene_load_path: Option<std::path::PathBuf>,
    /// Pending scene file path to save
    pending_scene_save_path: Option<std::path::PathBuf>,
}

impl<'a> App<'a> {
    /// Handle the RedrawRequested event - build UI and render the frame
    fn handle_redraw_requested(&mut self) {
        // Process any pending file operations
        if self.pending_hdr_path.is_some() {
            self.load_hdr_file();
        }
        if self.pending_scene_load_path.is_some() {
            self.load_scene_file();
        }
        if self.pending_scene_save_path.is_some() {
            self.save_scene_file();
        }

        // Build egui UI and render
        let mut ui_actions = ui::UiActions::default();
        {
            let viewer_app = self.viewer_app.as_mut().unwrap();
            let ui = &mut self.ui;
            if let Err(e) = viewer_app.render(|ctx, viewer| {
                ui_actions = ui.build(ctx, viewer);
            }) {
                log::error!("Render error: {}", e);
            }
        }

        // Handle UI actions (after releasing viewer_app borrow)
        if ui_actions.load_scene {
            self.open_scene_file_dialog();
        }
        if ui_actions.save_scene {
            self.save_scene_file_dialog();
        }
        if ui_actions.clear_scene {
            self.clear_scene();
        }
        if let Some(light_type) = ui_actions.add_light {
            self.add_light(light_type);
        }
        if ui_actions.load_environment {
            self.open_hdr_file_dialog();
        }
        if ui_actions.clear_environment {
            self.clear_environment();
        }
        for change in ui_actions.visibility_changes {
            let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
            viewer
                .scene_mut()
                .set_node_visibility(change.node_id, change.new_visibility);
        }
        if let Some(camera) = ui_actions.set_camera {
            self.viewer_app.as_mut().unwrap().viewer_mut().set_camera(camera);
        }

        // Request next frame
        self.viewer_app.as_ref().unwrap().request_redraw();
    }

    /// Clear the scene (remove all nodes, meshes, instances, etc.)
    fn clear_scene(&mut self) {
        let viewer_app = self.viewer_app.as_mut().unwrap();
        viewer_app.viewer_mut().set_scene(Scene::new());
        log::info!("Scene cleared");
    }

    /// Add a new light of the specified type to the scene
    fn add_light(&mut self, light_type: LightType) {
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let lights = &mut viewer.scene_mut().lights();

        // Check limit
        if lights.len() >= MAX_LIGHTS {
            log::warn!("Cannot add light: maximum of {} lights reached", MAX_LIGHTS);
            return;
        }

        let white = RgbaColor {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };

        let light = match light_type {
            LightType::Point => Light::point(Vector3::new(0.0, 3.0, 0.0), white, 1.0),
            LightType::Directional => {
                Light::directional(Vector3::new(0.0, -1.0, 0.0), white, 1.0)
            }
            LightType::Spot => Light::spot(
                Vector3::new(0.0, 3.0, 0.0),
                Vector3::new(0.0, -1.0, 0.0),
                white,
                1.0,
                30.0_f32.to_radians(),
                45.0_f32.to_radians(),
            ),
        };

        viewer.scene_mut().add_light(light);
        log::info!("Added {:?} light", light_type);
    }

    /// Open a file dialog to select an HDR environment map
    fn open_hdr_file_dialog(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter("HDR", &["hdr"])
            .pick_file();

        if let Some(path) = file {
            self.pending_hdr_path = Some(path);
        }
    }

    /// Load an HDR environment map into the scene
    fn load_hdr_file(&mut self) {
        let Some(path) = self.pending_hdr_path.take() else {
            return;
        };

        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let path_str = path.display().to_string();

        let scene = viewer.scene_mut();
        let env_id = scene.add_environment_map_from_hdr_path(&path);
        scene.set_active_environment_map(Some(env_id));
        log::info!("Loaded HDR environment: {}", path_str);
    }

    /// Clear the active environment map
    fn clear_environment(&mut self) {
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        viewer.scene_mut().set_active_environment_map(None);
        log::info!("Environment cleared");
    }

    /// Open a file dialog to select a scene file to load
    fn open_scene_file_dialog(&mut self) {
        #[allow(unused_mut)]
        let mut extensions: Vec<&str> = vec!["glb", "gltf", "duck"];

        #[cfg(feature = "assimp")]
        extensions.extend_from_slice(import_export::assimp::ASSIMP_EXTENSIONS);

        #[cfg(feature = "usd")]
        extensions.extend_from_slice(import_export::usd::USD_EXTENSIONS);

        #[cfg(feature = "cad")]
        extensions.extend_from_slice(import_export::cad::CAD_EXTENSIONS);

        let file = rfd::FileDialog::new()
            .add_filter("3D Scenes", &extensions)
            .pick_file();

        if let Some(path) = file {
            self.pending_scene_load_path = Some(path);
        }
    }

    /// Load a scene file using the unified loader (auto-detects format)
    fn load_scene_file(&mut self) {
        use import_export::{load_sync, SceneSource, LoadOptions};
        let Some(path) = self.pending_scene_load_path.take() else {
            return;
        };

        let path_str = path.display().to_string();
        match load_sync(SceneSource::Path(path), LoadOptions::default()) {
            Ok(result) => {
                let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
                viewer.set_scene(result.scene);
                if let Some(camera) = result.camera {
                    viewer.set_camera(camera);
                } else if let Some(bounds) = viewer.scene().bounding() {
                    viewer.camera_mut().fit_to_bounds(&bounds);
                }
                log::info!("Loaded scene: {}", path_str);
            }
            Err(e) => {
                log::error!("Failed to load scene {}: {}", path_str, e);
            }
        }
    }

    /// Open a file dialog to select where to save the scene
    fn save_scene_file_dialog(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter("Duck Scene", &["duck"])
            .set_file_name("scene.duck")
            .save_file();

        if let Some(path) = file {
            self.pending_scene_save_path = Some(path);
        }
    }

    /// Save the scene to a file
    fn save_scene_file(&mut self) {
        let Some(path) = self.pending_scene_save_path.take() else {
            return;
        };

        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let path_str = path.display().to_string();

        match import_export::format::save_to_file(viewer.scene(), &path) {
            Ok(()) => {
                log::info!("Saved scene: {}", path_str);
            }
            Err(e) => {
                log::error!("Failed to save scene {}: {}", path_str, e);
            }
        }
    }

    /// Handle debug key actions
    fn handle_debug_key_action(&mut self, action: DebugAction, _event_loop: &ActiveEventLoop) {
        match action {
            DebugAction::CycleOperator => self.cycle_operator_mode(),
            DebugAction::ToggleOrtho => self.toggle_ortho(),
            DebugAction::CycleWorkflow => self.cycle_workflow(),
        }
    }

    /// Check if event is a debug key press and return the action
    fn get_debug_key_action(event: &duck_engine_viewer::event::Event) -> Option<DebugAction> {
        let duck_engine_viewer::event::Event::KeyboardInput { event: key_event, .. } = event else {
            return None;
        };

        if key_event.state != ElementState::Pressed || key_event.repeat {
            return None;
        }

        match &key_event.logical_key {
            Key::Character('c') => Some(DebugAction::CycleOperator),
            Key::Character('o') => Some(DebugAction::ToggleOrtho),
            Key::Character('w') => Some(DebugAction::CycleWorkflow),
            _ => None,
        }
    }

    /// Cycle between Orbit and Walk navigation modes
    fn cycle_operator_mode(&mut self) {
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let new_mode = match viewer.navigation_mode() {
            NavigationMode::Turntable => NavigationMode::Walk,
            NavigationMode::Walk => NavigationMode::Trackball,
            NavigationMode::Trackball => NavigationMode::Turntable,
        };
        viewer.set_navigation_mode(new_mode);
    }

    /// Toggle between perspective and orthographic projection
    fn toggle_ortho(&mut self) {
        let viewer_app = self.viewer_app.as_mut().unwrap();
        let camera = viewer_app.viewer_mut().camera_mut();
        camera.ortho = !camera.ortho;
    }

    /// Cycle through the built-in rendering workflows.
    fn cycle_workflow(&mut self) {
        use duck_engine_viewer::renderer::RenderWorkflow;
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        self.workflow_index = (self.workflow_index + 1) % 2;
        let workflow: Box<dyn RenderWorkflow> = match self.workflow_index {
            0 => Box::new(viewer.shaded_workflow()),
            _ => Box::new(viewer.hidden_line_workflow()),
        };
        log::info!("Switched to '{}' workflow", workflow.name());
        viewer.set_workflow(workflow);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Initialize on first resume
        if self.viewer_app.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("Duck Engine - egui Example")
                .with_inner_size(winit::dpi::LogicalSize::new(1600, 800));

            let viewer_app = pollster::block_on(EguiViewerApp::with_window_attrs(
                event_loop,
                window_attrs,
            ));

            viewer_app.request_redraw();
            self.viewer_app = Some(viewer_app);

            let assets_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
            let default_scene = assets_dir.join("default-scene.duck");
            if default_scene.exists() {
                self.pending_scene_load_path = Some(default_scene);
            } else {
                log::warn!("Default scene not found: {}", default_scene.display());
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let viewer_app = self.viewer_app.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.handle_redraw_requested();
            }
            _ => {
                // Handle event - egui gets priority via handle_window_event
                viewer_app.handle_window_event(&event);

                // Check for debug keys (convert to app event for checking)
                if let Some(app_event) = winit_support::convert_window_event(event) {
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

fn main() {
    // Initialize logging
    env_logger::init();

    // Create event loop
    let event_loop = EventLoop::new().unwrap();

    // Create application state
    let mut app = App {
        viewer_app: None,
        ui: ui::UiState::default(),
        workflow_index: 0,
        pending_hdr_path: None,
        pending_scene_load_path: None,
        pending_scene_save_path: None,
    };

    // Run the event loop
    event_loop.run_app(&mut app).unwrap();
}