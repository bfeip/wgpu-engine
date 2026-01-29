mod ui;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use cgmath::Vector3;
use wgpu_engine::common::RgbaColor;
use wgpu_engine::egui_support::EguiViewerApp;
use wgpu_engine::input::{ElementState, Key};
use wgpu_engine::load_gltf_scene_from_path;
use wgpu_engine::operator::BuiltinOperatorId;
use wgpu_engine::scene::{Light, LightType, Scene, MAX_LIGHTS};

/// Debug actions triggered by key presses
enum DebugAction {
    CycleOperator,
    ToggleOrtho,
}

/// Application state for the winit event loop with egui integration
struct App<'a> {
    viewer_app: Option<EguiViewerApp<'a>>,
    /// Pending file path to load (set by file dialog, processed in main loop)
    pending_gltf_path: Option<std::path::PathBuf>,
    /// Pending HDR environment path to load
    pending_hdr_path: Option<std::path::PathBuf>,
}

impl<'a> App<'a> {
    /// Handle the RedrawRequested event - build UI and render the frame
    fn handle_redraw_requested(&mut self) {
        // Process any pending file loads
        if self.pending_gltf_path.is_some() {
            self.load_gltf_file();
        }
        if self.pending_hdr_path.is_some() {
            self.load_hdr_file();
        }

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
            self.open_gltf_file_dialog();
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

        // Request next frame
        self.viewer_app.as_ref().unwrap().request_redraw();
    }

    /// Open a file dialog to select a glTF file
    fn open_gltf_file_dialog(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter("glTF", &["gltf", "glb"])
            .pick_file();

        if let Some(path) = file {
            self.pending_gltf_path = Some(path);
        }
    }

    /// Load a glTF file into the scene
    fn load_gltf_file(&mut self) {
        let Some(path) = self.pending_gltf_path.take() else {
            return;
        };

        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let aspect = {
            let size = viewer.size();
            size.0 as f32 / size.1 as f32
        };

        let path_str = path.display().to_string();
        match load_gltf_scene_from_path(&path, aspect) {
            Ok(result) => {
                // Replace the scene with the loaded one
                viewer.set_scene(result.scene);

                // Apply camera if one was found in the glTF
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

    /// Clear the scene (remove all nodes, meshes, instances, etc.)
    fn clear_scene(&mut self) {
        let viewer_app = self.viewer_app.as_mut().unwrap();
        viewer_app.viewer_mut().set_scene(Scene::new());
        log::info!("Scene cleared");
    }

    /// Add a new light of the specified type to the scene
    fn add_light(&mut self, light_type: LightType) {
        let viewer = self.viewer_app.as_mut().unwrap().viewer_mut();
        let lights = &mut viewer.scene_mut().lights;

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

        lights.push(light);
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

    /// Handle debug key actions
    fn handle_debug_key_action(&mut self, action: DebugAction, _event_loop: &ActiveEventLoop) {
        match action {
            DebugAction::CycleOperator => self.cycle_operator_mode(),
            DebugAction::ToggleOrtho => self.toggle_ortho(),
        }
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
            Key::Character('c') => Some(DebugAction::CycleOperator),
            Key::Character('o') => Some(DebugAction::ToggleOrtho),
            _ => None,
        }
    }

    /// Cycle between Walk and Navigation operators by swapping their positions
    fn cycle_operator_mode(&mut self) {
        let viewer_app = self.viewer_app.as_mut().unwrap();
        let viewer = viewer_app.viewer_mut();
        let walk_id: u32 = BuiltinOperatorId::Walk.into();
        let nav_id: u32 = BuiltinOperatorId::Navigation.into();

        // Swap Walk and Navigation operators, preserving Selection operator position
        let (op_mgr, dispatcher) = viewer.operator_manager_and_dispatcher_mut();
        op_mgr.swap(walk_id, nav_id, dispatcher);
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
        // Initialize on first resume
        if self.viewer_app.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("WGPU Engine - egui Example")
                .with_min_inner_size(winit::dpi::PhysicalSize::new(1600, 800));

            let viewer_app = pollster::block_on(EguiViewerApp::with_window_attrs(
                event_loop,
                window_attrs,
            ));

            viewer_app.request_redraw();
            self.viewer_app = Some(viewer_app);
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

fn main() {
    // Initialize logging
    env_logger::init();

    // Create event loop
    let event_loop = EventLoop::new().unwrap();

    // Create application state
    let mut app = App {
        viewer_app: None,
        pending_gltf_path: None,
        pending_hdr_path: None,
    };

    // Run the event loop
    event_loop.run_app(&mut app).unwrap();
}
