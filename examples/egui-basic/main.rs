use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use wgpu_engine::{egui_support::EguiRenderer, winit_support, Viewer};

/// Application state for the winit event loop with egui integration
struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<Viewer<'a>>,
    egui_ctx: Option<egui::Context>,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<EguiRenderer>,
}

impl<'a> App<'a> {
    /// Initialize the window, viewer, and egui
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("WGPU Engine - egui Example");

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        // Create viewer with the window
        let size = window.inner_size();
        let viewer = pollster::block_on(Viewer::new(
            Arc::clone(&window),
            size.width,
            size.height,
        ));

        // Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let (device, _queue) = viewer.wgpu_resources();
        let egui_renderer = EguiRenderer::new(device, viewer.surface_format());

        window.request_redraw();

        self.window = Some(window);
        self.viewer = Some(viewer);
        self.egui_ctx = Some(egui_ctx);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Initialize on first resume
        if self.window.is_none() {
            self.initialize(event_loop);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();
        let viewer = self.viewer.as_mut().unwrap();
        let egui_ctx = self.egui_ctx.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();

        // Give egui a chance to handle window events first
        let response = egui_winit.on_window_event(window.as_ref(), &event);
        let egui_consumed = response.consumed;

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let egui_renderer = self.egui_renderer.as_mut().unwrap();

                // Build egui UI
                let raw_input = egui_winit.take_egui_input(window.as_ref());
                let full_output = egui_ctx.run(raw_input, |ctx| {
                    build_ui(ctx, viewer);
                });

                // Process egui platform output (clipboard, cursor, etc.)
                egui_winit.handle_platform_output(
                    window.as_ref(),
                    full_output.platform_output.clone(),
                );

                // Render 3D scene + egui overlay in one call
                if let Err(e) = viewer.render_with_egui(
                    egui_renderer,
                    egui_ctx,
                    window.scale_factor() as f32,
                    full_output,
                ) {
                    log::error!("Render error: {}", e);
                }

                // Request next frame
                window.request_redraw();
            }
            _ => {}
        }

        // Convert and handle all window events (only if egui didn't consume them)
        if !egui_consumed {
            if let Some(app_event) = winit_support::convert_window_event(event) {
                viewer.handle_event(&app_event);

                // Check for exit on Escape key
                if let wgpu_engine::event::Event::KeyboardInput { event: key_event, .. } = &app_event {
                    if matches!(
                        key_event.logical_key,
                        wgpu_engine::input::Key::Named(wgpu_engine::input::NamedKey::Escape)
                    ) {
                        event_loop.exit();
                    }
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Convert and handle device events (e.g., mouse motion)
        // Only handle if egui didn't consume the last window event
        if let Some(app_event) = winit_support::convert_device_event(event) {
            self.viewer.as_mut().unwrap().handle_event(&app_event);
        }
    }
}

/// Build the egui UI
fn build_ui(ctx: &egui::Context, viewer: &Viewer) {
    // Main control panel
    egui::Window::new("Viewer Controls")
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
            ui.heading("Camera");

            let camera = viewer.camera();
            ui.label(format!(
                "Position: ({:.2}, {:.2}, {:.2})",
                camera.eye.x, camera.eye.y, camera.eye.z
            ));
            ui.label(format!(
                "Target: ({:.2}, {:.2}, {:.2})",
                camera.target.x, camera.target.y, camera.target.z
            ));

            ui.separator();

            ui.heading("Controls");
            ui.label("Left Mouse: Orbit camera");
            ui.label("Right Mouse: Pan camera");
            ui.label("Mouse Wheel: Zoom in/out");
            ui.label("ESC: Exit application");

            ui.separator();

            ui.heading("Scene Info");
            ui.label(format!("Meshes: {}", viewer.scene.meshes.len()));
            ui.label(format!("Instances: {}", viewer.scene.instances.len()));
            ui.label(format!("Nodes: {}", viewer.scene.nodes.len()));
            ui.label(format!("Lights: {}", viewer.scene.lights.len()));
        });

    // Performance overlay
    egui::Window::new("Performance")
        .default_pos([10.0, 300.0])
        .show(ctx, |ui| {
            ui.label(format!(
                "FPS: {:.1}",
                ctx.input(|i| i.stable_dt).recip()
            ));
        });
}

fn main() {
    // Initialize logging
    env_logger::init();

    // Create event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    // Create application state
    let mut app = App {
        window: None,
        viewer: None,
        egui_ctx: None,
        egui_winit: None,
        egui_renderer: None,
    };

    // Run the event loop
    event_loop.run_app(&mut app).unwrap();
}
