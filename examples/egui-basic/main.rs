use std::sync::Arc;
use egui_wgpu::RendererOptions;
use winit::{
    application::ApplicationHandler, event, event_loop::{ActiveEventLoop, EventLoop}, window::{Window, WindowId}
};

use wgpu_engine::{Viewer, winit_support};
use wgpu_engine::input::{ElementState, Key, NamedKey};
use wgpu_engine::operator::BuiltinOperatorId;

/// Application state for the winit event loop with egui integration
struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<Viewer<'a>>,
    egui_ctx: Option<egui::Context>,
    egui_winit: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,
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
        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            viewer.surface_format(),
            RendererOptions::default()
        );

        window.request_redraw();

        self.window = Some(window);
        self.viewer = Some(viewer);
        self.egui_ctx = Some(egui_ctx);
        self.egui_winit = Some(egui_winit);
        self.egui_renderer = Some(egui_renderer);
    }

    /// Handle the RedrawRequested event - build UI and render the frame
    fn handle_redraw_requested(&mut self) {
        let window = self.window.as_ref().unwrap();
        let viewer = self.viewer.as_mut().unwrap();
        let egui_ctx = self.egui_ctx.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        // Dispatch Update event for continuous operations (WASD movement, etc.)
        viewer.update();

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

        // Capture viewer size and scale factor before the closure
        let viewer_size = viewer.size();
        let scale_factor = window.scale_factor() as f32;

        // Render 3D scene + egui overlay
        if let Err(e) = viewer.render_with_overlay(|device, queue, encoder, view| {
            render_egui_to_overlay(
                egui_renderer,
                egui_ctx,
                &full_output,
                viewer_size,
                scale_factor,
                device,
                queue,
                encoder,
                view,
            );
        }) {
            log::error!("Render error: {}", e);
        }

        // Request next frame
        window.request_redraw();
    }

    /// Handle events that egui didn't consume
    fn handle_non_egui_event(&mut self, event: event::Event<()>, event_loop: &ActiveEventLoop) {
        let Some(app_event) = winit_support::convert_event(event) else {
            return;
        };

        // Handle debug keys before passing to viewer
        if let Some(action) = Self::get_debug_key_action(&app_event) {
            match action {
                DebugAction::CycleOperator => self.cycle_operator_mode(),
                DebugAction::Exit => event_loop.exit(),
            }
        }

        self.viewer.as_mut().unwrap().handle_event(&app_event);
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
            Key::Named(NamedKey::Escape) => Some(DebugAction::Exit),
            _ => None,
        }
    }

    /// Cycle between Walk and Navigation operators
    fn cycle_operator_mode(&mut self) {
        let viewer = self.viewer.as_mut().unwrap();
        let walk_id: u32 = BuiltinOperatorId::Walk.into();
        let nav_id: u32 = BuiltinOperatorId::Navigation.into();

        let Some(front_id) = viewer.operator_manager.front_id() else {
            return;
        };

        let next_id = if front_id == walk_id { nav_id } else { walk_id };
        viewer.operator_manager.move_to_front(next_id, &mut viewer.dispatcher);
    }
}

/// Debug actions triggered by key presses
enum DebugAction {
    CycleOperator,
    Exit,
}

/// Render egui overlay on top of the 3D scene
fn render_egui_to_overlay(
    egui_renderer: &mut egui_wgpu::Renderer,
    egui_ctx: &egui::Context,
    full_output: &egui::FullOutput,
    viewer_size: wgpu_engine::common::PhysicalSize<u32>,
    scale_factor: f32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    // Update egui textures
    for (id, image_delta) in &full_output.textures_delta.set {
        egui_renderer.update_texture(device, queue, *id, image_delta);
    }

    // Tessellate and update buffers
    let clipped_primitives = egui_ctx.tessellate(
        full_output.shapes.clone(),
        full_output.pixels_per_point,
    );
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [viewer_size.width, viewer_size.height],
        pixels_per_point: scale_factor,
    };
    egui_renderer.update_buffers(
        device,
        queue,
        encoder,
        &clipped_primitives,
        &screen_descriptor,
    );

    // Create render pass for egui
    {
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Load 3D scene content
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        egui_renderer.render(
            &mut render_pass.forget_lifetime(),
            &clipped_primitives,
            &screen_descriptor,
        );
    }

    // Free textures marked for deletion
    for id in &full_output.textures_delta.free {
        egui_renderer.free_texture(id);
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
        window_id: WindowId,
        event: event::WindowEvent,
    ) {
        let window = self.window.as_ref().unwrap();
        let egui_winit = self.egui_winit.as_mut().unwrap();

        // Give egui a chance to handle window events first
        let response = egui_winit.on_window_event(window.as_ref(), &event);
        let egui_consumed = response.consumed;

        match event {
            event::WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            event::WindowEvent::RedrawRequested => {
                self.handle_redraw_requested();
            }
            _ => {}
        }

        // Convert and handle all window events (only if egui didn't consume them)
        if !egui_consumed {
            let event = event::Event::WindowEvent { window_id, event };
            self.handle_non_egui_event(event, event_loop);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: event::DeviceId,
        event: event::DeviceEvent,
    ) {
        let event = winit::event::Event::DeviceEvent { device_id, event };
        self.handle_non_egui_event(event, event_loop);
    }
}

/// Build the egui UI
fn build_ui(ctx: &egui::Context, viewer: &Viewer) {
    // Determine which operator mode we're in
    let walk_id: u32 = BuiltinOperatorId::Walk.into();
    let nav_id: u32 = BuiltinOperatorId::Navigation.into();
    let front_id = viewer.operator_manager.front_id();
    let is_walk_mode = front_id == Some(walk_id);
    let is_nav_mode = front_id == Some(nav_id);

    // Performance overlay
    egui::TopBottomPanel::new(egui::panel::TopBottomSide::Top, "Performance")
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "FPS: {:.1}",
                    ctx.input(|i| i.stable_dt).recip()
                ));
                ui.separator();
                // Show current mode
                if is_walk_mode {
                    ui.label("Mode: Walk");
                } else if is_nav_mode {
                    ui.label("Mode: Orbit");
                } else if let Some(front) = viewer.operator_manager.front() {
                    ui.label(format!("Mode: {}", front.name()));
                }
            });
        });

    // Main control panel
    egui::SidePanel::new(egui::panel::Side::Left, "Viewer Controls")
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

            // Show mode-specific controls
            if is_walk_mode {
                ui.label("WASD: Move");
                ui.label("Left Mouse Drag: Look around");
            } else if is_nav_mode {
                ui.label("Left Mouse Drag: Orbit camera");
                ui.label("Right Mouse Drag: Pan camera");
                ui.label("Mouse Wheel: Zoom in/out");
            } else {
                // Fallback: show both
                ui.label("WASD: Walk movement");
                ui.label("Left Mouse Drag: Look around / Orbit");
                ui.label("Right Mouse Drag: Pan camera");
                ui.label("Mouse Wheel: Zoom in/out");
            }

            ui.separator();
            ui.label("C: Cycle mode");
            ui.label("ESC: Exit application");

            ui.separator();

            ui.heading("Operators");
            for op in viewer.operator_manager.iter() {
                let prefix = if Some(op.id()) == front_id { "→ " } else { "  " };
                ui.label(format!("{}{}", prefix, op.name()));
            }

            ui.separator();

            ui.heading("Scene Info");
            ui.label(format!("Meshes: {}", viewer.scene.meshes.len()));
            ui.label(format!("Instances: {}", viewer.scene.instances.len()));
            ui.label(format!("Nodes: {}", viewer.scene.nodes.len()));
            ui.label(format!("Lights: {}", viewer.scene.lights.len()));
        });
}

fn main() {
    // Initialize logging
    env_logger::init();

    // Create event loop
    let event_loop = EventLoop::new().unwrap();

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
