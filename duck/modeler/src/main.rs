use std::sync::Arc;

use egui_wgpu::RendererOptions;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use cgmath::InnerSpace;

use duck_engine_viewer::winit_support;
use duck_engine_viewer::Viewer;
use duck_engine_viewer::common::{
    RgbaColor, Transform
};
use duck_engine_viewer::scene::{
    Scene, Material, Mesh, NodeFlags, NodePayload, PositionedCamera, PrimitiveType
};

/// Owns all rendering state: the 3D viewer plus egui context and GPU renderer.
///
/// Field order matters: Rust drops fields in declaration order, so egui
/// resources are released before the viewer and window. This prevents
/// segfaults from background threads on Wayland during shutdown.
struct ViewerState<'a> {
    egui_renderer: egui_wgpu::Renderer,
    egui_winit: egui_winit::State,
    egui_ctx: egui::Context,
    viewer: Viewer<'a>,
    window: Arc<Window>,
}

impl ViewerState<'static> {
    async fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Modeler"))
                .expect("Failed to create window"),
        );

        let viewer = Viewer::from_window(Arc::clone(&window)).await;

        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            viewer.device(),
            viewer.surface_format(),
            RendererOptions::default(),
        );

        Self { egui_renderer, egui_winit, egui_ctx, viewer, window }
    }

    fn set_default_scene(&mut self) {
        let mut scene = Scene::new();

        // Setup default camera and lighting
        let eye = [75.0, 50.0, 75.0].into();
        let target = [0.0, 0.0, 0.0].into();
        let forward: cgmath::Vector3<f32> = target - eye;
        let right = forward.cross([0.0, 1.0, 0.0].into()).normalize();
        let up = right.cross(forward);

        let size = self.viewer.size();

        let camera = PositionedCamera {
            eye,
            target,
            up,
            aspect: size.0 as f32 / size.1 as f32,
            fovy: 35.0,
            znear: 0.01,
            zfar: 10_000f32,
            ortho: false
        };
        let camera_transform = camera.to_node_transform();
        let camera_projection = camera.projection();
        let camera_node_id = scene.add_node(
            None,
            Some("Main camera".to_owned()),
            camera_transform,
            NodeFlags::NONE
        ).expect("Failed to add camera on default scene");
        scene.set_node_payload(camera_node_id, NodePayload::Camera(camera_projection));
        scene.set_active_camera(Some(camera_node_id));
        scene.set_default_light_nodes(camera_node_id);

        // Setup XZ grid
        let grid_mesh = Mesh::plane(100.0, 100.0, 50, 50, PrimitiveType::LineList);
        let grid_mesh_id = scene.add_mesh(grid_mesh);

        let grid_material = Material::new().with_line_color(RgbaColor {
            r: 0.3, g: 0.25, b: 0.3, a: 1.0
        });
        let grid_material_id = scene.add_material(grid_material);

        let _grid_node = scene.add_instance_node(
            None,
            grid_mesh_id,
            grid_material_id,
            Some("XZ Grid".to_owned()),
            Transform::IDENTITY,
            NodeFlags::inert()
        );
        self.viewer.set_scene(scene);
    }
}

impl<'a> ViewerState<'a> {
    fn handle_window_event(&mut self, event: &WindowEvent) {
        let response = self.egui_winit.on_window_event(&self.window, event);
        if !response.consumed {
            if let Some(app_event) = winit_support::convert_window_event(event.clone()) {
                self.viewer.handle_event(&app_event);
            }
        }
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let Some(app_event) = winit_support::convert_device_event(event.clone()) {
            self.viewer.handle_event(&app_event);
        }
    }

    fn handle_redraw(&mut self) {
        self.viewer.update();

        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let full_output = self.egui_ctx.run(raw_input, |_ctx| {});
        self.egui_winit.handle_platform_output(&self.window, full_output.platform_output.clone());

        match self.viewer.render_scene() {
            Ok((output, view, mut encoder)) => {
                self.render_egui_overlay(&full_output, &mut encoder, &view);
                self.viewer.present(encoder, output);
            }
            Err(e) => log::error!("Render error: {}", e),
        }

        self.window.request_redraw();
    }

    fn render_egui_overlay(
        &mut self,
        full_output: &egui::FullOutput,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let device = self.viewer.device();
        let queue = self.viewer.queue();

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let clipped_primitives =
            self.egui_ctx.tessellate(full_output.shapes.clone(), full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: {
                let (w, h) = self.viewer.size();
                [w, h]
            },
            pixels_per_point: self.window.scale_factor() as f32,
        };

        self.egui_renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &clipped_primitives,
                &screen_descriptor,
            );
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

struct App<'a> {
    state: Option<ViewerState<'a>>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let mut state = pollster::block_on(ViewerState::new(event_loop));
            state.set_default_scene();
            state.window.request_redraw();
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    state.handle_redraw();
                }
            }
            _ => {
                if let Some(state) = self.state.as_mut() {
                    state.handle_window_event(&event);
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
        if let Some(state) = self.state.as_mut() {
            state.handle_device_event(&event);
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App { state: None }).unwrap();
}
