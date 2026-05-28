mod boolean;
mod document;
mod grid;
mod operators;
mod tool;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use egui_wgpu::RendererOptions;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::winit_support;
use duck_engine_viewer::Viewer;
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator, SelectionMode};
use duck_engine_viewer::common::{
    Vector3, InnerSpace
};
use duck_engine_viewer::scene::{
    Scene, NodeFlags, NodePayload, PositionedCamera,
};
use duck_engine_viewer::selection::SelectionItem;

use crate::boolean::BooleanKind;
use crate::operators::{BooleanOperator, ConstructionOptions, SphereOperator};
use crate::tool::ModelingTool;

use document::Document;

#[derive(Default, PartialEq, Eq)]
enum OperatorKind {
    #[default]
    Selection,
    Sphere,
    Boolean,
}

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

    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,

    sel_op: Arc<Mutex<SelectionOperator>>,
    sphere_op: Arc<Mutex<SphereOperator>>,
    boolean_op: Arc<Mutex<BooleanOperator>>,
    active_operator: OperatorKind,
}

impl ViewerState<'static> {
    async fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Modeler"))
                .expect("Failed to create window"),
        );

        let mut viewer = Viewer::from_window(Arc::clone(&window)).await;

        let egui_ctx = egui::Context::default();
        egui_extras::install_image_loaders(&egui_ctx);

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

        let construction_options = Rc::new(RefCell::new(ConstructionOptions::new()));
        let document = Arc::new(Mutex::new(Document::new(viewer.scene())));

        let sel_op = Arc::new(Mutex::new(SelectionOperator::with_mode(SelectionMode::Node)));
        viewer.dispatcher_mut().push_back(sel_op.clone());
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(NavigationOperator::new())));

        let sphere_op = Arc::new(Mutex::new(SphereOperator::new(
            Rc::clone(&construction_options),
            Arc::clone(&document),
        )));

        let boolean_op = Arc::new(Mutex::new(BooleanOperator::new(
            Rc::clone(&construction_options),
            Arc::clone(&document),
        )));

        Self {
            egui_renderer,
            egui_winit,
            egui_ctx,
            viewer,
            window,
            construction_options,
            document,
            sel_op,
            sphere_op,
            boolean_op,
            active_operator: OperatorKind::Selection,
        }
    }

    fn set_default_scene(&mut self) {
        let mut scene = Scene::new();

        // Setup default camera and lighting
        let eye = [75.0, 50.0, 75.0].into();
        let target = [0.0, 0.0, 0.0].into();
        let forward: Vector3 = target - eye;
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

        let coptions = self.construction_options.borrow();
        let _grid = grid::Grid::add_to_scene(&mut scene, &coptions.grid, &coptions.construction_plane);
        drop(coptions);

        let scene_arc = Arc::new(Mutex::new(scene));
        self.viewer.set_scene(Arc::clone(&scene_arc));
        self.document.lock().unwrap().set_scene(scene_arc);
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
        let egui_ctx = self.egui_ctx.clone();
        let full_output = egui_ctx.run(raw_input, |ctx| {
            self.render_operator_palette(ctx);
            if self.active_operator == OperatorKind::Boolean {
                self.render_boolean_panel(ctx);
            }
        });
        self.egui_winit.handle_platform_output(&self.window, full_output.platform_output.clone());

        // Exit boolean mode if the operator has finished (via keyboard or panel buttons).
        if self.active_operator == OperatorKind::Boolean
            && self.boolean_op.lock().unwrap().is_finished()
        {
            self.switch_tool(OperatorKind::Selection);
        }

        match self.viewer.render_scene() {
            Ok((output, view, mut encoder)) => {
                self.render_egui_overlay(&full_output, &mut encoder, &view);
                self.viewer.present(encoder, output);
            }
            Err(e) => log::error!("Render error: {}", e),
        }

        self.window.request_redraw();
    }

    fn switch_tool(&mut self, kind: OperatorKind) {
        self.sphere_op.lock().unwrap().deactivate();
        self.boolean_op.lock().unwrap().deactivate();
        self.viewer.dispatcher_mut().remove(&self.sphere_op);
        self.viewer.dispatcher_mut().remove(&self.boolean_op);
        match kind {
            OperatorKind::Selection => { self.viewer.dispatcher_mut().move_to_front(&self.sel_op); }
            OperatorKind::Sphere    => { self.viewer.dispatcher_mut().push_front(self.sphere_op.clone()); }
            OperatorKind::Boolean   => { self.viewer.dispatcher_mut().push_front(self.boolean_op.clone()); }
        }
        self.active_operator = kind;
    }

    fn render_operator_palette(&mut self, ctx: &egui::Context) {
        const CURSOR_SVG: &[u8] =
            include_bytes!("../../../assets/svg/cursor-svgrepo-com.svg");
        const SPHERE_SVG: &[u8] =
            include_bytes!("../../../assets/svg/sphere-svgrepo-com.svg");
        const BOOLEAN_SVG: &[u8] =
            include_bytes!("../../../assets/svg/boolean-and.svg");

        egui::SidePanel::left("operator_palette")
            .resizable(false)
            .exact_width(56.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);

                let sel_btn = ui.add(
                    egui::Button::image(
                        egui::Image::from_bytes("bytes://cursor.svg", CURSOR_SVG)
                            .fit_to_exact_size(egui::vec2(32.0, 32.0)),
                    )
                    .selected(self.active_operator == OperatorKind::Selection),
                );
                if sel_btn.clicked() {
                    self.switch_tool(OperatorKind::Selection);
                }

                ui.add_space(4.0);

                let sphere_btn = ui.add(
                    egui::Button::image(
                        egui::Image::from_bytes("bytes://sphere.svg", SPHERE_SVG)
                            .fit_to_exact_size(egui::vec2(32.0, 32.0)),
                    )
                    .selected(self.active_operator == OperatorKind::Sphere),
                );
                if sphere_btn.clicked() {
                    self.switch_tool(OperatorKind::Sphere);
                }

                ui.add_space(4.0);

                let boolean_btn = ui.add(
                    egui::Button::image(
                        egui::Image::from_bytes("bytes://boolean-and.svg", BOOLEAN_SVG)
                            .fit_to_exact_size(egui::vec2(32.0, 32.0)),
                    )
                    .selected(self.active_operator == OperatorKind::Boolean),
                );
                if boolean_btn.clicked() && self.active_operator != OperatorKind::Boolean {
                    self.switch_tool(OperatorKind::Boolean);
                }
            });
    }

    fn render_boolean_panel(&mut self, ctx: &egui::Context) {
        // Collect deselections and action requests; apply them after the egui closure
        // to avoid holding borrows across mutable viewer calls.
        let mut deselect: Vec<SelectionItem> = Vec::new();
        let mut apply_clicked = false;
        let mut cancel_clicked = false;

        egui::Window::new("Boolean Operation")
            .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                // --- Operation type ---
                ui.label("Operation");
                ui.horizontal(|ui| {
                    let mut op = self.boolean_op.lock().unwrap();
                    ui.selectable_value(&mut op.kind, BooleanKind::Subtract, "Subtract");
                    ui.selectable_value(&mut op.kind, BooleanKind::Union, "Union");
                    ui.selectable_value(&mut op.kind, BooleanKind::Intersect, "Intersect");
                });

                ui.separator();

                // --- Target (primary selection) ---
                ui.label("Target");
                let primary = self.viewer.selection().primary();
                let target_node = primary.and_then(|item| match item {
                    SelectionItem::Node(id) => Some(id),
                    _ => None,
                });
                let doc = self.document.lock().unwrap();
                let target_name = target_node
                    .and_then(|n| doc.part_for_node(n))
                    .and_then(|p| doc.get_part(p).map(|part| part.name.clone()))
                    .unwrap_or_else(|| "(none — click a part)".to_owned());
                drop(doc);
                ui.label(&target_name);

                ui.separator();

                // --- Tool parts (all non-primary selections) ---
                ui.label("Tools");
                let tool_items: Vec<SelectionItem> = self.viewer.selection().iter()
                    .filter(|&&item| Some(item) != primary)
                    .copied()
                    .collect();
                if tool_items.is_empty() {
                    ui.label("(shift-click parts to add tools)");
                } else {
                    let doc = self.document.lock().unwrap();
                    for &item in &tool_items {
                        let node_id = match item { SelectionItem::Node(id) => Some(id), _ => None };
                        let name = node_id
                            .and_then(|n| doc.part_for_node(n))
                            .and_then(|p| doc.get_part(p).map(|part| part.name.clone()))
                            .unwrap_or_else(|| "Unknown".to_owned());
                        ui.horizontal(|ui| {
                            ui.label(&name);
                            if ui.small_button("×").clicked() {
                                deselect.push(item);
                            }
                        });
                    }
                }

                ui.separator();

                // --- Action buttons ---
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                    if ui.button("Apply  ⏎").clicked() {
                        apply_clicked = true;
                    }
                });
            });

        // Apply × deselections outside the closure.
        for item in deselect {
            self.viewer.selection_mut().remove(&item);
        }

        // Dispatch panel Apply / Cancel through the operator so all cleanup is co-located.
        if apply_clicked {
            let mut op = self.boolean_op.lock().unwrap();
            if let Err(e) = op.apply() {
                log::error!("Boolean failed: {e}");
            } else {
                drop(op);
                self.viewer.selection_mut().clear();
            }
        } else if cancel_clicked {
            self.boolean_op.lock().unwrap().cancel();
        }
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
