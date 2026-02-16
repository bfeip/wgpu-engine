use cgmath::Vector3;
use web_time::Instant;

use crate::{
    common::RgbaColor,
    event::{Event, EventContext, EventDispatcher, EventKind},
    operator::{
        BuiltinOperatorId, NavigationOperator, OperatorManager, SelectionOperator, TransformOperator, WalkOperator
    },
    renderer::Renderer,
    scene::Scene,
    selection::SelectionManager,
};

/// Main viewer that encapsulates the renderer, scene, and event handling
pub struct Viewer<'a> {
    renderer: Renderer<'a>,
    scene: Scene,
    selection: SelectionManager,
    dispatcher: EventDispatcher,
    operator_manager: OperatorManager,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
}

impl<'a> Viewer<'a> {
    /// Create a new Viewer with the given surface target
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let renderer = Renderer::new(surface_target, width, height).await;
        let mut scene = Scene::new();

        // Set up default lighting
        scene.lights = vec![crate::scene::Light::point(
            Vector3::new(1.0, -1.0, 1.0),
            RgbaColor{ r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            100.0
        )];

        let mut dispatcher = EventDispatcher::new();
        let mut operator_manager = OperatorManager::new();

        // Add operators in priority order (first added = highest priority)
        // Selection operator handles picking and must be first
        let selection_operator =
            Box::new(SelectionOperator::new(BuiltinOperatorId::Selection.into()));
        operator_manager.push_back(selection_operator, &mut dispatcher);

        let transform_operator =
            Box::new(TransformOperator::new(BuiltinOperatorId::Transform.into()));
        operator_manager.push_back(transform_operator, &mut dispatcher);

        // Navigation operator for orbit/pan/zoom
        let nav_operator = Box::new(NavigationOperator::new(
            BuiltinOperatorId::Navigation.into(),
        ));
        operator_manager.push_back(nav_operator, &mut dispatcher);

        // Create viewer
        let mut viewer = Self {
            renderer,
            scene,
            selection: SelectionManager::new(),
            dispatcher,
            operator_manager,
            last_update_time: None,
        };

        // Register default event handlers
        viewer.register_default_handlers();

        viewer
    }

    /// Create a new Viewer from a winit Window (native platforms)
    /// The viewer size is automatically determined from the window's inner size
    #[cfg(feature = "winit-support")]
    pub async fn from_window(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        Self::new(window, size.width, size.height).await
    }

    /// Create a new Viewer from an HTML canvas element (WebAssembly)
    /// The viewer size is automatically determined from the canvas dimensions
    #[cfg(target_arch = "wasm32")]
    pub async fn from_canvas(canvas: web_sys::HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        Self::new(wgpu::SurfaceTarget::Canvas(canvas), width, height).await
    }

    /// Register default event handlers for common viewer operations
    fn register_default_handlers(&mut self) {
        // Register Resized handler
        self.dispatcher.register(EventKind::Resized, |event, ctx| {
            if let crate::event::Event::Resized(physical_size) = event {
                ctx.renderer.resize(*physical_size);
            }
            true
        });

        // Register CursorMoved handler to track cursor position
        self.dispatcher
            .register(EventKind::CursorMoved, |event, ctx| {
                if let Event::CursorMoved { position } = event {
                    ctx.renderer.cursor_position = Some((position.0 as f32, position.1 as f32));
                }
                false // Don't stop propagation - other handlers may need cursor position too
            });
    }

    /// Handle a single event by dispatching it to registered handlers
    pub fn handle_event(&mut self, event: &Event) {
        let mut ctx = EventContext {
            renderer: &mut self.renderer,
            scene: &mut self.scene,
            selection: &mut self.selection,
        };
        self.dispatcher.dispatch(event, &mut ctx);
    }

    /// Dispatch an Update event with delta_time since last update.
    ///
    /// Call this once per frame before rendering to enable smooth continuous
    /// operations like WASD movement in the WalkOperator.
    ///
    /// The delta_time is automatically calculated from the time since the
    /// last call to update(). On the first call, a small default delta is used.
    pub fn update(&mut self) {
        let now = Instant::now();
        let delta_time = match self.last_update_time {
            Some(last) => now.duration_since(last).as_secs_f32(),
            None => 1.0 / 60.0, // Assume 60 FPS on first frame
        };
        self.last_update_time = Some(now);

        let event = Event::Update { delta_time };
        self.handle_event(&event);
    }

    /// Get a reference to the current camera
    pub fn camera(&self) -> &crate::camera::Camera {
        self.renderer.camera()
    }

    /// Get a mutable reference to the current camera
    pub fn camera_mut(&mut self) -> &mut crate::camera::Camera {
        self.renderer.camera_mut()
    }

    /// Set the camera to a new value
    pub fn set_camera(&mut self, camera: crate::camera::Camera) {
        *self.renderer.camera_mut() = camera;
    }

    /// Get the current viewport size as (width, height)
    pub fn size(&self) -> (u32, u32) {
        self.renderer.size
    }

    /// Get the surface texture format
    /// Useful for creating render pipelines that need to match the surface format
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.renderer.config.format
    }

    /// Get references to the wgpu device and queue for creating GPU resources
    ///
    /// # Returns
    /// A tuple of (&Device, &Queue) for creating buffers, pipelines, etc.
    pub fn wgpu_resources(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.renderer.device, &self.renderer.queue)
    }

    /// Get a reference to the scene
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Get a mutable reference to the scene
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Set the scene to a new value
    pub fn set_scene(&mut self, scene: Scene) {
        self.scene = scene;
    }

    /// Get a reference to the selection manager
    pub fn selection(&self) -> &SelectionManager {
        &self.selection
    }

    /// Get a mutable reference to the selection manager
    pub fn selection_mut(&mut self) -> &mut SelectionManager {
        &mut self.selection
    }

    /// Get a reference to the event dispatcher
    pub fn dispatcher(&self) -> &EventDispatcher {
        &self.dispatcher
    }

    /// Get a mutable reference to the event dispatcher
    pub fn dispatcher_mut(&mut self) -> &mut EventDispatcher {
        &mut self.dispatcher
    }

    /// Get a reference to the operator manager
    pub fn operator_manager(&self) -> &OperatorManager {
        &self.operator_manager
    }

    /// Get a mutable reference to the operator manager
    pub fn operator_manager_mut(&mut self) -> &mut OperatorManager {
        &mut self.operator_manager
    }

    /// Get mutable references to both the operator manager and event dispatcher
    ///
    /// This method allows you to access both components simultaneously, which is
    /// necessary for operations like adding, removing, or reordering operators.
    /// Many OperatorManager methods (push_front, push_back, remove, move_to_front,
    /// move_to_back, swap) require both references to maintain synchronization
    /// between operator priority and callback ordering.
    ///
    /// # Example
    /// ```no_run
    /// # use wgpu_engine::Viewer;
    /// # fn example(viewer: &mut Viewer, id1: u32, id2: u32) {
    /// let (op_mgr, dispatcher) = viewer.operator_manager_and_dispatcher_mut();
    /// op_mgr.swap(id1, id2, dispatcher);
    /// # }
    /// ```
    pub fn operator_manager_and_dispatcher_mut(&mut self) -> (&mut OperatorManager, &mut EventDispatcher) {
        (&mut self.operator_manager, &mut self.dispatcher)
    }

    /// Render the scene using the default rendering path
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        self.renderer.render(&mut self.scene, Some(&self.selection))
    }

    /// Render the 3D scene with a custom overlay
    ///
    /// This method provides a generic way to render any overlay on top of the 3D scene.
    /// The callback receives references to the device, queue, encoder, and texture view,
    /// allowing you to create additional render passes for UI, post-processing, etc.
    ///
    /// # Example
    /// ```no_run
    /// # use wgpu_engine::Viewer;
    /// # fn example(viewer: &mut Viewer) {
    /// viewer.render_with_overlay(|device, queue, encoder, view| {
    ///     // Create your custom render pass here
    ///     // For example, render egui, ImGui, debug overlays, etc.
    ///     let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    ///         label: Some("Custom Overlay"),
    ///         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
    ///             view,
    ///             resolve_target: None,
    ///             ops: wgpu::Operations {
    ///                 load: wgpu::LoadOp::Load,  // Load existing 3D content
    ///                 store: wgpu::StoreOp::Store,
    ///             },
    ///             depth_slice: None,
    ///         })],
    ///         depth_stencil_attachment: None,
    ///         occlusion_query_set: None,
    ///         timestamp_writes: None,
    ///     });
    ///     // ... render your overlay ...
    /// }).unwrap();
    /// # }
    /// ```
    pub fn render_with_overlay<F>(&mut self, overlay_fn: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView),
    {
        // Get surface texture
        let output = self.renderer.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = self.renderer.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder with Overlay"),
            },
        );

        // Render 3D scene first
        self.renderer.prepare_scene(&mut self.scene).unwrap();
        self.renderer
            .render_scene_to_view(&view, &mut encoder, &mut self.scene, Some(&self.selection))?;

        // Call user's overlay function
        overlay_fn(
            &self.renderer.device,
            &self.renderer.queue,
            &mut encoder,
            &view,
        );

        // Submit and present
        self.renderer
            .queue
            .submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Render the 3D scene to a specific view using a specific encoder
    ///
    /// This is a low-level API for advanced rendering scenarios where you need
    /// full manual control over the render pipeline (e.g., multiple render targets,
    /// custom command buffer management, deferred rendering).
    ///
    /// For most overlay use cases, prefer `render_with_overlay()` which handles
    /// surface management and command submission automatically.
    pub fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), anyhow::Error> {
        self.renderer
            .render_scene_to_view(view, encoder, &mut self.scene, Some(&self.selection))
    }
}
