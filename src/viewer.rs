use std::time::Instant;

use crate::{
    annotation::AnnotationManager,
    drawstate::DrawState,
    event::{Event, EventContext, EventDispatcher, EventKind},
    operator::{BuiltinOperatorId, NavigationOperator, OperatorManager, SelectionOperator, WalkOperator},
    scene::Scene,
};

/// Main viewer that encapsulates the rendering state, scene, and event handling
pub struct Viewer<'a> {
    state: DrawState<'a>,
    pub scene: Scene,
    pub dispatcher: EventDispatcher,
    pub operator_manager: OperatorManager,
    pub annotation_manager: AnnotationManager,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
}

impl<'a> Viewer<'a> {
    /// Create a new Viewer with the given surface target
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let mut state = DrawState::new(surface_target, width, height).await;

        // Load default scene (TODO: make this configurable)
        let aspect = width as f32 / height as f32;
        let load_result = crate::gltf::load_gltf_scene(
            "/home/zachary/src/glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf",
            aspect,
        )
        .unwrap();

        let mut scene = load_result.scene;

        // Use camera from glTF if available, otherwise fit camera to scene bounds
        if let Some(camera) = load_result.camera {
            state.camera = camera;
        } else if let Some(bounds) = scene.bounding() {
            state.camera.fit_to_bounds(&bounds);
        }

        // Set up default lighting
        scene.lights = vec![crate::scene::Light::new(
            cgmath::Vector3 {
                x: 3.,
                y: 3.,
                z: 3.,
            },
            crate::common::RgbaColor {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        )];

        let annotation_manager = AnnotationManager::new();

        let mut dispatcher = EventDispatcher::new();
        let mut operator_manager = OperatorManager::new();

        // Add operators in priority order (first added = highest priority)
        // Selection operator handles picking and must be first
        let selection_operator =
            Box::new(SelectionOperator::new(BuiltinOperatorId::Selection.into()));
        operator_manager.push_back(selection_operator, &mut dispatcher);

        // Walk operator for WASD movement
        let walk_operator = Box::new(WalkOperator::new(
            BuiltinOperatorId::Walk.into(),
        ));
        operator_manager.push_back(walk_operator, &mut dispatcher);

        // Navigation operator for orbit/pan/zoom
        let nav_operator = Box::new(NavigationOperator::new(
            BuiltinOperatorId::Navigation.into(),
        ));
        operator_manager.push_back(nav_operator, &mut dispatcher);

        // Create viewer
        let mut viewer = Self {
            state,
            scene,
            dispatcher,
            operator_manager,
            annotation_manager,
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
                ctx.state.resize(*physical_size);
            }
            true
        });

        // Register CursorMoved handler to track cursor position
        self.dispatcher
            .register(EventKind::CursorMoved, |event, ctx| {
                if let Event::CursorMoved { position } = event {
                    ctx.state.cursor_position = Some((position.0 as f32, position.1 as f32));
                }
                false // Don't stop propagation - other handlers may need cursor position too
            });
    }

    /// Handle a single event by dispatching it to registered handlers
    pub fn handle_event(&mut self, event: &Event) {
        let mut ctx = EventContext {
            state: &mut self.state,
            scene: &mut self.scene,
            annotation_manager: &mut self.annotation_manager,
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
        &self.state.camera
    }

    /// Get a mutable reference to the current camera
    pub fn camera_mut(&mut self) -> &mut crate::camera::Camera {
        &mut self.state.camera
    }

    /// Get the current viewport size as (width, height)
    pub fn size(&self) -> (u32, u32) {
        self.state.size
    }

    /// Get the surface texture format
    /// Useful for creating render pipelines that need to match the surface format
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.state.config.format
    }

    /// Get references to the wgpu device and queue for creating GPU resources
    ///
    /// # Returns
    /// A tuple of (&Device, &Queue) for creating buffers, pipelines, etc.
    pub fn wgpu_resources(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.state.device, &self.state.queue)
    }

    /// Render the scene using the default rendering path
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        self.state.render(&mut self.scene)
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
        let output = self.state.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = self.state.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder with Overlay"),
            },
        );

        // Render 3D scene first
        self.state.prepare_scene(&mut self.scene).unwrap();
        self.state
            .render_scene_to_view(&view, &mut encoder, &mut self.scene)?;

        // Call user's overlay function
        overlay_fn(
            &self.state.device,
            &self.state.queue,
            &mut encoder,
            &view,
        );

        // Submit and present
        self.state
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
    ///
    /// # Example
    /// ```no_run
    /// # use wgpu_engine::Viewer;
    /// # fn example(viewer: &mut Viewer) -> Result<(), anyhow::Error> {
    /// // This example shows manual control - not recommended for most use cases
    /// # Ok(())
    /// # }
    /// ```
    pub fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), anyhow::Error> {
        self.state
            .render_scene_to_view(view, encoder, &mut self.scene)
    }
}
