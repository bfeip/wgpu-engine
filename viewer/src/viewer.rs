use cgmath::Vector3;
use web_time::Instant;

use std::cell::RefCell;
use std::rc::Rc;

use crate::{
    common::RgbaColor,
    event::{Event, EventContext, EventDispatcher, EventKind},
    operator::{
        BuiltinOperatorId, NavigationMode, NavigationOperator, OperatorManager,
        SelectionOperator, TransformOperator,
    },
    scene::{Camera, Scene},
    selection::SelectionManager,
    selection_query::SelectionQuery,
};

use wgpu_engine_renderer::Renderer;

/// Main viewer that encapsulates the renderer, scene, and event handling
pub struct Viewer<'a> {
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    camera: Camera,
    scene: Scene,
    selection: SelectionManager,
    dispatcher: EventDispatcher,
    operator_manager: OperatorManager,
    navigation_mode: Rc<RefCell<NavigationMode>>,
    /// Current cursor position in screen coordinates
    cursor_position: Option<(f32, f32)>,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
}

impl<'a> Viewer<'a> {
    /// Create a new Viewer with the given surface target
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        // Create wgpu instance
        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::util::new_instance_with_webgpu_detection(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        }).await;

        let surface = instance.create_surface(surface_target).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let is_gl_backend = adapter.get_info().backend == wgpu::Backend::Gl;
        let has_compute = !is_gl_backend;

        if cfg!(target_arch = "wasm32") {
            if is_gl_backend {
                log::info!("WebGPU not available, falling back to WebGL.");
            } else {
                log::info!("Using WebGPU backend.");
            }
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: if is_gl_backend {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        // Use 4x MSAA on native backends, disable on GL
        let sample_count = if is_gl_backend { 1 } else { 4 };

        let renderer = Renderer::new(device, queue, surface_format, width, height, sample_count, has_compute);
        let mut scene = Scene::new();

        // Set up default lighting
        scene.set_lights(vec![crate::scene::Light::point(
            Vector3::new(1.0, -1.0, 1.0),
            RgbaColor{ r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            100.0
        )]);

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

        // Navigation operator for orbit/pan/zoom/walk
        let nav_operator = NavigationOperator::new(BuiltinOperatorId::Navigation.into());
        let navigation_mode = nav_operator.mode_handle();
        operator_manager.push_back(Box::new(nav_operator), &mut dispatcher);

        let camera = Camera {
            eye: (0.0, 0.1, 0.2).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: width as f32 / height as f32,
            fovy: 45.0,
            znear: 0.001,
            zfar: 100.0,
            ortho: false,
        };

        // Create viewer
        let mut viewer = Self {
            surface,
            surface_config,
            renderer,
            camera,
            scene,
            selection: SelectionManager::new(),
            dispatcher,
            operator_manager,
            navigation_mode,
            cursor_position: None,
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
        // Register CursorMoved handler to track cursor position
        self.dispatcher
            .register(EventKind::CursorMoved, |event, ctx| {
                if let Event::CursorMoved { position } = event {
                    *ctx.cursor_position = Some((position.0 as f32, position.1 as f32));
                }
                false // Don't stop propagation - other handlers may need cursor position too
            });
    }

    /// Handle a single event by dispatching it to registered handlers
    pub fn handle_event(&mut self, event: &Event) {
        // Handle resize at the viewer level (needs surface, renderer, and camera)
        if let Event::Resized(physical_size) = event {
            let (w, h) = *physical_size;
            if w > 0 && h > 0 {
                self.surface_config.width = w;
                self.surface_config.height = h;
                self.surface.configure(&self.renderer.device(), &self.surface_config);
                self.renderer.resize(*physical_size);
                self.camera.aspect = w as f32 / h as f32;
            }
        }

        let mut ctx = EventContext {
            camera: &mut self.camera,
            size: self.renderer.size(),
            cursor_position: &mut self.cursor_position,
            scene: &mut self.scene,
            selection: &mut self.selection,
        };
        self.dispatcher.dispatch(event, &mut ctx);
    }

    /// Get the current navigation mode.
    pub fn navigation_mode(&self) -> NavigationMode {
        *self.navigation_mode.borrow()
    }

    /// Set the navigation mode (Orbit or Walk).
    pub fn set_navigation_mode(&mut self, mode: NavigationMode) {
        *self.navigation_mode.borrow_mut() = mode;
    }

    /// Dispatch an Update event with delta_time since last update.
    ///
    /// Call this once per frame before rendering to enable smooth continuous
    /// operations like WASD movement in walk mode.
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
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Get a mutable reference to the current camera
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Set the camera to a new value
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    /// Get the current viewport size as (width, height)
    pub fn size(&self) -> (u32, u32) {
        self.renderer.size()
    }

    /// Get the surface texture format
    /// Useful for creating render pipelines that need to match the surface format
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.renderer.surface_format()
    }

    /// Get references to the wgpu device and queue for creating GPU resources
    ///
    /// # Returns
    /// A tuple of (&Device, &Queue) for creating buffers, pipelines, etc.
    pub fn wgpu_resources(&self) -> (&wgpu::Device, &wgpu::Queue) {
        (&self.renderer.device(), &self.renderer.queue())
    }

    /// Get a reference to the scene
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Get a mutable reference to the scene
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Set the scene to a new value.
    ///
    /// This clears all scene-specific GPU resources and selection state
    /// to prevent stale data from persisting across scene changes.
    pub fn set_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.selection.clear();
        self.renderer.clear_gpu_resources();
    }

    /// Clear the scene, removing all geometry, materials, textures, and
    /// associated GPU resources.
    ///
    /// This is the recommended way to reset the viewer before loading
    /// new content. It clears:
    /// - All scene nodes, instances, meshes, materials (except default), textures
    /// - All cached GPU resources (vertex buffers, texture views, material bind groups)
    /// - The current selection
    pub fn clear_scene(&mut self) {
        self.scene.clear();
        self.selection.clear();
        self.renderer.clear_gpu_resources();
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
    /// # use wgpu_engine_viewer::Viewer;
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
        self.renderer.prepare_scene(&self.camera, &mut self.scene)?;

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.renderer.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") },
        );

        self.renderer.render_scene_to_view(&view, &mut encoder, &self.camera, &self.scene, Self::selection_for_render(&self.selection))?;

        self.renderer.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Render the 3D scene with a custom overlay
    ///
    /// This method provides a generic way to render any overlay on top of the 3D scene.
    /// The callback receives references to the device, queue, encoder, and texture view,
    /// allowing you to create additional render passes for UI, post-processing, etc.
    ///
    /// # Example
    /// ```no_run
    /// # use wgpu_engine_viewer::Viewer;
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
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = self.renderer.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder with Overlay"),
            },
        );

        // Render 3D scene first
        self.renderer.prepare_scene(&self.camera, &mut self.scene).unwrap();
        self.renderer
            .render_scene_to_view(&view, &mut encoder, &self.camera, &self.scene, Self::selection_for_render(&self.selection))?;

        // Call user's overlay function
        overlay_fn(
            &self.renderer.device(),
            &self.renderer.queue(),
            &mut encoder,
            &view,
        );

        // Submit and present
        self.renderer
            .queue()
            .submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Returns a selection query for the renderer if outline rendering is enabled.
    fn selection_for_render(selection: &SelectionManager) -> Option<&dyn SelectionQuery> {
        if selection.config().outline_enabled {
            Some(selection)
        } else {
            None
        }
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
            .render_scene_to_view(view, encoder, &self.camera, &self.scene, Self::selection_for_render(&self.selection))
    }
}
