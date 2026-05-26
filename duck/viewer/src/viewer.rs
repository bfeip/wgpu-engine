use duck_engine_common::Vector3;
use duck_engine_scene::{NodeFlags, NodeId};
use web_time::Instant;

#[cfg(feature = "streaming")]
use crate::streaming::ViewerStreamClient;

use crate::{
    event::{Event, EventContext, EventDispatcher},
    scene::{NodePayload, PositionedCamera, Scene},
    selection::SelectionManager,
    renderer::{Renderer, HighlightQuery},
};

/// Main viewer that encapsulates the renderer, scene, and event handling
pub struct Viewer<'a> {
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    scene: Scene,
    selection: SelectionManager,
    dispatcher: EventDispatcher,
    /// Current cursor position in screen coordinates
    cursor_position: Option<(f32, f32)>,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
    #[cfg(feature = "streaming")]
    stream_client: Option<ViewerStreamClient>,
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

        let renderer = Renderer::new(device.clone(), queue.clone(), surface_format, width, height, sample_count, has_compute);
        let scene = Scene::new();

        let dispatcher = EventDispatcher::new();

        // Create viewer
        let mut viewer = Self {
            surface,
            surface_config,
            device,
            queue,
            renderer,
            scene,
            selection: SelectionManager::new(),
            dispatcher,
            cursor_position: None,
            last_update_time: None,
            #[cfg(feature = "streaming")]
            stream_client: None,
        };

        // Ensure there is always an active camera in the scene
        viewer.ensure_active_camera();

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

    /// Handle a single event by dispatching it to registered handlers
    pub fn handle_event(&mut self, event: &Event) {
        if let Event::CursorMoved { position } = event {
            self.cursor_position = Some((position.0 as f32, position.1 as f32));
        }
        // Handle resize at the viewer level (needs surface, renderer, and camera)
        if let Event::Resized(physical_size) = event {
            let (w, h) = *physical_size;
            if w > 0 && h > 0 {
                self.surface_config.width = w;
                self.surface_config.height = h;
                self.surface.configure(&self.device, &self.surface_config);
                self.renderer.resize(*physical_size);

            }
        }

        let mut ctx = EventContext {
            size: self.renderer.size(),
            cursor_position: &mut self.cursor_position,
            scene: &mut self.scene,
            selection: &mut self.selection,
            modifiers: Default::default(), // dispatcher overwrites this in dispatch()
        };
        self.dispatcher.dispatch(event, &mut ctx);
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

        #[cfg(feature = "streaming")]
        self.poll_stream();
    }

    /// Connect to a streaming server. Replaces any existing connection.
    #[cfg(feature = "streaming")]
    pub fn connect_stream(&mut self, addr: &str) -> anyhow::Result<()> {
        use duck_engine_streaming::SubscribeOptions;
        let camera = self.scene.active_camera_positioned(1.0).map(|cam| {
            let fwd = cam.forward();
            duck_engine_streaming::CameraHint {
                position: cam.eye.into(),
                forward: fwd.into(),
                fov_y_rad: cam.fovy.to_radians(),
            }
        });
        let client = ViewerStreamClient::connect(addr, SubscribeOptions { camera, ..Default::default() })?;
        self.stream_client = Some(client);
        Ok(())
    }

    /// Disconnect from the streaming server.
    #[cfg(feature = "streaming")]
    pub fn disconnect_stream(&mut self) {
        self.stream_client = None;
    }

    /// Returns `true` once the initial priority sync from the server is complete.
    #[cfg(feature = "streaming")]
    pub fn stream_sync_complete(&self) -> bool {
        self.stream_client.as_ref().map(|c| c.sync_complete).unwrap_or(false)
    }

    /// Drain pending scene updates from the streaming client. Called every frame from `update`.
    #[cfg(feature = "streaming")]
    fn poll_stream(&mut self) {
        use crate::streaming::PollResult;
        let result = self.stream_client.as_mut().map(|c| c.poll(&mut self.scene));
        match result {
            Some(PollResult::Disconnected) => { self.stream_client = None; }
            _ => {}
        }
    }

    /// Get a reference to the active camera.
    ///
    /// Panics if the scene has no active camera node.
    pub fn camera(&self) -> PositionedCamera {
        let (w, h) = self.renderer.size();
        let aspect = if h > 0 { w as f32 / h as f32 } else { 16.0 / 9.0 };
        self.scene.active_camera_positioned(aspect).expect("no active camera in scene")
    }

    /// Clones the active camera, passes it to `f` for mutation, then writes it back.
    pub fn with_camera_mut<F: FnOnce(&mut PositionedCamera)>(&mut self, f: F) {
        let mut cam = self.camera();
        f(&mut cam);
        self.set_camera(cam);
    }

    /// Replace the active camera.
    pub fn set_camera(&mut self, camera: PositionedCamera) {
        let id = self.scene.active_camera().expect("no active camera in scene");
        self.scene.set_node_transform(id, camera.to_node_transform());
        self.scene.set_node_payload(id, NodePayload::Camera(camera.projection()));
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

    /// Replace the active rendering workflow.
    pub fn set_workflow(&mut self, workflow: Box<dyn crate::renderer::RenderWorkflow>) {
        self.renderer.set_workflow(workflow);
    }

    /// Create a [`ShadedWorkflow`](crate::renderer::ShadedWorkflow) configured for this viewer.
    pub fn shaded_workflow(&mut self) -> crate::renderer::ShadedWorkflow {
        self.renderer.shaded_workflow()
    }

    /// Create a [`HiddenLineWorkflow`](crate::renderer::HiddenLineWorkflow) configured for this viewer.
    pub fn hidden_line_workflow(&mut self, config: crate::renderer::HiddenLineConfig) -> crate::renderer::HiddenLineWorkflow {
        self.renderer.hidden_line_workflow(config)
    }

    /// Get a reference to the wgpu device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the wgpu queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
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
    /// If the incoming scene has no active camera a default one is added.
    pub fn set_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.selection.clear();
        self.renderer.clear_gpu_resources();
        self.ensure_active_camera();
        self.ensure_default_lights();
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
        self.ensure_active_camera();
    }

    /// Adds a default camera node to the scene if no active camera is set.
    fn ensure_active_camera(&mut self) -> NodeId {
        if let Some(camera_id) = self.scene.active_camera() {
            return camera_id;
        }
        let cam = PositionedCamera {
            eye: (0.0, 0.1, 0.2).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vector3::unit_y(),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.001,
            zfar: 100.0,
            ortho: false,
        };
        let id = self.scene.add_node(None, Some("Camera".to_string()), cam.to_node_transform(), NodeFlags::NONE)
            .expect("Failed to add default camera node");
        self.scene.set_node_payload(id, NodePayload::Camera(cam.projection()));
        self.scene.set_active_camera(Some(id));
        return id;
    }

    /// Adds default lights as children of the camera if the scene is otherwise unlit.
    fn ensure_default_lights(&mut self) {
        if self.scene.has_light_nodes() || self.scene.active_environment_map().is_some() {
            // If the scene has any lights or an environment map we don't add defaults
            return;
        }
        let camera = self.ensure_active_camera();
        self.scene.set_default_light_nodes(camera);
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

    /// Render the scene using the default rendering path
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        let (output, _view, encoder) = self.render_scene()?;
        self.present(encoder, output);
        Ok(())
    }

    /// Prepare and render the 3D scene, returning the surface output, view, and
    /// command encoder for further rendering (overlays, post-processing, etc.).
    ///
    /// Call [`present()`](Self::present) when done to submit commands and display the frame.
    pub fn render_scene(&mut self) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView, wgpu::CommandEncoder), anyhow::Error> {
        self.renderer.prepare_scene(&mut self.scene)?;

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") },
        );

        let highlight = Self::selection_for_render(&self.selection);
        self.renderer.render_scene_to_view(&view, &mut encoder, None, &self.scene, highlight)?;

        Ok((output, view, encoder))
    }

    /// Submit the command encoder and present the surface texture.
    pub fn present(&self, encoder: wgpu::CommandEncoder, output: wgpu::SurfaceTexture) {
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Returns a highlight query for the renderer if outline rendering is enabled.
    fn selection_for_render(selection: &SelectionManager) -> Option<&dyn HighlightQuery> {
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
    /// For most overlay use cases, prefer `render_scene()` + `present()` which
    /// handle surface management and command submission automatically.
    pub fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), anyhow::Error> {
        let highlight = Self::selection_for_render(&self.selection);
        self.renderer.render_scene_to_view(view, encoder, None, &self.scene, highlight)
    }
}
