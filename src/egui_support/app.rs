use std::sync::Arc;

use anyhow::Result;
use egui_wgpu::RendererOptions;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

use crate::winit_support;
use crate::Viewer;

use super::render::render_egui_overlay;

/// High-level application wrapper that manages window, viewer, and egui state.
///
/// This provides a simple API for creating applications with egui UI overlays
/// on top of a 3D scene. It owns the window, viewer, and all egui state,
/// handling event routing and rendering automatically.
///
/// # Example
///
/// ```rust,no_run
/// use wgpu_engine::egui_support::EguiViewerApp;
/// use winit::event_loop::{EventLoop, ActiveEventLoop};
/// use winit::application::ApplicationHandler;
///
/// struct App<'a> {
///     viewer_app: Option<EguiViewerApp<'a>>,
/// }
///
/// impl<'a> ApplicationHandler for App<'a> {
///     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
///         if self.viewer_app.is_none() {
///             self.viewer_app = Some(pollster::block_on(
///                 EguiViewerApp::new(event_loop)
///             ));
///         }
///     }
///
///     fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
///         let app = self.viewer_app.as_mut().unwrap();
///         match event {
///             WindowEvent::CloseRequested => event_loop.exit(),
///             WindowEvent::RedrawRequested => {
///                 app.render(|ctx, viewer| {
///                     egui::CentralPanel::default().show(ctx, |ui| {
///                         ui.label("Hello World!");
///                     });
///                 }).unwrap();
///                 app.window().request_redraw();
///             }
///             _ => {
///                 app.handle_window_event(&event);
///             }
///         }
///     }
/// }
/// ```
pub struct EguiViewerApp<'a> {
    window: Arc<Window>,
    viewer: Viewer<'a>,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

impl<'a> EguiViewerApp<'a> {
    /// Create a new EguiViewerApp with default window settings.
    ///
    /// This creates a window with default attributes and initializes
    /// the viewer and egui state.
    ///
    /// # Arguments
    ///
    /// * `event_loop` - The active event loop (from winit's `resumed` callback)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use wgpu_engine::egui_support::EguiViewerApp;
    ///
    /// let app = pollster::block_on(EguiViewerApp::new(event_loop));
    /// ```
    pub async fn new(event_loop: &ActiveEventLoop) -> Self {
        Self::with_window_attrs(event_loop, Window::default_attributes()).await
    }

    /// Create a new EguiViewerApp with custom window attributes.
    ///
    /// This allows you to customize window settings like title, size, etc.
    ///
    /// # Arguments
    ///
    /// * `event_loop` - The active event loop (from winit's `resumed` callback)
    /// * `window_attrs` - Window attributes for customization
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use wgpu_engine::egui_support::EguiViewerApp;
    /// use winit::window::Window;
    /// use winit::dpi::LogicalSize;
    ///
    /// let attrs = Window::default_attributes()
    ///     .with_title("My 3D App")
    ///     .with_inner_size(LogicalSize::new(1280, 720));
    ///
    /// let app = pollster::block_on(
    ///     EguiViewerApp::with_window_attrs(event_loop, attrs)
    /// );
    /// ```
    pub async fn with_window_attrs(
        event_loop: &ActiveEventLoop,
        window_attrs: WindowAttributes,
    ) -> Self {
        // Create window
        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        // Create viewer
        let size = window.inner_size();
        let viewer = Viewer::new(Arc::clone(&window), size.width, size.height).await;

        // Initialize egui context
        let egui_ctx = egui::Context::default();

        // Initialize egui winit state for event handling
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // Initialize egui wgpu renderer
        let (device, _queue) = viewer.wgpu_resources();
        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            viewer.surface_format(),
            RendererOptions::default(),
        );

        Self {
            window,
            viewer,
            egui_ctx,
            egui_winit,
            egui_renderer,
        }
    }

    /// Get a reference to the viewer.
    ///
    /// Use this to access the scene, camera, operators, etc.
    pub fn viewer(&self) -> &Viewer<'a> {
        &self.viewer
    }

    /// Get a mutable reference to the viewer.
    ///
    /// Use this to modify the scene, camera, operators, etc.
    pub fn viewer_mut(&mut self) -> &mut Viewer<'a> {
        &mut self.viewer
    }

    /// Get a reference to the window.
    ///
    /// Use this to request redraws, get window size, etc.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Handle a window event.
    ///
    /// This processes the event through egui first, then forwards it to the
    /// viewer if egui didn't consume it. This ensures proper event priority.
    ///
    /// # Arguments
    ///
    /// * `event` - The window event to handle
    ///
    /// # Returns
    ///
    /// `true` if egui consumed the event, `false` otherwise
    pub fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        // Give egui first chance to handle the event
        let response = self.egui_winit.on_window_event(&self.window, event);

        // Forward to viewer if egui didn't consume it
        if !response.consumed {
            if let Some(app_event) = winit_support::convert_window_event(event.clone()) {
                self.viewer.handle_event(&app_event);
            }
        }

        response.consumed
    }

    /// Handle a device event.
    ///
    /// Device events (like raw mouse motion) are not consumed by egui,
    /// so they're always forwarded to the viewer.
    ///
    /// # Arguments
    ///
    /// * `event` - The device event to handle
    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let Some(app_event) = winit_support::convert_device_event(event.clone()) {
            self.viewer.handle_event(&app_event);
        }
    }

    /// Render a frame with the provided UI function.
    ///
    /// This handles the complete rendering pipeline:
    /// 1. Dispatches per-frame update events to the viewer
    /// 2. Collects egui input from winit
    /// 3. Runs the UI function to build the egui UI
    /// 4. Handles egui platform output (clipboard, cursor, etc.)
    /// 5. Renders the 3D scene
    /// 6. Renders the egui overlay on top
    ///
    /// The UI function receives the egui context and a mutable reference to
    /// the viewer. It can build egui UI and also capture/mutate external state
    /// directly (since egui runs in immediate mode).
    ///
    /// # Arguments
    ///
    /// * `ui_fn` - A closure that builds the egui UI
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if rendering fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// app.render(|ctx, viewer| {
    ///     egui::Window::new("Info").show(ctx, |ui| {
    ///         ui.label(format!("FPS: {:.1}", ctx.input(|i| i.stable_dt).recip()));
    ///         ui.label(format!("Meshes: {}", viewer.scene().meshes.len()));
    ///     });
    /// })?;
    /// ```
    pub fn render<F>(&mut self, mut ui_fn: F) -> Result<()>
    where
        F: FnMut(&egui::Context, &mut Viewer<'a>),
    {
        // Dispatch per-frame update event (for WASD movement, etc.)
        self.viewer.update();

        // Take egui input from winit
        let raw_input = self.egui_winit.take_egui_input(&self.window);

        // Run egui frame with user's UI function
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            ui_fn(ctx, &mut self.viewer);
        });

        // Handle egui platform output (clipboard, cursor shape, etc.)
        self.egui_winit.handle_platform_output(
            &self.window,
            full_output.platform_output.clone(),
        );

        // Get rendering parameters
        let viewer_size = self.viewer.size();
        let scale_factor = self.window.scale_factor() as f32;

        // Render 3D scene + egui overlay
        self.viewer.render_with_overlay(|device, queue, encoder, view| {
            render_egui_overlay(
                &mut self.egui_renderer,
                &self.egui_ctx,
                &full_output,
                viewer_size,
                scale_factor,
                device,
                queue,
                encoder,
                view,
            );
        })
    }
}
