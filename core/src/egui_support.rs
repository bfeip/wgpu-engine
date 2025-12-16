//! egui integration support for wgpu-engine.
//!
//! This module provides a high-level wrapper ([`EguiViewerApp`]) that simplifies
//! creating applications with egui UI overlays on top of 3D scenes.
//!
//! # Features
//!
//! Enable the `egui-support` feature to use this module:
//!
//! ```toml
//! [dependencies]
//! wgpu-engine = { version = "0.1", features = ["egui-support"] }
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use wgpu_engine::egui_support::EguiViewerApp;
//! use winit::event_loop::EventLoop;
//! use winit::application::ApplicationHandler;
//!
//! struct App<'a> {
//!     viewer_app: Option<EguiViewerApp<'a>>,
//! }
//!
//! impl<'a> ApplicationHandler for App<'a> {
//!     fn resumed(&mut self, event_loop: &ActiveEventLoop) {
//!         if self.viewer_app.is_none() {
//!             self.viewer_app = Some(pollster::block_on(
//!                 EguiViewerApp::new(event_loop)
//!             ));
//!         }
//!     }
//!
//!     fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
//!         let app = self.viewer_app.as_mut().unwrap();
//!         match event {
//!             WindowEvent::RedrawRequested => {
//!                 app.render(|ctx, viewer| {
//!                     // Build your UI here
//!                     egui::CentralPanel::default().show(ctx, |ui| {
//!                         ui.label("Hello World!");
//!                     });
//!                 }).unwrap();
//!             }
//!             _ => {
//!                 app.handle_window_event(&event);
//!             }
//!         }
//!     }
//! }
//! ```

mod app;
mod render;

// Re-export public API
pub use app::EguiViewerApp;
