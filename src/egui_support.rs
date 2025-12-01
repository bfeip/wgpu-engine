//! egui integration for wgpu-engine
//!
//! This module provides utilities for rendering egui overlays on top of 3D scenes.
//! It only handles rendering - input and UI state management remain application responsibilities.

use crate::common::PhysicalSize;

/// Wrapper for egui_wgpu::Renderer that provides a simplified rendering interface
pub struct EguiRenderer {
    renderer: egui_wgpu::Renderer,
}

impl EguiRenderer {
    /// Creates a new egui renderer
    ///
    /// # Arguments
    /// * `device` - The WGPU device (obtain via `viewer.wgpu_resources()`)
    /// * `surface_format` - The surface format (obtain via `viewer.surface_format()`)
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self { renderer }
    }

    /// Render egui primitives to the given texture view
    ///
    /// This should be called after rendering the 3D scene but before submitting
    /// the command encoder. The render pass will use LoadOp::Load to preserve
    /// the existing 3D scene content.
    ///
    /// # Arguments
    /// * `device` - WGPU device reference
    /// * `queue` - WGPU queue reference
    /// * `encoder` - Command encoder (must have completed 3D scene rendering)
    /// * `view` - The same texture view used for the 3D scene
    /// * `screen_descriptor` - Screen size and scale information
    /// * `full_output` - egui's rendering output from `Context::run()`
    /// * `context` - egui context for tessellation
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        screen_descriptor: egui_wgpu::ScreenDescriptor,
        full_output: egui::FullOutput,
        context: &egui::Context,
    ) {
        // Update textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        // Tessellate and update buffers
        let clipped_primitives = context.tessellate(
            full_output.shapes,
            full_output.pixels_per_point,
        );
        self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        // Create render pass that loads existing content
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
                depth_stencil_attachment: None, // egui doesn't use depth
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &clipped_primitives,
                &screen_descriptor,
            );
        }

        // Free textures marked for deletion
        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

/// Helper to create a screen descriptor from viewer size and scale factor
///
/// # Arguments
/// * `size` - Viewer viewport size (from `viewer.size()`)
/// * `scale_factor` - Display scale factor (from `window.scale_factor()` on native)
pub fn screen_descriptor_from_size(
    size: PhysicalSize<u32>,
    scale_factor: f32,
) -> egui_wgpu::ScreenDescriptor {
    egui_wgpu::ScreenDescriptor {
        size_in_pixels: [size.width, size.height],
        pixels_per_point: scale_factor,
    }
}
