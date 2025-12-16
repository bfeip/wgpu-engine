/// Internal rendering utilities for egui overlay.
///
/// This module contains the low-level rendering logic for rendering egui
/// on top of the 3D scene. This is not part of the public API.

/// Render egui overlay on top of the 3D scene.
///
/// This function handles the complete egui rendering pipeline:
/// 1. Updates egui textures that have changed
/// 2. Tessellates egui shapes into GPU primitives
/// 3. Updates GPU buffers with vertex/index data
/// 4. Creates a render pass with LoadOp::Load to preserve 3D content
/// 5. Renders the egui primitives
/// 6. Frees textures marked for deletion
///
/// # Arguments
///
/// * `egui_renderer` - The egui wgpu renderer
/// * `egui_ctx` - The egui context (for tessellation)
/// * `full_output` - The output from egui's frame (shapes, textures, etc.)
/// * `viewer_size` - The viewport size in pixels (width, height)
/// * `scale_factor` - The window scale factor for high-DPI displays
/// * `device` - The wgpu device
/// * `queue` - The wgpu queue
/// * `encoder` - The command encoder to record rendering commands
/// * `view` - The texture view to render to
pub(crate) fn render_egui_overlay(
    egui_renderer: &mut egui_wgpu::Renderer,
    egui_ctx: &egui::Context,
    full_output: &egui::FullOutput,
    viewer_size: (u32, u32),
    scale_factor: f32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    // Update egui textures that have changed
    for (id, image_delta) in &full_output.textures_delta.set {
        egui_renderer.update_texture(device, queue, *id, image_delta);
    }

    // Tessellate egui shapes into GPU primitives
    let clipped_primitives = egui_ctx.tessellate(
        full_output.shapes.clone(),
        full_output.pixels_per_point,
    );

    // Create screen descriptor for buffer updates
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [viewer_size.0, viewer_size.1],
        pixels_per_point: scale_factor,
    };

    // Update GPU buffers with vertex and index data
    egui_renderer.update_buffers(
        device,
        queue,
        encoder,
        &clipped_primitives,
        &screen_descriptor,
    );

    // Create render pass for egui overlay
    {
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Load existing 3D scene content
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Render egui primitives on top of 3D content
        egui_renderer.render(
            &mut render_pass.forget_lifetime(),
            &clipped_primitives,
            &screen_descriptor,
        );
    }

    // Free textures that are no longer needed
    for id in &full_output.textures_delta.free {
        egui_renderer.free_texture(id);
    }
}
