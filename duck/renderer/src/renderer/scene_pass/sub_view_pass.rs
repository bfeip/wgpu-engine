use wgpu::util::{BufferInitDescriptor, DeviceExt};

use super::super::batching::DrawData;
use super::super::gpu_resources::{CameraUniform, GpuTexture};
use super::super::pass_context::{FrameContext, SceneRenderPass};
use super::super::pipeline::MaterialPipelineCache;
use super::main_pass::draw_batches;

/// A camera uniform buffer + bind group dedicated to one sub-view slot.
///
/// Sub-views each need their own camera buffer: the renderer writes all camera
/// uniforms via `queue.write_buffer`, which executes on the queue timeline before
/// any recorded pass, so a single shared buffer cannot carry per-pass values.
/// Distinct buffers per slot are written independently with no aliasing.
// TODO: Basically a duplicate of camera resources. Remove ASAP.
struct CameraSlot {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl CameraSlot {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Sub-View Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("sub_view_camera_bind_group"),
        });
        Self { buffer, bind_group }
    }
}

/// Draws each sub-view in its own region of the surface, after the main view.
///
/// For every [`SubViewDraw`](super::super::batching) the pass:
/// - writes that sub-view's camera into a dedicated slot buffer,
/// - opens a render pass that *loads* the existing color (so the main view shows
///   through) and *clears* this pass's private depth buffer,
/// - confines drawing to the sub-view's pixel rectangle via `set_viewport` +
///   `set_scissor_rect`,
/// - binds the slot camera at group 0 and draws the subtree's batches.
///
/// Because `LoadOp::Clear` clears the whole depth attachment (WebGPU has no
/// sub-rect clear), the viewport+scissor confine all draws to the sub-rect; the
/// shared depth buffer is intentionally not reused so sub-views depth-test only
/// among their own geometry.
///
/// Limitations (v1): sub-views share the main scene's lights and IBL; the region
/// is not cleared to a background, so gaps between sub-view geometry reveal the
/// main view beneath.
pub(crate) struct SubViewPass {
    depth: GpuTexture,
    camera_layout: wgpu::BindGroupLayout,
    slots: Vec<CameraSlot>,
}

impl SubViewPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
        camera_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        Self {
            depth: GpuTexture::depth_sized(device, width, height, sample_count, "sub_view_depth_texture"),
            camera_layout: camera_layout.clone(),
            slots: Vec::new(),
        }
    }

    /// Ensures at least `count` camera slots exist.
    fn ensure_slots(&mut self, device: &wgpu::Device, count: usize) {
        while self.slots.len() < count {
            self.slots.push(CameraSlot::new(device, &self.camera_layout));
        }
    }
}

impl SceneRenderPass for SubViewPass {
    fn is_active(&self, draw_data: &DrawData) -> bool {
        draw_data.has_sub_views()
    }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        self.depth = GpuTexture::depth_sized(device, size.0, size.1, sample_count, "sub_view_depth_texture");
    }

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut MaterialPipelineCache,
        draw_data: &DrawData,
    ) {
        let sub_views = draw_data.sub_views();
        self.ensure_slots(ctx.device, sub_views.len());

        for (i, sv) in sub_views.iter().enumerate() {
            // Write this sub-view's camera into its dedicated slot buffer.
            let uniform = CameraUniform::from_positioned_camera(&sv.camera);
            ctx.queue
                .write_buffer(&self.slots[i].buffer, 0, bytemuck::cast_slice(&[uniform]));

            let (x, y, w, h) = sv.rect.to_pixels(ctx.size.0, ctx.size.1);
            let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sub-View Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
            render_pass.set_scissor_rect(x, y, w, h);

            // Bind this sub-view's camera at group 0; share lights/IBL with the scene.
            render_pass.set_bind_group(0, &self.slots[i].bind_group, &[]);
            render_pass.set_bind_group(1, ctx.lights_bind_group, &[]);
            if let Some(ibl) = ctx.ibl_bind_group {
                render_pass.set_bind_group(3, ibl, &[]);
            }

            draw_batches(&mut render_pass, &sv.batches, ctx, pipeline_cache, true);
        }
    }
}
