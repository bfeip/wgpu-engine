use super::common::{bind_scene_groups, draw_batches};
use super::super::batching::DrawData;
use super::super::gpu_resources::GpuTexture;
use super::super::pass_context::{FrameContext, SceneRenderPass};
use super::super::pipeline::PipelineCache;

/// Pass 1: Main scene render.
///
/// Clears color/depth/stencil, binds camera/lights/IBL, runs a depth pre-pass for
/// `Blend`-mode materials, then draws all batches with pipeline caching.
pub(crate) struct MainPass;

impl SceneRenderPass for MainPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D Scene Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.04,
                        g: 0.04,
                        b: 0.04,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.renderer_textures.depth.view,
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

        bind_scene_groups(&mut render_pass, ctx);
        draw_batches(&mut render_pass, draw_data.all_batches(), ctx, pipeline_cache, true);
    }
}

/// Pass 2 (conditional): Overlay render.
///
/// Loads the existing color attachment, clears a separate depth buffer, and draws
/// always-on-top geometry so it depth-tests among itself but not against the scene.
/// Owns its own depth buffer so it can be independently resized.
pub(crate) struct OverlayPass {
    depth: GpuTexture,
}

impl OverlayPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        Self {
            depth: GpuTexture::depth_sized(device, width, height, sample_count, "overlay_depth_texture"),
        }
    }
}

impl SceneRenderPass for OverlayPass {
    fn is_active(&self, draw_data: &DrawData) -> bool {
        draw_data.has_overlay()
    }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        self.depth = GpuTexture::depth_sized(device, size.0, size.1, sample_count, "overlay_depth_texture");
    }

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Render Pass"),
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

        bind_scene_groups(&mut render_pass, ctx);
        draw_batches(&mut render_pass, draw_data.overlay_batches(), ctx, pipeline_cache, false);
    }
}
