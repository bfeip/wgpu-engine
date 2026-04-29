use crate::scene::PrimitiveType;

use super::common::bind_scene_groups;
use super::super::batching::DrawData;
use super::super::gpu_resources::PipelineCacheKey;
use super::super::pass_context::{FrameContext, SceneRenderPass};
use super::super::pipeline::MaterialPipelineCache;

// ---------------------------------------------------------------------------
// HiddenLineEdgesPass — depth-tested edges for the hidden-line workflow
// ---------------------------------------------------------------------------

/// Hidden-line workflow edge pass.
///
/// Loads the existing color (written by the solid pass) and renders all
/// `LineList` primitives using the standard material pipeline. Lines whose depth
/// is greater than the solid geometry depth are occluded and not drawn.
pub(crate) struct HiddenLineEdgesPass;

impl SceneRenderPass for HiddenLineEdgesPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut MaterialPipelineCache,
        draw_data: &DrawData,
    ) {
        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Hidden Line Edges Pass"),
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
                view: ctx.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        bind_scene_groups(&mut render_pass, ctx);

        let mut current_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in draw_data.all_batches().iter().filter(|b| b.primitive_type == PrimitiveType::LineList) {
            let material = ctx.scene.get_material(batch.material_id).unwrap();
            let material_props = material.get_properties(batch.primitive_type);
            let material_gpu = ctx
                .gpu_resources
                .get_material(batch.material_id, batch.primitive_type)
                .expect("Material GPU resources not initialized");
            render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

            let pipeline_key = PipelineCacheKey {
                material_props: material_props.clone(),
                scene_props: ctx.scene_props.clone(),
                primitive_type: batch.primitive_type,
                depth_prepass: false,
            };
            if current_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = pipeline_cache.get_or_create(ctx.device, pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                current_pipeline_key = Some(pipeline_key);
            }
            ctx.draw_batch(&mut render_pass, batch);
        }
    }
}
