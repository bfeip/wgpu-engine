use duck_engine_scene::{AlphaMode, MaterialProperties, SceneProperties};

use crate::DrawBatch;
use crate::renderer::gpu_resources::{self, PipelineCacheKey};

use super::super::batching::DrawData;
use super::super::gpu_resources::GpuTexture;
use super::super::pass_context::{FrameContext, SceneRenderPass};
use super::super::pipeline::MaterialPipelineCache;

/// Bind the scene-level bind groups shared by all geometry passes:
/// - Group 0: Camera (view/proj + eye position)
/// - Group 1: Lights
/// - Group 3: IBL environment (when active)
pub(crate) fn bind_scene_groups<'r>(render_pass: &mut wgpu::RenderPass<'r>, ctx: &'r FrameContext<'r>) {
    render_pass.set_bind_group(0, ctx.camera_bind_group, &[]);
    render_pass.set_bind_group(1, ctx.lights_bind_group, &[]);
    if let Some(ibl) = ctx.ibl_bind_group {
        render_pass.set_bind_group(3, ibl, &[]);
    }
}

/// Draw a list of batches through the scene geometry pipeline.
///
/// If `with_depth_prepass` is true, a depth-only pre-pass for `Blend`-mode materials
/// is executed first so their opaque portions establish correct depth occlusion before
/// the main draw loop renders them with blending.
pub(crate) fn draw_batches(
    render_pass: &mut wgpu::RenderPass<'_>,
    batches: &[DrawBatch],
    ctx: &FrameContext<'_>,
    pipeline_cache: &mut MaterialPipelineCache,
    with_depth_prepass: bool,
) {
    if with_depth_prepass {
        // Depth pre-pass for transparent objects: render depth-only with alpha test
        // so opaque portions of blend materials establish correct depth occlusion.
        let mut prepass_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in batches {
            let material = ctx.scene.get_material(batch.material_id).unwrap();
            let material_props = material.get_properties(batch.primitive_type);

            if material_props.alpha_mode != AlphaMode::Blend {
                continue;
            }

            let material_gpu = ctx
                .gpu_resources
                .get_material(batch.material_id, batch.primitive_type)
                .expect("Material GPU resources not initialized");
            render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

            // Normalize key: only has_lighting (pipeline layout), double_sided
            // (cull mode), and primitive_type matter for depth-only output.
            // alpha_mode is always Mask, IBL is unused.
            let pipeline_key = PipelineCacheKey {
                material_props: MaterialProperties {
                    alpha_mode: AlphaMode::Mask,
                    has_lighting: material_props.has_lighting,
                    double_sided: material_props.double_sided,
                    always_on_top: material_props.always_on_top,
                },
                scene_props: SceneProperties { has_ibl: false },
                primitive_type: batch.primitive_type,
                depth_prepass: true,
            };
            if prepass_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = pipeline_cache.get_or_create(ctx.device, pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                prepass_pipeline_key = Some(pipeline_key);
            }

            let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
            let gpu_mesh = ctx
                .gpu_resources
                .get_mesh(batch.mesh_id)
                .expect("Mesh GPU resources not initialized");
            gpu_resources::draw_mesh_instances(
                ctx.device,
                render_pass,
                gpu_mesh,
                batch.primitive_type,
                &batch.instances,
                mesh.index_count(batch.primitive_type),
            );
        }
    }

    // Main draw loop
    let mut current_pipeline_key: Option<PipelineCacheKey> = None;
    for batch in batches {
        let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
        let material = ctx.scene.get_material(batch.material_id).unwrap();
        let material_props = material.get_properties(batch.primitive_type);

        let material_gpu = ctx
            .gpu_resources
            .get_material(batch.material_id, batch.primitive_type)
            .expect("Material GPU resources not initialized");
        render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

        // Only change pipeline if material properties, scene properties, or primitive type changes
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

        let gpu_mesh = ctx
            .gpu_resources
            .get_mesh(batch.mesh_id)
            .expect("Mesh GPU resources not initialized");
        gpu_resources::draw_mesh_instances(
            ctx.device,
            render_pass,
            gpu_mesh,
            batch.primitive_type,
            &batch.instances,
            mesh.index_count(batch.primitive_type),
        );
    }
}

/// Clears color/depth/stencil, binds camera/lights/IBL, runs a depth pre-pass for
/// `Blend`-mode materials, then draws all batches with pipeline caching.
pub(crate) struct MainPass;

impl SceneRenderPass for MainPass {
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
        pipeline_cache: &mut MaterialPipelineCache,
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
