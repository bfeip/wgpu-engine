use crate::scene::{AlphaMode, MaterialProperties, SceneProperties};

use super::super::batching::DrawBatch;
use super::super::gpu_resources::{self, PipelineCacheKey};
use super::super::pass_context::FrameContext;
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
