use crate::scene::{AlphaMode, MaterialProperties, PrimitiveType, SceneProperties};

use super::batching::{DrawBatch, SubGeomBatch};
use super::gpu_resources::{self, PipelineCacheKey};
use super::pass_context::FrameContext;
use super::pipeline::PipelineCache;

/// Bind the scene-level bind groups shared by all geometry passes:
/// - Group 0: Camera (view/proj + eye position)
/// - Group 1: Lights
/// - Group 3: IBL environment (when active)
fn bind_scene_groups<'r>(render_pass: &mut wgpu::RenderPass<'r>, ctx: &'r FrameContext<'r>) {
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
fn draw_batches(
    render_pass: &mut wgpu::RenderPass<'_>,
    batches: &[DrawBatch],
    ctx: &FrameContext<'_>,
    pipeline_cache: &mut PipelineCache,
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

/// Pass 1: Main scene render.
///
/// Clears color/depth/stencil, binds camera/lights/IBL, runs a depth pre-pass for
/// `Blend`-mode materials, then draws all batches with pipeline caching.
pub(super) fn run_main_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    batches: &[DrawBatch],
    ctx: &FrameContext<'_>,
    pipeline_cache: &mut PipelineCache,
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
    draw_batches(&mut render_pass, batches, ctx, pipeline_cache, true);
}

/// Pass 2 (conditional): Overlay render.
///
/// Loads the existing color attachment, clears a separate depth buffer, and draws
/// always-on-top geometry so it depth-tests among itself but not against the scene.
pub(super) fn run_overlay_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    batches: &[DrawBatch],
    ctx: &FrameContext<'_>,
    pipeline_cache: &mut PipelineCache,
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
            view: &ctx.renderer_textures.overlay_depth.view,
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
    draw_batches(&mut render_pass, batches, ctx, pipeline_cache, false);
}

/// Pass 3 (conditional): Selection mask.
///
/// Renders selected geometry into a single-channel mask texture (R8Unorm). The main
/// depth buffer is loaded (not cleared) so occluded selections are masked correctly.
/// Only triangle meshes are outlined; lines and points are skipped.
///
/// Must run before `run_outline_pass`, which reads the mask texture this writes.
pub(super) fn run_selection_mask_pass(
    encoder: &mut wgpu::CommandEncoder,
    selected_batches: &[DrawBatch],
    sub_geom_batches: &[SubGeomBatch],
    ctx: &FrameContext<'_>,
) {
    let mask_view = &ctx.outline_resources.screenspace.texture.view;

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Selection Mask Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: mask_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &ctx.renderer_textures.depth.view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        occlusion_query_set: None,
        timestamp_writes: None,
    });

    render_pass.set_pipeline(&ctx.outline_resources.mask_pipeline);
    render_pass.set_bind_group(0, ctx.camera_bind_group, &[]);

    for batch in selected_batches {
        if batch.primitive_type != PrimitiveType::TriangleList {
            continue; // Only outline triangle meshes
        }
        let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
        let gpu_mesh = ctx
            .gpu_resources
            .get_mesh(batch.mesh_id)
            .expect("Mesh GPU resources not initialized");
        gpu_resources::draw_mesh_instances(
            ctx.device,
            &mut render_pass,
            gpu_mesh,
            batch.primitive_type,
            &batch.instances,
            mesh.index_count(batch.primitive_type),
        );
    }

    for batch in sub_geom_batches {
        if batch.primitive_type != PrimitiveType::TriangleList {
            continue; // Only outline triangle sub-geometry
        }
        let gpu_mesh = ctx
            .gpu_resources
            .get_mesh(batch.mesh_id)
            .expect("Mesh GPU resources not initialized");
        gpu_resources::draw_mesh_subgeom(
            ctx.device,
            &mut render_pass,
            gpu_mesh,
            batch.primitive_type,
            &batch.instance_transform,
            batch.first_index,
            batch.index_count,
        );
    }
}

/// Pass 4 (conditional): Screenspace outline post-process.
///
/// Fullscreen triangle pass that reads the mask written by `run_selection_mask_pass`
/// and composites the outline color over the existing scene color.
///
/// Must run after `run_selection_mask_pass`.
pub(super) fn run_outline_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    ctx: &FrameContext<'_>,
) {
    let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Outline Screenspace Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color_view,
            resolve_target,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });

    render_pass.set_pipeline(&ctx.outline_resources.screenspace.pipeline);
    render_pass.set_bind_group(0, &ctx.outline_resources.screenspace.bind_group, &[]);
    render_pass.draw(0..3, 0..1); // Fullscreen triangle
}
