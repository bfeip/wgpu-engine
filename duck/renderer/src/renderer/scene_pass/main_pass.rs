use duck_engine_scene::AlphaMode;

use crate::DrawBatch;
use crate::abi;
use crate::render_core::{FrameTargets, Gpu, GpuTexture};
use crate::renderer::pipeline::PipelineCacheKey;
use crate::renderer::surface_config::SurfaceConfig;

use super::super::pass_context::{SceneFrame, SceneRenderPass};

/// Bind the scene-level bind groups shared by all geometry passes:
/// - Group 0: Camera (view/proj + eye position)
/// - Group 1: Lights
/// - Group 3: IBL environment (when active)
pub(crate) fn bind_scene_groups<'p>(render_pass: &mut wgpu::RenderPass<'p>, frame: &SceneFrame<'p>) {
    render_pass.set_bind_group(abi::GROUP_CAMERA, frame.bindings.camera, &[]);
    render_pass.set_bind_group(abi::GROUP_LIGHTS, frame.bindings.lights, &[]);
    if let Some(ibl) = frame.bindings.ibl {
        render_pass.set_bind_group(abi::GROUP_IBL, ibl, &[]);
    }
}

/// Draw a list of batches through the scene geometry pipeline.
///
/// If `with_depth_prepass` is true, a depth-only pre-pass for `Blend`-mode materials
/// is executed first so their opaque portions establish correct depth occlusion before
/// the main draw loop renders them with blending.
pub(crate) fn draw_batches(
    gpu: &Gpu,
    render_pass: &mut wgpu::RenderPass<'_>,
    batches: &[DrawBatch],
    frame: &mut SceneFrame<'_>,
    with_depth_prepass: bool,
) {
    let scene = frame.scene;
    let gpu_meshes = frame.gpu_meshes;
    let scene_props = frame.scene_props.clone();
    let materials = &mut *frame.materials;

    if with_depth_prepass {
        // Depth pre-pass for transparent objects: render depth-only with alpha test
        // so opaque portions of blend materials establish correct depth occlusion.
        let mut prepass_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in batches {
            let material_props = &batch.material_props;

            if material_props.alpha_mode != AlphaMode::Blend {
                continue;
            }

            let Some(mesh) = scene.get_mesh(batch.mesh_id) else {
                continue;
            };
            let Some(gpu_mesh) = gpu_meshes.get(batch.mesh_id) else {
                continue;
            };

            // Pipeline (mutable phase) before the material bind group (shared
            // phase): both borrow `materials`, and wgpu does not retain either
            // borrow past the set call, so they must not overlap.
            //
            // depth_prepass=true compiles in the alpha-test discard and masks
            // color writes; IBL is irrelevant for depth-only output (scene IBL
            // passed as false). Texture presence still matches the material so
            // its bind group stays compatible with this pipeline's layout.
            let pipeline_key = PipelineCacheKey {
                surface: SurfaceConfig::new(material_props.clone(), false, true),
                primitive_type: batch.primitive_type,
            };
            if prepass_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = materials.pipeline(&gpu.device, pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                prepass_pipeline_key = Some(pipeline_key);
            }

            let Some(material_gpu) = materials.bind_group(batch.material) else {
                continue;
            };
            render_pass.set_bind_group(abi::GROUP_MATERIAL, &material_gpu.bind_group, &[]);

            gpu_mesh.draw_instances(
                &gpu.device,
                render_pass,
                batch.primitive_type,
                &batch.instances,
                mesh.index_count(batch.primitive_type),
            );
        }
    }

    // Main draw loop
    let mut current_pipeline_key: Option<PipelineCacheKey> = None;
    for batch in batches {
        let Some(mesh) = scene.get_mesh(batch.mesh_id) else {
            continue;
        };
        let Some(gpu_mesh) = gpu_meshes.get(batch.mesh_id) else {
            continue;
        };

        // Pipeline (mutable phase) before material bind group (shared phase); see
        // the prepass note above on the borrow ordering.
        let pipeline_key = PipelineCacheKey {
            surface: SurfaceConfig::new(batch.material_props.clone(), scene_props.has_ibl, false),
            primitive_type: batch.primitive_type,
        };
        if current_pipeline_key.as_ref() != Some(&pipeline_key) {
            let pipeline = materials.pipeline(&gpu.device, pipeline_key.clone());
            render_pass.set_pipeline(pipeline);
            current_pipeline_key = Some(pipeline_key);
        }

        let Some(material_gpu) = materials.bind_group(batch.material) else {
            continue;
        };
        render_pass.set_bind_group(abi::GROUP_MATERIAL, &material_gpu.bind_group, &[]);

        gpu_mesh.draw_instances(
            &gpu.device,
            render_pass,
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
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let (color_view, resolve_target) = targets.color_views(view);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D Scene Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(frame.background_color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None, // No stencil buffer
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        bind_scene_groups(&mut render_pass, frame);
        let batches = frame.draw.all_batches();
        draw_batches(gpu, &mut render_pass, batches, frame, true);
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
            depth: GpuTexture::depth(device, width, height, sample_count, "overlay_depth_texture"),
        }
    }
}

impl SceneRenderPass for OverlayPass {
    fn is_active(&self, frame: &SceneFrame<'_>) -> bool {
        frame.draw.has_overlay()
    }

    fn resize(&mut self, gpu: &Gpu, targets: &FrameTargets) {
        let (w, h) = targets.size();
        self.depth = GpuTexture::depth(&gpu.device, w, h, targets.sample_count(), "overlay_depth_texture");
    }

    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut SceneFrame<'_>,
    ) {
        let (color_view, resolve_target) = targets.color_views(view);

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

        bind_scene_groups(&mut render_pass, frame);
        let batches = frame.draw.overlay_batches();
        draw_batches(gpu, &mut render_pass, batches, frame, false);
    }
}
