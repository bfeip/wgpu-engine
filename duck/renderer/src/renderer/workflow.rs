use crate::scene::PrimitiveType;
use crate::scene::common::RgbaColor;

use super::batching::DrawData;
use super::pass_context::{FrameContext, SceneRenderPass};
use super::pipeline::MaterialPipelineCache;
use super::scene_pass::{
    FlatColorPass, FlatColorPassDesc, rgba_to_wgpu_color,
    MainPass, OverlayPass, OutlinePass, SilhouetteEdgesPass,
};
use crate::shaders::ShaderGenerator;

/// A named rendering strategy that owns all passes and GPU resources for one frame style.
///
/// Workflows are the top-level unit of render customization. The renderer runs exactly
/// one workflow per frame, and workflows can be swapped at runtime via
/// [`Renderer::set_workflow`](super::super::Renderer::set_workflow).
///
/// Built-in workflows: [`ShadedWorkflow`], [`HiddenLineWorkflow`].
/// Custom workflows implement this trait directly.
pub trait RenderWorkflow: 'static {
    fn name(&self) -> &'static str;

    /// Called after a viewport resize. Implementations must forward to any passes
    /// that own size-dependent GPU resources.
    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32);

    /// Execute all passes for this frame.
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut MaterialPipelineCache,
        draw_data: &DrawData,
    );
}

// ---------------------------------------------------------------------------
// ShadedWorkflow
// ---------------------------------------------------------------------------

/// The default shaded rendering workflow.
///
/// Runs the standard pass sequence: main geometry, overlay (always-on-top)
/// geometry, and highlight outlines. Holds passes as `Box<dyn SceneRenderPass>`
/// so custom passes can be injected via [`ShadedWorkflow::set_passes`].
pub struct ShadedWorkflow {
    passes: Vec<Box<dyn SceneRenderPass>>,
}

impl ShadedWorkflow {
    pub(super) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
        sample_count: u32,
    ) -> Self {
        Self {
            passes: vec![
                Box::new(MainPass),
                Box::new(OverlayPass::new(device, config.width, config.height, sample_count)),
                Box::new(OutlinePass::new(device, config, camera_bgl, shader_generator, sample_count)),
            ],
        }
    }

    /// Replace the pass list. Passes execute in order; each receives the same
    /// [`FrameContext`] and can skip itself by returning `false` from
    /// [`SceneRenderPass::is_active`].
    pub fn set_passes(&mut self, passes: Vec<Box<dyn SceneRenderPass>>) {
        self.passes = passes;
    }
}

impl RenderWorkflow for ShadedWorkflow {
    fn name(&self) -> &'static str { "Shaded" }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        for pass in &mut self.passes {
            pass.resize(device, size, sample_count);
        }
    }

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut MaterialPipelineCache,
        draw_data: &DrawData,
    ) {
        for pass in &mut self.passes {
            if pass.is_active(draw_data) {
                pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HiddenLineConfig
// ---------------------------------------------------------------------------

/// Color configuration for [`HiddenLineWorkflow`].
#[derive(Clone, Debug)]
pub struct HiddenLineConfig {
    /// Background / face color (also used as the clear color).
    pub face_color: RgbaColor,
    /// Color of lines that are visible (in front of solid geometry).
    pub visible_line_color: RgbaColor,
    /// Color of lines that are occluded (behind solid geometry).
    pub hidden_line_color: RgbaColor,
}

impl Default for HiddenLineConfig {
    fn default() -> Self {
        Self {
            face_color: RgbaColor::WHITE,
            visible_line_color: RgbaColor::BLACK,
            hidden_line_color: RgbaColor { r: 0.6, g: 0.6, b: 0.6, a: 1.0 },
        }
    }
}

// ---------------------------------------------------------------------------
// HiddenLineWorkflow
// ---------------------------------------------------------------------------

/// Hidden-line rendering workflow.
///
/// Renders scene geometry as solid faces with silhouette edges detected from
/// the depth buffer, plus explicit `LineList` primitives in two flat colors:
/// one for occluded lines and one for visible lines. All colors are configured
/// via [`HiddenLineConfig`].
///
/// Pass sequence:
/// 1. `solid_pass`     ([`FlatColorPass`]) — clear to face color, render all triangles, write depth.
/// 2. `silhouette_pass`([`SilhouetteEdgesPass`]) — fullscreen depth-discontinuity edge detection.
/// 3. `occluded_pass`  ([`FlatColorPass`]) — hidden line color where depth compare is `Greater`.
/// 4. `visible_pass`   ([`FlatColorPass`]) — visible line color where depth compare is `LessEqual`.
pub struct HiddenLineWorkflow {
    solid_pass: FlatColorPass,
    silhouette_pass: SilhouetteEdgesPass,
    occluded_pass: FlatColorPass,
    visible_pass: FlatColorPass,
}

impl HiddenLineWorkflow {
    pub(super) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
        config: HiddenLineConfig,
    ) -> Self {
        let solid_pass = FlatColorPass::new(
            device, surface_format, sample_count,
            camera_bgl, lights_bgl, material_color_bgl, shader_generator,
            FlatColorPassDesc {
                label: "Hidden Line Solid",
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                depth_compare: wgpu::CompareFunction::Less,
                depth_write: true,
                // Push faces slightly away so coplanar edges pass depth test.
                depth_bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
                clear_color: Some(rgba_to_wgpu_color(config.face_color)),
                primitive_filter: PrimitiveType::TriangleList,
                color: config.face_color,
            },
        );
        let silhouette_pass = SilhouetteEdgesPass::new(device, surface_format, sample_count, shader_generator);

        let mut make_line_pass = |label, depth_compare, color| FlatColorPass::new(
            device, surface_format, sample_count,
            camera_bgl, lights_bgl, material_color_bgl, shader_generator,
            FlatColorPassDesc {
                label,
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_compare,
                depth_write: false,
                depth_bias: Default::default(),
                clear_color: None,
                primitive_filter: PrimitiveType::LineList,
                color,
            },
        );
        let occluded_pass = make_line_pass(
            "Hidden Line Occluded",
            wgpu::CompareFunction::Greater,
            config.hidden_line_color,
        );
        let visible_pass = make_line_pass(
            "Hidden Line Visible",
            wgpu::CompareFunction::LessEqual,
            config.visible_line_color,
        );

        Self { solid_pass, silhouette_pass, occluded_pass, visible_pass }
    }
}

impl RenderWorkflow for HiddenLineWorkflow {
    fn name(&self) -> &'static str { "Hidden Line" }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        self.solid_pass.resize(device, size, sample_count);
        self.silhouette_pass.resize(device, size, sample_count);
        self.occluded_pass.resize(device, size, sample_count);
        self.visible_pass.resize(device, size, sample_count);
    }

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut MaterialPipelineCache,
        draw_data: &DrawData,
    ) {
        self.solid_pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
        self.silhouette_pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
        let has_lines = draw_data.all_batches().iter().any(|b| b.primitive_type == PrimitiveType::LineList);
        if has_lines {
            self.occluded_pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
            self.visible_pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
        }
    }
}
