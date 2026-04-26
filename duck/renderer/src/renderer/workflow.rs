use super::batching::DrawData;
use super::pass_context::{FrameContext, SceneRenderPass};
use super::pipeline::PipelineCache;
use super::scene_pass::{HiddenLineEdgesPass, HiddenLineSolidPass, MainPass, OverlayPass, SelectionOutlinePass};
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
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    );
}

// ---------------------------------------------------------------------------
// ShadedWorkflow
// ---------------------------------------------------------------------------

/// The default shaded rendering workflow.
///
/// Runs the standard pass sequence: main geometry, overlay (always-on-top)
/// geometry, and selection outlines. Holds passes as `Box<dyn SceneRenderPass>`
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
                Box::new(SelectionOutlinePass::new(device, config, camera_bgl, shader_generator, sample_count)),
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
        pipeline_cache: &mut PipelineCache,
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
// HiddenLineWorkflow
// ---------------------------------------------------------------------------

/// Hidden-line rendering workflow.
///
/// Renders scene geometry as white solid faces with depth-occluded edges on
/// top. Only meshes that carry explicit `LineList` primitives show edges;
/// triangle-only meshes render as flat-white solids with no edges.
///
/// Pass sequence:
/// 1. [`HiddenLineSolidPass`] — clear to white, render all triangles white, write depth.
/// 2. [`HiddenLineEdgesPass`] — render `LineList` geometry depth-tested against the solids.
pub struct HiddenLineWorkflow {
    solid_pass: HiddenLineSolidPass,
    edge_pass: HiddenLineEdgesPass,
}

impl HiddenLineWorkflow {
    pub(super) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
    ) -> Self {
        Self {
            solid_pass: HiddenLineSolidPass::new(device, surface_format, sample_count, camera_bgl, shader_generator),
            edge_pass: HiddenLineEdgesPass,
        }
    }
}

impl RenderWorkflow for HiddenLineWorkflow {
    fn name(&self) -> &'static str { "Hidden Line" }

    fn resize(&mut self, _device: &wgpu::Device, _size: (u32, u32), _sample_count: u32) {}

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        self.solid_pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
        self.edge_pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
    }
}
