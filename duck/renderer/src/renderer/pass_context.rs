use crate::scene::{Scene, SceneProperties};

use super::batching::DrawData;
use super::gpu_resources::{GpuResourceManager, RendererTextures};
use super::outline::OutlineResources;
use super::pipeline::PipelineCache;

/// Per-frame read-only snapshot of all renderer state a pass needs.
///
/// Constructed once at the top of `render_scene_to_view` from named field borrows
/// of `Renderer`. Because it does not borrow `pipeline_cache`, the caller can hold
/// a `&FrameContext` while separately passing `&mut pipeline_cache` to pass functions
/// — a disjoint named-field borrow the borrow checker accepts without conflict.
pub(super) struct FrameContext<'a> {
    pub device: &'a wgpu::Device,
    pub scene: &'a Scene,
    pub gpu_resources: &'a GpuResourceManager,
    pub camera_bind_group: &'a wgpu::BindGroup,
    pub lights_bind_group: &'a wgpu::BindGroup,
    /// `Some` if there is an active, fully-processed environment map for IBL.
    pub ibl_bind_group: Option<&'a wgpu::BindGroup>,
    /// Derived from `ibl_bind_group` — `has_ibl` is true iff `ibl_bind_group` is `Some`.
    pub scene_props: SceneProperties,
    pub renderer_textures: &'a RendererTextures,
    pub outline_resources: &'a OutlineResources,
    // The following fields are not used by built-in passes but are exposed here so
    // that user-defined passes implementing `SceneRenderPass` have access to them.
    #[allow(dead_code)]
    pub sample_count: u32,
    #[allow(dead_code)]
    pub surface_format: wgpu::TextureFormat,
    #[allow(dead_code)]
    pub size: (u32, u32),
}

/// Extension point for user-defined render passes.
///
/// Implement this trait and pass instances to `render_scene_to_view_with_passes`
/// (not yet wired up) to inject custom draw passes into the frame.
///
/// Each pass receives `FrameContext` for read-only renderer state and a mutable
/// `PipelineCache` for pipeline creation/lookup. The `DrawData` provides access
/// to the sorted, partitioned batch lists for the current frame.
#[allow(dead_code, unused_variables)]
pub trait SceneRenderPass {
    fn is_active(&self, draw_data: &DrawData) -> bool {
        true
    }
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    );
}
