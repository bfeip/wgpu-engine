use crate::scene::{Scene, SceneProperties};

use super::batching::{DrawBatch, DrawData};
use super::gpu_resources::{self, GpuResourceManager, RendererTextures};
use super::pipeline::PipelineCache;

/// Per-frame read-only snapshot of all renderer state a pass needs.
///
/// Constructed once at the top of `render_scene_to_view` from named field borrows
/// of `Renderer`. Because it does not borrow `pipeline_cache`, the caller can hold
/// a `&FrameContext` while separately passing `&mut pipeline_cache` to pass functions
/// — a disjoint named-field borrow the borrow checker accepts without conflict.
pub struct FrameContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub scene: &'a Scene,
    pub(crate) gpu_resources: &'a GpuResourceManager,
    pub camera_bind_group: &'a wgpu::BindGroup,
    pub lights_bind_group: &'a wgpu::BindGroup,
    /// `Some` if there is an active, fully-processed environment map for IBL.
    pub ibl_bind_group: Option<&'a wgpu::BindGroup>,
    /// Derived from `ibl_bind_group` — `has_ibl` is true iff `ibl_bind_group` is `Some`.
    pub scene_props: SceneProperties,
    pub(in crate::renderer) renderer_textures: &'a RendererTextures,
    pub sample_count: u32,
    pub surface_format: wgpu::TextureFormat,
    pub size: (u32, u32),
}

impl<'a> FrameContext<'a> {
    /// Draw a [`DrawBatch`] into `render_pass`.
    ///
    /// Looks up the mesh's GPU vertex/index buffers and issues the instanced draw
    /// call. Silently skips batches whose GPU resources haven't been uploaded yet.
    pub fn draw_batch(&self, render_pass: &mut wgpu::RenderPass<'_>, batch: &DrawBatch) {
        let Some(mesh) = self.scene.get_mesh(batch.mesh_id) else { return };
        let Some(gpu_mesh) = self.gpu_resources.get_mesh(batch.mesh_id) else { return };
        gpu_resources::draw_mesh_instances(
            self.device,
            render_pass,
            gpu_mesh,
            batch.primitive_type,
            &batch.instances,
            mesh.index_count(batch.primitive_type),
        );
    }

    /// The shared depth buffer view for this frame.
    ///
    /// Pass this as the `depth_stencil_attachment` view in your render pass
    /// descriptor to depth-test against scene geometry drawn by earlier passes.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.renderer_textures.depth.view
    }
}

/// Extension point for user-defined render passes.
///
/// Each pass receives [`FrameContext`] for read-only renderer state and a mutable
/// [`PipelineCache`] for pipeline creation/lookup. The [`DrawData`] provides access
/// to the sorted, partitioned batch lists for the current frame.
///
/// Passes may be stateless (zero-size structs) or stateful (holding owned GPU resources
/// such as textures or pipelines). Stateful passes implement [`resize`](Self::resize)
/// to recreate size-dependent resources when the viewport changes.
pub trait SceneRenderPass {
    fn is_active(&self, _draw_data: &DrawData) -> bool {
        true
    }

    /// Called after a viewport resize. Passes that own size-dependent GPU resources
    /// (textures, bind groups) should recreate them here. The default is a no-op.
    fn resize(&mut self, _device: &wgpu::Device, _size: (u32, u32), _sample_count: u32) {}

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    );
}
