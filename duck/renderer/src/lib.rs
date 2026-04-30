// Re-export scene crate so internal modules can use `crate::scene::*`
pub use duck_engine_scene as scene;

pub(crate) fn rgba_to_wgpu_color(c: scene::common::RgbaColor) -> wgpu::Color {
    wgpu::Color { r: c.r as f64, g: c.g as f64, b: c.b as f64, a: c.a as f64 }
}

pub mod ibl;
mod highlight_query;
mod renderer;
mod shaders;

pub use renderer::{
    CustomPipelineBuilder, DrawBatch, DrawData, FrameContext, HiddenLineConfig, HiddenLineWorkflow,
    MaterialPipelineCache, Renderer, RenderWorkflow, SceneRenderPass, ShadedWorkflow,
    instance_buffer_layout, vertex_buffer_layout,
};
pub use highlight_query::{OutlineConfig, HighlightQuery};
