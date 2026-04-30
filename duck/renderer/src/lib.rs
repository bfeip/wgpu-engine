// Re-export scene crate so internal modules can use `crate::scene::*`
pub use duck_engine_scene as scene;

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
