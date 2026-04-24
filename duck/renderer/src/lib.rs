// Re-export scene crate so internal modules can use `crate::scene::*`
pub use duck_engine_scene as scene;

pub mod ibl;
mod selection_query;
mod renderer;
mod shaders;

pub use renderer::{
    CustomPipelineBuilder, DrawBatch, DrawData, FrameContext, PipelineCache, Renderer,
    SceneRenderPass, instance_buffer_layout, vertex_buffer_layout,
};
pub use selection_query::{OutlineConfig, SelectionQuery};
