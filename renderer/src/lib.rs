// Re-export scene crate so internal modules can use `crate::scene::*`
pub use wgpu_engine_scene as scene;

pub mod ibl;
pub mod selection_query;
mod renderer;
mod shaders;

pub use renderer::Renderer;
pub use selection_query::{OutlineConfig, SelectionQuery};
