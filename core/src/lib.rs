// Re-export scene crate as `scene` module
pub use wgpu_engine_scene as scene;

// Convenience re-exports from scene crate
pub use wgpu_engine_scene::common;
pub use wgpu_engine_scene::geom_query;
pub use wgpu_engine_scene::gltf;
pub use wgpu_engine_scene::camera as scene_camera;
pub use wgpu_engine_scene::{Camera, EnvironmentMap, EnvironmentMapId, Scene};
pub use wgpu_engine_scene::gltf::{load_gltf_scene_from_path, load_gltf_scene_from_slice, GltfLoadResult};

// Core modules
mod camera;
pub mod event;
pub mod ibl;
pub mod input;
pub mod operator;
pub mod selection;
mod renderer;
mod scene_scale;
mod shaders;
mod viewer;

pub use selection::{SelectionItem, SelectionManager};
pub use viewer::Viewer;

// Optional integrations
#[cfg(feature = "winit-support")]
pub mod winit_support;

#[cfg(feature = "egui-support")]
pub mod egui_support;
