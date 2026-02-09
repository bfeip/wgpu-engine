mod camera;
pub mod ibl;
pub mod input;
mod renderer;
mod shaders;
pub mod event;
pub mod operator;
pub mod selection;
mod scene_scale;
mod viewer;

// Re-export scene crate modules for backward compatibility
pub use wgpu_engine_scene::common;
pub use wgpu_engine_scene::scene;
pub use wgpu_engine_scene::geom_query;
pub use wgpu_engine_scene::gltf;
pub use wgpu_engine_scene::camera as scene_camera;

// Winit support - only available when winit is a dependency
#[cfg(feature = "winit-support")]
pub mod winit_support;

// Egui support - only available when egui is a dependency
#[cfg(feature = "egui-support")]
pub mod egui_support;

pub use wgpu_engine_scene::scene::Scene;
pub use viewer::Viewer;
pub use wgpu_engine_scene::Camera;
pub use wgpu_engine_scene::gltf::{load_gltf_scene_from_path, load_gltf_scene_from_slice, GltfLoadResult};
pub use wgpu_engine_scene::scene::{EnvironmentMap, EnvironmentMapId};
pub use selection::{SelectionItem, SelectionManager};
