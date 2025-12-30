mod annotation;
mod camera;
pub mod common;
pub mod input;
pub mod scene;
pub mod geom_query;
mod drawstate;
mod shaders;
mod gltf;
pub mod event;
pub mod operator;
mod scene_scale;
mod viewer;

// Winit support - only available when winit is a dependency
#[cfg(feature = "winit-support")]
pub mod winit_support;

// Egui support - only available when egui is a dependency
#[cfg(feature = "egui-support")]
pub mod egui_support;

pub use annotation::{AnnotationId, AnnotationManager};
pub use scene::Scene;
pub use viewer::Viewer;
pub use camera::Camera;
pub use gltf::{load_gltf_scene_from_path, load_gltf_scene_from_slice, GltfLoadResult};
