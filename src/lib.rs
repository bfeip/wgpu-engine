#![allow(dead_code)]

mod texture;
mod camera;
mod light;
pub mod common;
pub mod input;
pub mod scene;
pub mod geom_query;
mod drawstate;
mod material;
mod shaders;
mod gltf;
pub mod event;
pub mod operator;
mod annotation;
mod viewer;

// Winit support - only available when winit is a dependency
#[cfg(feature = "winit-support")]
pub mod winit_support;

pub use scene::Scene;
pub use viewer::Viewer;
pub use camera::Camera;
pub use light::Light;
pub use gltf::load_gltf_scene;
pub use annotation::{AnnotationId, AnnotationManager};
