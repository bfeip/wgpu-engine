#![allow(dead_code)]

mod texture;
mod camera;
mod light;
mod common;
mod input;
mod scene;
mod geom_query;
mod drawstate;
mod material;
mod shaders;
mod gltf;
mod event;
mod operator;
mod annotation;

// Winit support - only available when winit is a dependency
#[cfg(feature = "winit-support")]
pub mod winit_support;

// Viewer - the main application viewer
pub mod viewer;

// Re-export commonly used types from the library
pub use common::{PhysicalSize, RgbaColor};
pub use input::*;
pub use event::{Event, EventKind, EventDispatcher, EventContext};
pub use drawstate::DrawState;
pub use scene::Scene;
pub use viewer::Viewer;
