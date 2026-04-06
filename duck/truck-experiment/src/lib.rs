pub use wgpu_engine_common as common;
pub use wgpu_engine_scene as scene;

mod body;
mod edge;
mod face;
mod scene_map;
mod tessellation;

pub use body::{Body, BodyId};
pub use edge::{Edge, EdgeId};
pub use face::{Face, FaceId};
pub use scene_map::{add_body_to_scene, CadSceneMap};
pub use tessellation::{tessellate_body, TessellatedBody, TessellationOptions};

#[cfg(test)]
mod tests;
