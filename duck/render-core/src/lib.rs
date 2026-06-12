//! Scene-agnostic rendering core for duck-engine.
//!
//! This crate contains the parts of the renderer that do not depend on any
//! particular object model: GPU plumbing, render-target management, headless
//! readback, and reusable caching utilities. It deliberately has no dependency
//! on `duck-engine-scene` or `duck-engine-common` — nothing here knows what a
//! scene, mesh, material, or camera is.
//!
//! The standard scene renderer (`duck-engine-renderer`) builds on this crate
//! and re-exports it as `render_core`.

mod gen_cache;
mod pipeline_cache;
mod readback;
mod shader;
mod texture;

pub use gen_cache::GenCache;
pub use pipeline_cache::PipelineCache;
pub use readback::ReadbackTarget;
pub use shader::ShaderLibrary;
pub use texture::GpuTexture;
