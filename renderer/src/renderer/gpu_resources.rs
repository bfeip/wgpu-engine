//! GPU resource management for scene objects.
//!
//! This module centralizes GPU resource tracking for meshes, textures, and materials.
//! It tracks which resources have been uploaded and their sync state using generation numbers.

mod buffer_layouts;
mod renderer_resources;
mod state;
mod texture_helpers;
mod uniforms;

// Re-export everything at the module level to preserve existing import paths.

// From uniforms
pub use uniforms::{CameraUniform, LightsArrayUniform, PbrUniform};

// From state
pub(crate) use state::{
    draw_mesh_instances, GpuResourceManager, GpuTexture, MaterialGpuResources,
};

// From buffer_layouts
pub(crate) use buffer_layouts::{instance_buffer_layout, vertex_buffer_layout};

// From renderer_resources
pub(super) use renderer_resources::{
    CameraResources, DefaultTextures, LightResources,
    MaterialBindGroupLayouts, MaterialPipelineLayouts, PipelineCacheKey,
};
