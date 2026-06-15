//! GPU resource management for scene objects.
//!
//! This module centralizes GPU resource tracking for meshes, textures, and materials.
//! It tracks which resources have been uploaded and their sync state using generation numbers.

mod buffer_layouts;
mod renderer_resources;
mod state;
mod uniforms;

// Re-export everything at the module level to preserve existing import paths.

// From render-core
pub(crate) use crate::render_core::GpuTexture;

// From uniforms
pub use uniforms::{CameraUniform, LightsArrayUniform, MaterialUniform};
pub(super) use uniforms::{OutlineUniform, SilhouetteUniform};

// From state
pub(crate) use state::{
    create_mesh_gpu_resources, create_texture_gpu_resources, draw_mesh_instances,
    draw_mesh_subgeom, ColorResources, MaterialGpuResources, MeshGpuResources,
};

// From buffer_layouts
pub use buffer_layouts::{instance_buffer_layout, vertex_buffer_layout};

// From renderer_resources
pub(super) use renderer_resources::{
    BindGroupLayouts, CameraResources, LightResources, MaterialLayoutCache, PipelineCacheKey,
};
