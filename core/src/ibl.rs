//! Image Based Lighting (IBL) support for PBR rendering.
//!
//! This module provides environment map loading, processing, and GPU resource management
//! for image-based lighting. Environment maps are loaded from equirectangular HDR images
//! and processed into the required formats (irradiance map, pre-filtered environment map,
//! and BRDF LUT) using GPU compute shaders.

mod cubemap;
mod equirect;
mod hdr_loader;

pub use hdr_loader::{load_hdr_from_path, HdrImage};

pub(crate) use cubemap::{CubemapFace, GpuCubemap};
pub(crate) use equirect::EquirectToCubePipeline;

use std::path::PathBuf;

/// Unique identifier for an environment map in a scene.
pub type EnvironmentMapId = u32;

/// Size of the environment cubemap (per face).
pub const ENVIRONMENT_CUBEMAP_SIZE: u32 = 512;

/// Size of the irradiance cubemap (per face). Low resolution since irradiance is low-frequency.
pub const IRRADIANCE_CUBEMAP_SIZE: u32 = 32;

/// Size of the pre-filtered environment cubemap base level (per face).
pub const PREFILTERED_CUBEMAP_SIZE: u32 = 128;

/// Number of mip levels for the pre-filtered cubemap (roughness levels).
pub const PREFILTERED_MIP_LEVELS: u32 = 5;

/// Size of the BRDF integration LUT.
pub const BRDF_LUT_SIZE: u32 = 512;

/// Source data for an environment map.
#[derive(Debug, Clone)]
pub enum EnvironmentSource {
    /// Equirectangular HDR image loaded from a file path.
    EquirectangularPath(PathBuf),
}

/// An environment map used for image-based lighting.
///
/// Environment maps provide ambient lighting through diffuse irradiance and
/// specular reflections through a pre-filtered environment map.
#[derive(Debug)]
pub struct EnvironmentMap {
    /// Unique identifier for this environment map.
    pub id: EnvironmentMapId,
    /// Source data for the environment map.
    pub(crate) source: EnvironmentSource,
    /// Intensity multiplier for the environment lighting.
    pub(crate) intensity: f32,
    /// Rotation around the Y axis in radians.
    pub(crate) rotation: f32,
    /// Whether the environment needs to be (re)generated.
    pub(crate) dirty: bool,
}

impl EnvironmentMap {
    /// Create an environment map from an equirectangular HDR file path.
    ///
    /// The HDR file will be loaded and processed when the environment is first used.
    pub fn from_hdr_path(id: EnvironmentMapId, path: impl Into<PathBuf>) -> Self {
        Self {
            id,
            source: EnvironmentSource::EquirectangularPath(path.into()),
            intensity: 1.0,
            rotation: 0.0,
            dirty: true,
        }
    }

    /// Set the intensity multiplier for this environment.
    ///
    /// Default is 1.0. Higher values make the environment brighter.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    /// Set the rotation of the environment around the Y axis.
    ///
    /// Rotation is in radians. Default is 0.0.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    /// Check if this environment needs GPU resource generation.
    pub fn needs_generation(&self) -> bool {
        self.dirty
    }

    /// Mark the environment as needing regeneration.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}
