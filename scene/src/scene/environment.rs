use std::path::PathBuf;

/// Unique identifier for an environment map in a scene.
pub type EnvironmentMapId = u32;

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
    pub source: EnvironmentSource,
    /// Intensity multiplier for the environment lighting.
    pub intensity: f32,
    /// Rotation around the Y axis in radians.
    pub rotation: f32,
    /// Whether the environment needs to be (re)generated.
    pub dirty: bool,
}

impl EnvironmentMap {
    /// Create an environment map from an equirectangular HDR file path.
    ///
    /// The HDR file will be loaded and processed when the environment is first used.
    /// This is internal - use `Scene::add_environment_map_from_hdr_path` to create environment maps.
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

    /// Get the intensity multiplier.
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Get the rotation in radians.
    pub fn rotation(&self) -> f32 {
        self.rotation
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
