use std::path::PathBuf;

/// Unique identifier for an environment map in a scene.
pub type EnvironmentMapId = u32;

/// Source data for an environment map.
#[derive(Debug, Clone)]
pub enum EnvironmentSource {
    /// Equirectangular HDR image loaded from a file path.
    EquirectangularPath(PathBuf),
    /// Equirectangular HDR image stored as raw .hdr file bytes.
    EquirectangularHdr(Vec<u8>),
}

/// An environment map used for image-based lighting.
///
/// Environment maps provide ambient lighting through diffuse irradiance and
/// specular reflections through a pre-filtered environment map.
#[derive(Debug, Clone)]
pub struct EnvironmentMap {
    /// Unique identifier for this environment map.
    pub id: EnvironmentMapId,
    /// Source data for the environment map.
    source: EnvironmentSource,
    /// Intensity multiplier for the environment lighting.
    intensity: f32,
    /// Rotation around the Y axis in radians.
    rotation: f32,
    /// Generation counter, incremented on each change.
    generation: u64,
}

impl EnvironmentMap {
    /// Create an environment map from an equirectangular HDR file path.
    ///
    /// The HDR file will be loaded and processed when the environment is first used.
    /// This is internal - use `Scene::add_environment_map_from_hdr_path` to create environment maps.
    pub(crate) fn from_hdr_path(id: EnvironmentMapId, path: impl Into<PathBuf>) -> Self {
        Self {
            id,
            source: EnvironmentSource::EquirectangularPath(path.into()),
            intensity: 1.0,
            rotation: 0.0,
            generation: 1,
        }
    }

    /// Create an environment map from in-memory HDR data.
    ///
    /// The HDR data will be processed into IBL maps when the environment is first used.
    /// This is internal - use `Scene::add_environment_map_from_hdr_path` to create environment maps.
    pub(crate) fn from_hdr_data(id: EnvironmentMapId, data: Vec<u8>) -> Self {
        Self {
            id,
            source: EnvironmentSource::EquirectangularHdr(data),
            intensity: 1.0,
            rotation: 0.0,
            generation: 1,
        }
    }

    /// Set the intensity multiplier for this environment.
    ///
    /// Default is 1.0. Higher values make the environment brighter.
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self.generation += 1;
        self
    }

    /// Set the rotation of the environment around the Y axis.
    ///
    /// Rotation is in radians. Default is 0.0.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self.generation += 1;
        self
    }

    /// Set the intensity multiplier.
    pub fn set_intensity(&mut self, intensity: f32) {
        self.intensity = intensity;
        self.generation += 1;
    }

    /// Set the rotation of the environment around the Y axis in radians.
    pub fn set_rotation(&mut self, radians: f32) {
        self.rotation = radians;
        self.generation += 1;
    }

    /// Get the source from which the environment was created.
    pub fn source(&self) -> &EnvironmentSource {
        &self.source
    }

    /// Get the intensity multiplier.
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Get the rotation in radians.
    pub fn rotation(&self) -> f32 {
        self.rotation
    }

    /// Returns the current generation counter.
    ///
    /// Starts at 1 and increments on each change. GPU sync code compares
    /// this against a last-synced generation to detect changes.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}
