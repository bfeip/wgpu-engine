use std::path::PathBuf;

/// Unique identifier for an environment map in a scene.
pub type EnvironmentMapId = u32;

/// Number of faces in a cubemap.
pub const CUBEMAP_FACES: usize = 6;

/// Raw pixel data for a single cubemap face at a single mip level.
/// Stored as Rgba16Float (8 bytes/pixel), row-major.
pub type CubemapFaceData = Vec<u8>;

/// Raw pixel data for one mip level of a cubemap: one [`CubemapFaceData`] per face.
pub type CubemapMipData = [CubemapFaceData; CUBEMAP_FACES];

/// Raw pixel data for a preprocessed cubemap texture.
///
/// Each mip level contains 6 faces of Rgba16Float pixel data (8 bytes per pixel).
/// Face order follows the standard cubemap convention: +X, -X, +Y, -Y, +Z, -Z.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PreprocessedCubemap {
    /// Width and height of each face at the base (mip 0) level.
    pub face_size: u32,
    /// Number of mip levels. Each successive mip is half the size of the previous.
    pub mip_levels: u32,
    /// Pixel data indexed as `mip_data[mip][face]`.
    pub mip_data: Vec<CubemapMipData>,
}

/// Preprocessed IBL (Image-Based Lighting) data for an environment map.
///
/// Contains the precomputed textures needed for split-sum PBR environment lighting,
/// stored as CPU-side pixel data with no GPU dependencies. This allows IBL to work
/// on platforms without compute shader support (e.g. WebGL) by uploading the data
/// directly to GPU textures at runtime.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PreprocessedIbl {
    /// Diffuse irradiance cubemap (typically 32x32 per face, 1 mip level).
    pub irradiance: PreprocessedCubemap,
    /// Pre-filtered specular cubemap (typically 128x128 base, 5 mip levels).
    /// Each mip level corresponds to increasing roughness.
    pub prefiltered: PreprocessedCubemap,
    /// Optional BRDF integration LUT (512x512, Rgba16Float).
    /// When `None`, the renderer uses its built-in baked LUT.
    /// Provided for future support of custom BRDF models.
    pub brdf_lut: Option<Vec<u8>>,
}

/// Source data for an environment map.
#[derive(Debug, Clone)]
pub enum EnvironmentSource {
    /// Equirectangular HDR image loaded from a file path.
    EquirectangularPath(PathBuf),
    /// Equirectangular HDR image stored as raw .hdr file bytes.
    EquirectangularHdr(Vec<u8>),
    /// Source was discarded after preprocessing; only baked IBL data remains.
    Preprocessed,
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
    /// Preprocessed IBL data, if available.
    preprocessed_ibl: Option<PreprocessedIbl>,
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
            preprocessed_ibl: None,
        }
    }

    /// Create an environment map from in-memory HDR data.
    ///
    /// The HDR data will be processed into IBL maps when the environment is first used.
    pub fn from_hdr_data(id: EnvironmentMapId, data: Vec<u8>) -> Self {
        Self {
            id,
            source: EnvironmentSource::EquirectangularHdr(data),
            intensity: 1.0,
            rotation: 0.0,
            generation: 1,
            preprocessed_ibl: None,
        }
    }

    /// Create an environment map from preprocessed IBL data.
    ///
    /// Used when the original HDR source has been discarded and only the
    /// baked irradiance/prefiltered cubemaps remain.
    pub fn from_preprocessed(id: EnvironmentMapId, preprocessed: PreprocessedIbl) -> Self {
        Self {
            id,
            source: EnvironmentSource::Preprocessed,
            intensity: 1.0,
            rotation: 0.0,
            generation: 1,
            preprocessed_ibl: Some(preprocessed),
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

    /// Drop the original HDR source data, marking this environment as preprocessed-only.
    ///
    /// After calling this, the environment can only be used if preprocessed IBL data
    /// has been attached via [`set_preprocessed_ibl`]. This is used when baking IBL
    /// data offline to avoid storing the (large) HDR source in the serialized scene.
    pub fn drop_source(&mut self) {
        self.source = EnvironmentSource::Preprocessed;
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

    /// Get the preprocessed IBL data, if available.
    pub fn preprocessed_ibl(&self) -> Option<&PreprocessedIbl> {
        self.preprocessed_ibl.as_ref()
    }

    /// Set preprocessed IBL data for this environment map.
    pub fn set_preprocessed_ibl(&mut self, data: PreprocessedIbl) {
        self.preprocessed_ibl = Some(data);
        self.generation += 1;
    }
}
