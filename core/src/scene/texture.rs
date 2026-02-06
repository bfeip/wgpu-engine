use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};

/// Unique identifier for a texture in the scene.
pub type TextureId = u32;

/// Describes how texture source data is stored.
///
/// Textures can be created from embedded image data or loaded lazily from a file path.
/// The `CachedPath` variant allows releasing image data from memory after GPU upload
/// while retaining the ability to reload it if needed.
pub enum TextureSource {
    /// Image data embedded in memory (always available).
    /// Optionally includes original compressed bytes (PNG/JPEG) for efficient serialization.
    Embedded {
        image: DynamicImage,
        /// Original compressed bytes preserved from loading (e.g., from glTF).
        original_bytes: Option<(Vec<u8>, super::format::TextureFormat)>,
    },
    /// Path to load image from (image loaded on demand)
    Path(PathBuf),
    /// Image was loaded from path, with optional cached data.
    /// The image can be released to save memory and reloaded later if needed.
    CachedPath {
        path: PathBuf,
        image: Option<DynamicImage>,
    },
}

/// A texture that can exist without GPU resources.
///
/// Textures store their source data (embedded image or file path) and lazily
/// create GPU resources when first needed for rendering. This allows textures
/// to be created and manipulated without access to the GPU device.
///
/// # Examples
///
/// ```
/// use wgpu_engine::scene::{Texture, Scene};
///
/// // Create from path (loaded lazily when GPU resources are created)
/// let texture = Texture::from_path("texture.png");
///
/// // Add to scene
/// let mut scene = Scene::new();
/// let texture_id = scene.add_texture(texture);
///
/// // GPU resources are created automatically during rendering
/// ```
pub struct Texture {
    /// Unique identifier (assigned by Scene)
    pub(crate) id: TextureId,
    /// Source data for the texture
    source: TextureSource,
    /// Cached dimensions (set after first image load)
    dimensions: Option<(u32, u32)>,
    /// Generation counter - increments on any mutation (for GPU sync tracking)
    generation: u64,
}

impl Texture {
    /// Create a texture from an embedded image.
    ///
    /// The image data is stored in memory. GPU resources are created lazily
    /// when the texture is first used for rendering.
    ///
    /// # Arguments
    /// * `image` - The image data to use for this texture
    pub fn from_image(image: DynamicImage) -> Self {
        let dimensions = Some(image.dimensions());
        Self {
            id: 0, // Assigned by Scene
            source: TextureSource::Embedded { image, original_bytes: None },
            dimensions,
            generation: 1,
        }
    }

    /// Create a texture from a file path.
    ///
    /// The image is not loaded immediately - it will be loaded on demand when
    /// `get_image()` is called or when GPU resources are created.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            id: 0, // Assigned by Scene
            source: TextureSource::Path(path.into()),
            dimensions: None,
            generation: 1,
        }
    }

    /// Create a texture from an embedded image with original compressed bytes preserved.
    ///
    /// This is used when loading from formats like glTF that embed compressed images.
    /// The original bytes are preserved for efficient serialization (avoids re-encoding).
    ///
    /// # Arguments
    /// * `image` - The decoded image data
    /// * `original_bytes` - The original compressed bytes (PNG or JPEG)
    /// * `format` - The format of the original bytes
    pub fn from_image_with_original_bytes(
        image: DynamicImage,
        original_bytes: Vec<u8>,
        format: super::format::TextureFormat,
    ) -> Self {
        let dimensions = Some(image.dimensions());
        Self {
            id: 0,
            source: TextureSource::Embedded {
                image,
                original_bytes: Some((original_bytes, format)),
            },
            dimensions,
            generation: 1,
        }
    }

    /// Get the texture's unique identifier.
    pub fn id(&self) -> TextureId {
        self.id
    }

    /// Get the texture dimensions, if known.
    ///
    /// Returns `None` if the texture was created from a path and hasn't been loaded yet.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.dimensions
    }

    /// Load and return a reference to the image data.
    ///
    /// For path-based textures, this loads the image from disk on first access.
    /// The loaded image is cached for future access.
    ///
    /// # Errors
    /// Returns an error if the image cannot be loaded from the path.
    pub fn get_image(&mut self) -> Result<&DynamicImage> {
        // First, ensure the image is loaded (may mutate self.source)
        self.ensure_image_loaded()?;

        // Now we can return a reference
        match &self.source {
            TextureSource::Embedded { image, .. } => Ok(image),
            TextureSource::CachedPath { image: Some(img), .. } => Ok(img),
            _ => unreachable!("ensure_image_loaded should have loaded the image"),
        }
    }

    /// Ensure the image is loaded into memory.
    fn ensure_image_loaded(&mut self) -> Result<()> {
        match &self.source {
            TextureSource::Embedded { .. } => Ok(()),
            TextureSource::CachedPath { image: Some(_), .. } => Ok(()),
            TextureSource::Path(_) | TextureSource::CachedPath { image: None, .. } => {
                // Need to load the image
                let path = match &self.source {
                    TextureSource::Path(p) => p.clone(),
                    TextureSource::CachedPath { path, .. } => path.clone(),
                    _ => unreachable!(),
                };

                let img = image::open(&path)
                    .with_context(|| format!("Failed to load texture from {:?}", path))?;
                self.dimensions = Some(img.dimensions());

                self.source = TextureSource::CachedPath {
                    path,
                    image: Some(img),
                };
                Ok(())
            }
        }
    }

    /// Returns the current generation counter.
    ///
    /// This value increments on any mutation to the texture data.
    /// Used by renderers to track when GPU resources need updating.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Release cached image data to free memory.
    ///
    /// For path-based textures, this releases the cached image data while
    /// retaining the path for potential reloading. The image can be reloaded
    /// by calling `get_image()` again.
    ///
    /// For embedded textures, this has no effect (the image cannot be reloaded).
    ///
    /// Note: This does not release GPU resources - only the CPU-side image data.
    pub fn release_image_cache(&mut self) {
        if let TextureSource::CachedPath { image, .. } = &mut self.source {
            *image = None;
        }
        // Embedded textures cannot release their image data
    }

    /// Replace the texture's image data.
    ///
    /// This increments the generation counter, so GPU resources will be updated
    /// on the next render.
    pub fn set_image(&mut self, image: DynamicImage) {
        self.dimensions = Some(image.dimensions());
        self.source = TextureSource::Embedded { image, original_bytes: None };
        self.generation += 1;
    }

    /// Get the source path if this texture was created from a path.
    pub fn source_path(&self) -> Option<&Path> {
        match &self.source {
            TextureSource::Path(path) => Some(path),
            TextureSource::CachedPath { path, .. } => Some(path),
            TextureSource::Embedded { .. } => None,
        }
    }

    /// Get the original compressed bytes if available.
    ///
    /// Returns the original PNG/JPEG bytes if this texture was created with
    /// `from_image_with_original_bytes`. This is used for efficient serialization
    /// to avoid re-encoding the image.
    pub fn original_bytes(&self) -> Option<(&[u8], super::format::TextureFormat)> {
        match &self.source {
            TextureSource::Embedded { original_bytes: Some((bytes, format)), .. } => {
                Some((bytes.as_slice(), *format))
            }
            _ => None,
        }
    }
}
