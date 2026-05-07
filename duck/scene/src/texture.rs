use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};

/// Unique identifier for a texture in the scene.
pub type TextureId = crate::Id;

/// Texture image format.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureFormat {
    Png = 0,
    Jpeg = 1,
    Raw = 2,
}

impl TryFrom<image::ImageFormat> for TextureFormat {
    type Error = anyhow::Error;

    fn try_from(format: image::ImageFormat) -> Result<Self> {
        match format {
            image::ImageFormat::Png => Ok(TextureFormat::Png),
            image::ImageFormat::Jpeg => Ok(TextureFormat::Jpeg),
            _ => anyhow::bail!("unsupported image format: {:?}", format),
        }
    }
}

/// Describes how texture source data is stored.
///
/// Textures can be created from embedded image data or loaded lazily from a file path.
pub enum TextureSource {
    /// Image data embedded in memory.
    /// Optionally includes original compressed bytes (PNG/JPEG) for efficient serialization.
    Embedded {
        image: DynamicImage,
        /// Original compressed bytes preserved from loading (e.g., from glTF).
        original_bytes: Option<(Vec<u8>, TextureFormat)>,
    },
    /// Image loaded lazily from a file path.
    /// The image can be released and reloaded by replacing the `OnceLock`.
    File {
        path: PathBuf,
        cache: OnceLock<DynamicImage>,
    },
}

impl Clone for TextureSource {
    fn clone(&self) -> Self {
        match self {
            TextureSource::Embedded { image, original_bytes } => TextureSource::Embedded {
                image: image.clone(),
                original_bytes: original_bytes.clone(),
            },
            TextureSource::File { path, cache } => TextureSource::File {
                path: path.clone(),
                cache: match cache.get() {
                    Some(img) => {
                        let lock = OnceLock::new();
                        let _ = lock.set(img.clone());
                        lock
                    }
                    None => OnceLock::new(),
                },
            },
        }
    }
}

/// An image texture.
///
/// # Examples
///
/// ```
/// use duck_engine_scene::{Texture, Scene};
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
    pub id: TextureId,
    /// Source data for the texture
    source: TextureSource,
    /// Generation counter - increments on any mutation
    generation: u64,
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            source: self.source.clone(),
            generation: self.generation,
        }
    }
}

impl Texture {
    /// Create a texture from an embedded image.
    ///
    /// # Arguments
    /// * `image` - The image data to use for this texture
    pub fn from_image(image: DynamicImage) -> Self {
        Self {
            id: crate::Id::new(),
            source: TextureSource::Embedded { image, original_bytes: None },
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
        format: TextureFormat,
    ) -> Self {
        Self {
            id: crate::Id::new(),
            source: TextureSource::Embedded {
                image,
                original_bytes: Some((original_bytes, format)),
            },
            generation: 1,
        }
    }

    /// Create a texture from a file path.
    ///
    /// The image is not loaded immediately - it will be loaded on demand when
    /// `get_image()` is called.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            id: crate::Id::new(),
            source: TextureSource::File {
                path: path.into(),
                cache: OnceLock::new(),
            },
            generation: 1,
        }
    }

    /// Load and return a reference to the image data.
    ///
    /// For path-based textures, this loads the image from disk on first access.
    /// The loaded image is cached for future access.
    ///
    /// # Errors
    /// Returns an error if the image cannot be loaded from the path.
    pub fn get_image(&self) -> Result<&DynamicImage> {
        match &self.source {
            TextureSource::Embedded { image, .. } => Ok(image),
            TextureSource::File { path, cache } => {
                if let Some(img) = cache.get() {
                    return Ok(img);
                }
                let img = image::open(path)
                    .with_context(|| format!("Failed to load texture from {:?}", path))?;
                // If another thread set it first, our image is dropped and
                // we use theirs. This is correct — just a rare double-load.
                let _ = cache.set(img);
                Ok(cache.get().expect("just set or another thread set it"))
            }
        }
    }


    /// Replace the texture's image data.
    pub fn set_image(&mut self, image: DynamicImage) {
        self.source = TextureSource::Embedded { image, original_bytes: None };
        self.generation += 1;
    }

    /// Get the texture's unique identifier.
    pub fn id(&self) -> TextureId {
        self.id
    }

    /// Get the texture dimensions, if known.
    ///
    /// Returns `None` if the texture was created from a path and hasn't been loaded yet.
    /// Call `get_image()` first to trigger loading if dimensions are needed.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        match &self.source {
            TextureSource::Embedded { image, .. } => Some(image.dimensions()),
            TextureSource::File { cache, .. } => cache.get().map(|img| img.dimensions()),
        }
    }

    /// Returns the current generation counter.
    ///
    /// This value increments on any mutation to the texture data.
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
    pub fn release_image_cache(&mut self) {
        if let TextureSource::File { path, .. } = &self.source {
            let path = path.clone();
            self.source = TextureSource::File {
                path,
                cache: OnceLock::new(),
            };
        }
        // Embedded textures cannot release their image data
    }

    /// Get the source path if this texture was created from a path.
    pub fn source_path(&self) -> Option<&Path> {
        match &self.source {
            TextureSource::File { path, .. } => Some(path),
            TextureSource::Embedded { .. } => None,
        }
    }

    /// Get the original compressed bytes if available.
    ///
    /// Returns the original compressed bytes if this texture was created with
    /// `from_image_with_original_bytes`. This is used for efficient serialization
    /// to avoid re-encoding the image.
    pub fn original_bytes(&self) -> Option<(&[u8], TextureFormat)> {
        match &self.source {
            TextureSource::Embedded { original_bytes: Some((bytes, format)), .. } => {
                Some((bytes.as_slice(), *format))
            }
            _ => None,
        }
    }
}

#[cfg(feature = "serde")]
fn detect_texture_format(bytes: &[u8]) -> Option<TextureFormat> {
    image::guess_format(bytes).ok().and_then(|f| TextureFormat::try_from(f).ok())
}

#[cfg(feature = "serde")]
impl serde::Serialize for Texture {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeStruct};
        use std::io::Cursor;

        let image = self.get_image().map_err(Error::custom)?;
        let (width, height) = image.dimensions();

        // Last-resort: PNG-encode the decoded image.
        let encode_png = || -> std::result::Result<Vec<u8>, S::Error> {
            let mut buf = Vec::new();
            image
                .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .map_err(Error::custom)?;
            Ok(buf)
        };

        let (data, format) = if let Some((bytes, fmt)) = self.original_bytes() {
            // Best case: original compressed bytes preserved from load (e.g. from glTF).
            (bytes.to_vec(), fmt)
        } else if let Some(path) = self.source_path() {
            // File texture: read the already-compressed file bytes directly.
            let bytes = std::fs::read(path).map_err(Error::custom)?;
            if let Some(fmt) = detect_texture_format(&bytes) {
                (bytes, fmt)
            } else {
                (encode_png()?, TextureFormat::Png)
            }
        } else {
            // Decoded-only texture: re-encode as PNG.
            (encode_png()?, TextureFormat::Png)
        };

        let mut s = serializer.serialize_struct("Texture", 5)?;
        s.serialize_field("id", &self.id)?;
        s.serialize_field("format", &format)?;
        s.serialize_field("width", &width)?;
        s.serialize_field("height", &height)?;
        s.serialize_field("data", &data)?;
        s.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Texture {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        #[derive(serde::Deserialize)]
        struct TextureData {
            id: TextureId,
            format: TextureFormat,
            width: u32,
            height: u32,
            data: Vec<u8>,
        }

        let td = TextureData::deserialize(deserializer)?;

        let (image, original_bytes) = match td.format {
            TextureFormat::Png | TextureFormat::Jpeg => {
                let img = image::load_from_memory(&td.data)
                    .map_err(|e| Error::custom(e.to_string()))?;
                (img, Some((td.data, td.format)))
            }
            TextureFormat::Raw => {
                // Raw RGBA8 data: reconstruct directly, no original_bytes to preserve.
                let rgba = image::RgbaImage::from_raw(td.width, td.height, td.data)
                    .ok_or_else(|| Error::custom("invalid raw image dimensions"))?;
                (DynamicImage::ImageRgba8(rgba), None)
            }
        };

        Ok(Texture {
            id: td.id,
            source: TextureSource::Embedded { image, original_bytes },
            generation: 1,
        })
    }
}
