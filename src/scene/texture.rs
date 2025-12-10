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
    /// Image data embedded in memory (always available)
    Embedded(DynamicImage),
    /// Path to load image from (image loaded on demand)
    Path(PathBuf),
    /// Image was loaded from path, with optional cached data.
    /// The image can be released to save memory and reloaded later if needed.
    CachedPath {
        path: PathBuf,
        image: Option<DynamicImage>,
    },
}

/// GPU resources for a texture.
///
/// These are created lazily when the texture is first needed for rendering.
pub(crate) struct GpuTexture {
    #[allow(unused)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
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
    /// GPU resources (created lazily)
    gpu: Option<GpuTexture>,
    /// True if source data changed since last GPU upload
    dirty: bool,
}

impl Texture {
    /// Depth texture format used for depth buffers.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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
            source: TextureSource::Embedded(image),
            dimensions,
            gpu: None,
            dirty: true,
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
            gpu: None,
            dirty: true,
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
            TextureSource::Embedded(img) => Ok(img),
            TextureSource::CachedPath { image: Some(img), .. } => Ok(img),
            _ => unreachable!("ensure_image_loaded should have loaded the image"),
        }
    }

    /// Ensure the image is loaded into memory.
    fn ensure_image_loaded(&mut self) -> Result<()> {
        match &self.source {
            TextureSource::Embedded(_) => Ok(()),
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

    /// Check if GPU resources need to be created or updated.
    pub(crate) fn needs_gpu_upload(&self) -> bool {
        self.gpu.is_none() || self.dirty
    }

    /// Create or update GPU resources for this texture.
    ///
    /// This method is called automatically by `DrawState::prepare_scene()` before rendering.
    /// After this call, `gpu()` can be used to access the GPU resources.
    ///
    /// # Arguments
    /// * `device` - The wgpu device for creating GPU resources
    /// * `queue` - The wgpu queue for uploading texture data
    ///
    /// # Errors
    /// Returns an error if the texture image cannot be loaded.
    pub(crate) fn ensure_gpu_resources(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        if !self.needs_gpu_upload() {
            return Ok(());
        }

        // Get or load the image
        let img = self.get_image()?;
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.gpu = Some(GpuTexture { texture, view, sampler });
        self.dirty = false;

        Ok(())
    }

    /// Get the GPU resources for this texture.
    ///
    /// # Panics
    /// Panics if GPU resources haven't been initialized yet.
    /// Call `ensure_gpu_resources()` first, or use `has_gpu_resources()` to check.
    pub(crate) fn gpu(&self) -> &GpuTexture {
        self.gpu
            .as_ref()
            .expect("Texture GPU resources not initialized. Call ensure_gpu_resources() first.")
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
    /// This marks the texture as dirty, so GPU resources will be updated
    /// on the next render.
    pub fn set_image(&mut self, image: DynamicImage) {
        self.dimensions = Some(image.dimensions());
        self.source = TextureSource::Embedded(image);
        self.dirty = true;
    }

    /// Get the source path if this texture was created from a path.
    pub fn source_path(&self) -> Option<&Path> {
        match &self.source {
            TextureSource::Path(path) => Some(path),
            TextureSource::CachedPath { path, .. } => Some(path),
            TextureSource::Embedded(_) => None,
        }
    }

    /// Create a depth texture for use as a depth buffer.
    ///
    /// Unlike regular textures, depth textures are always created immediately
    /// with GPU resources since they are internal rendering resources.
    pub(crate) fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> GpuTexture {
        let size = wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        GpuTexture { texture, view, sampler }
    }
}
