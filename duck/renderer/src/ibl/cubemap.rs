//! Cubemap texture infrastructure for IBL.
//!
//! Cubemaps are used to store environment maps, irradiance maps, and pre-filtered
//! specular maps for image-based lighting.

use duck_engine_scene::{CubemapFaceData, PreprocessedCubemap, CUBEMAP_FACES};

/// GPU resources for a cubemap texture.
///
/// Cubemaps consist of 6 square faces representing the +X, -X, +Y, -Y, +Z, -Z directions.
/// The `view` is a cube view suitable for sampling in shaders.
#[derive(Debug)]
pub struct GpuCubemap {
    /// The underlying wgpu texture (2D array with 6 layers).
    pub texture: wgpu::Texture,
    /// Cube view for sampling the entire cubemap.
    pub view: wgpu::TextureView,
    /// Sampler configured for cubemap sampling.
    pub sampler: wgpu::Sampler,
}

impl GpuCubemap {
    /// Create a new cubemap texture.
    ///
    /// # Arguments
    /// * `device` - The wgpu device
    /// * `size` - Width and height of each face in pixels
    /// * `format` - Texture format (typically Rgba16Float for HDR)
    /// * `mip_levels` - Number of mip levels (1 for no mipmaps)
    /// * `usage` - Texture usage flags
    /// * `label` - Debug label for the texture
    pub fn new(
        device: &wgpu::Device,
        size: u32,
        format: wgpu::TextureFormat,
        mip_levels: u32,
        usage: wgpu::TextureUsages,
        label: &str,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        // Create cube view for sampling
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{} Cube View", label)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });

        // Create sampler with linear filtering for smooth cubemap sampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} Sampler", label)),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: if mip_levels > 1 {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    /// Read back cubemap texture data from the GPU to a [`PreprocessedCubemap`].
    ///
    /// Uses `copy_texture_to_buffer` + buffer mapping to read each face at each mip level.
    /// This is a blocking operation that polls the device until the readback completes.
    ///
    /// The `bytes_per_pixel` must match the texture format (e.g. 8 for Rgba16Float).
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        face_size: u32,
        mip_levels: u32,
        bytes_per_pixel: u32,
    ) -> PreprocessedCubemap {
        let mut mip_data = Vec::with_capacity(mip_levels as usize);

        for mip in 0..mip_levels {
            let mip_size = face_size >> mip;
            let unpadded_bytes_per_row = mip_size * bytes_per_pixel;
            // wgpu requires rows aligned to COPY_BYTES_PER_ROW_ALIGNMENT (256)
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
            let buffer_size = (padded_bytes_per_row * mip_size) as u64;

            let mut faces: [CubemapFaceData; CUBEMAP_FACES] =
                std::array::from_fn(|_| Vec::new());

            for face in 0..CUBEMAP_FACES as u32 {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Cubemap Readback Mip {} Face {}", mip, face)),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Cubemap Readback Encoder"),
                });

                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(padded_bytes_per_row),
                            rows_per_image: Some(mip_size),
                        },
                    },
                    wgpu::Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    },
                );

                queue.submit(std::iter::once(encoder.finish()));

                // Map and read
                let (sender, receiver) = std::sync::mpsc::channel();
                staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap();
                });
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                receiver.recv().unwrap().unwrap();

                let mapped = staging.slice(..).get_mapped_range();

                // Strip row padding if present
                if padded_bytes_per_row == unpadded_bytes_per_row {
                    faces[face as usize] = mapped.to_vec();
                } else {
                    let mut face_data = Vec::with_capacity((unpadded_bytes_per_row * mip_size) as usize);
                    for row in 0..mip_size {
                        let start = (row * padded_bytes_per_row) as usize;
                        let end = start + unpadded_bytes_per_row as usize;
                        face_data.extend_from_slice(&mapped[start..end]);
                    }
                    faces[face as usize] = face_data;
                }

                drop(mapped);
                staging.unmap();
            }

            mip_data.push(faces);
        }

        PreprocessedCubemap {
            face_size,
            mip_levels,
            mip_data,
        }
    }

    /// Create a 2D array view suitable for compute shader storage texture binding.
    ///
    /// This creates a view of a single mip level as a 2D array with 6 layers,
    /// which can be bound as a storage texture for compute shaders to write to.
    pub fn create_storage_view(
        &self,
        mip_level: u32,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("Cubemap Storage View Mip {}", mip_level)),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        })
    }
}
