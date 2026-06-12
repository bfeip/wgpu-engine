/// An offscreen render target with a staging buffer for CPU readback.
///
/// Render into [`view`](Self::view), call [`encode_copy`](Self::encode_copy)
/// on the same encoder after drawing, submit, then [`read`](Self::read) the
/// tightly-packed pixel data. Reused across frames at the same size.
///
/// Assumes a 4-byte-per-pixel color format (e.g. `Rgba8UnormSrgb`).
///
/// # Example
///
/// ```no_run
/// use duck_engine_render_core::ReadbackTarget;
///
/// fn render_to_pixels(
///     device: &wgpu::Device,
///     queue: &wgpu::Queue,
/// ) -> anyhow::Result<Vec<u8>> {
///     let target = ReadbackTarget::new(device, 800, 600, wgpu::TextureFormat::Rgba8UnormSrgb);
///
///     let mut encoder = device.create_command_encoder(&Default::default());
///     // ... encode render passes drawing into target.view() ...
///     target.encode_copy(&mut encoder);
///     queue.submit(std::iter::once(encoder.finish()));
///
///     // Blocks until the GPU work completes; returns 800 * 600 * 4 bytes.
///     target.read(device)
/// }
/// ```
pub struct ReadbackTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    size: (u32, u32),
}

impl ReadbackTarget {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Readback Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // wgpu requires buffer copy rows to be aligned to COPY_BYTES_PER_ROW_ALIGNMENT (256).
        // We pad each row to meet this alignment, then strip the padding when reading back.
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            texture,
            view,
            staging_buffer,
            padded_bytes_per_row,
            size: (width, height),
        }
    }

    /// The render target view to draw into.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The target dimensions as (width, height).
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Encode the texture-to-staging-buffer copy. Call after all draws into
    /// [`view`](Self::view) have been encoded, before submitting the encoder.
    pub fn encode_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        let (width, height) = self.size;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Map the staging buffer and read back the pixels, stripping row padding.
    ///
    /// Blocks until the GPU work feeding the staging buffer has completed.
    /// Returns tightly-packed pixel data (width × height × 4 bytes).
    pub fn read(&self, device: &wgpu::Device) -> anyhow::Result<Vec<u8>> {
        let (width, height) = self.size;
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| anyhow::anyhow!("Failed to poll device for readback: {e:?}"))?;
        receiver.recv()??;

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let data = buffer_slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * self.padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        self.staging_buffer.unmap();

        Ok(pixels)
    }
}
