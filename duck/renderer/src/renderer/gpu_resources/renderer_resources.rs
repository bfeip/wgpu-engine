use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::scene::{MaterialProperties, PrimitiveType, SceneProperties};

use super::state::GpuTexture;
use super::uniforms::{CameraUniform, LightsArrayUniform};

/// GPU resources for camera view/projection uniforms.
pub(crate) struct CameraResources {
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl CameraResources {
    /// Create camera resources including bind group layout.
    pub fn new(device: &wgpu::Device) -> CameraResources {
        let camera_uniform = CameraUniform::new();

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Visible in both VERTEX (for view_proj) and FRAGMENT (for eye_position in PBR)
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        CameraResources {
            buffer,
            bind_group_layout,
            bind_group,
        }
    }
}

/// GPU resources for lighting uniform data.
pub(crate) struct LightResources {
    pub buffer: wgpu::Buffer,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub synced_generation: u64,
}

impl LightResources {
    /// Create light resources including bind group layout.
    pub fn new(device: &wgpu::Device) -> LightResources {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<LightsArrayUniform>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Light bind group layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("Light bind group"),
        });

        LightResources {
            buffer,
            bind_group_layout,
            bind_group,
            synced_generation: 0,
        }
    }
}

/// Bind group layouts for different material types.
pub(crate) struct MaterialBindGroupLayouts {
    pub color: wgpu::BindGroupLayout,
    pub pbr: wgpu::BindGroupLayout,
}

impl MaterialBindGroupLayouts {
    /// Create bind group layouts for all material types.
    pub fn new(device: &wgpu::Device) -> MaterialBindGroupLayouts {
        let color = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Material Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pbr = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBR Material Bind Group Layout"),
            entries: &[
                // Binding 0: PbrUniform (factors + flags)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Base color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 2: Base color sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 3: Normal texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 4: Normal sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 5: Metallic-roughness texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Binding 6: Metallic-roughness sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        MaterialBindGroupLayouts { color, pbr }
    }
}

/// Pipeline layouts for different material types.
pub(crate) struct MaterialPipelineLayouts {
    pub color: wgpu::PipelineLayout,
    pub pbr: wgpu::PipelineLayout,
    /// PBR with IBL (includes environment bind group)
    pub pbr_ibl: wgpu::PipelineLayout,
}

impl MaterialPipelineLayouts {
    /// Create pipeline layouts for all material types.
    pub fn new(
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        light_bind_group_layout: &wgpu::BindGroupLayout,
        material_layouts: &MaterialBindGroupLayouts,
        ibl_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> MaterialPipelineLayouts {
        let color = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Color Material Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                light_bind_group_layout,
                &material_layouts.color,
            ],
            push_constant_ranges: &[],
        });

        let pbr = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Material Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                light_bind_group_layout,
                &material_layouts.pbr,
            ],
            push_constant_ranges: &[],
        });

        let pbr_ibl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR IBL Material Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                light_bind_group_layout,
                &material_layouts.pbr,
                ibl_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        MaterialPipelineLayouts { color, pbr, pbr_ibl }
    }
}

/// Textures used as part of the rendering process.
pub(crate) struct RendererTextures {
    pub depth: GpuTexture,
    pub white: GpuTexture,
    pub default_normal: GpuTexture,
    /// Multisampled color attachment for MSAA rendering. None when sample_count == 1.
    pub msaa_color_attachment: Option<GpuTexture>,
}

impl RendererTextures {
    /// Create textures for rendering.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> RendererTextures {
        let depth = GpuTexture::depth(device, config, sample_count, "depth_texture");
        let white = GpuTexture::solid_color(
            device,
            queue,
            [255, 255, 255, 255],
            "default_white_texture",
        );
        let normal = GpuTexture::solid_color(
            device,
            queue,
            [128, 128, 255, 255], // Neutral normal (0.5, 0.5, 1.0) in tangent space
            "default_normal_texture",
        );
        let msaa_color_attachment = if sample_count > 1 {
            Some(GpuTexture::color_attachment(device, config, sample_count, "msaa_color_attachment"))
        } else {
            None
        };

        RendererTextures { depth, white, default_normal: normal, msaa_color_attachment }
    }

    /// Returns `(render_view, resolve_target)` for a render pass that may use MSAA.
    ///
    /// When MSAA is active, the pass should render into `render_view` (the multisampled
    /// attachment) and resolve into `target` (typically the swapchain). When MSAA is
    /// inactive, renders directly into `target` with no resolve step.
    pub fn msaa_views<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
    ) -> (&'a wgpu::TextureView, Option<&'a wgpu::TextureView>) {
        match &self.msaa_color_attachment {
            Some(msaa) => (&msaa.view, Some(target)),
            None => (target, None),
        }
    }
}

/// Cached GPU resources for headless rendering, reused across frames at the same size.
pub(crate) struct HeadlessResources {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub staging_buffer: wgpu::Buffer,
    pub padded_bytes_per_row: u32,
    pub size: (u32, u32),
}

impl HeadlessResources {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Headless Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // wgpu requires buffer copy rows to be aligned to COPY_BYTES_PER_ROW_ALIGNMENT (256).
        // We pad each row to meet this alignment, then strip the padding when reading back.
        let bytes_per_pixel = 4u32; // RGBA8
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = (padded_bytes_per_row * height) as u64;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Headless Staging Buffer"),
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
}

/// Cache key for render pipelines, combining all properties that require
/// a distinct compiled pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineCacheKey {
    pub material_props: MaterialProperties,
    pub scene_props: SceneProperties,
    pub primitive_type: PrimitiveType,
    pub depth_prepass: bool,
}
