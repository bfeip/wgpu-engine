use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::scene::{MaterialProperties, PrimitiveType, SceneProperties};

use super::state::GpuTexture;
use super::uniforms::{CameraUniform, LightsArrayUniform};

/// GPU resources for camera view/projection uniforms.
pub(in crate::renderer) struct CameraResources {
    pub(in crate::renderer) buffer: wgpu::Buffer,
    pub(in crate::renderer) bind_group_layout: wgpu::BindGroupLayout,
    pub(in crate::renderer) bind_group: wgpu::BindGroup,
}

impl CameraResources {
    /// Create camera resources including bind group layout.
    pub(in crate::renderer) fn new(device: &wgpu::Device) -> CameraResources {
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
pub(in crate::renderer) struct LightResources {
    pub(in crate::renderer) buffer: wgpu::Buffer,
    pub(in crate::renderer) bind_group_layout: wgpu::BindGroupLayout,
    pub(in crate::renderer) bind_group: wgpu::BindGroup,
    pub(in crate::renderer) synced_generation: u64,
}

impl LightResources {
    /// Create light resources including bind group layout.
    pub(in crate::renderer) fn new(device: &wgpu::Device) -> LightResources {
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
pub(in crate::renderer) struct MaterialBindGroupLayouts {
    pub(in crate::renderer) color: wgpu::BindGroupLayout,
    pub(in crate::renderer) pbr: wgpu::BindGroupLayout,
}

impl MaterialBindGroupLayouts {
    /// Create bind group layouts for all material types.
    pub(in crate::renderer) fn new(device: &wgpu::Device) -> MaterialBindGroupLayouts {
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
pub(in crate::renderer) struct MaterialPipelineLayouts {
    pub(in crate::renderer) color: wgpu::PipelineLayout,
    pub(in crate::renderer) pbr: wgpu::PipelineLayout,
    /// PBR with IBL (includes environment bind group)
    pub(in crate::renderer) pbr_ibl: wgpu::PipelineLayout,
}

impl MaterialPipelineLayouts {
    /// Create pipeline layouts for all material types.
    pub(in crate::renderer) fn new(
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

/// Default fallback textures for rendering.
pub(in crate::renderer) struct DefaultTextures {
    pub(in crate::renderer) depth: GpuTexture,
    pub(in crate::renderer) white: GpuTexture,
    pub(in crate::renderer) normal: GpuTexture,
    /// Multisampled color attachment for MSAA rendering. None when sample_count == 1.
    pub(in crate::renderer) msaa_color_attachment: Option<GpuTexture>,
}

impl DefaultTextures {
    /// Create default fallback textures for rendering.
    pub(in crate::renderer) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> DefaultTextures {
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

        DefaultTextures { depth, white, normal, msaa_color_attachment }
    }
}

/// Cache key for render pipelines, combining all properties that require
/// a distinct compiled pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(in crate::renderer) struct PipelineCacheKey {
    pub(in crate::renderer) material_props: MaterialProperties,
    pub(in crate::renderer) scene_props: SceneProperties,
    pub(in crate::renderer) primitive_type: PrimitiveType,
    pub(in crate::renderer) depth_prepass: bool,
}
