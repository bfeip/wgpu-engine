use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::ibl::ibl_bind_group_layout;
use crate::render_core::GpuTexture;
use crate::scene::{MaterialProperties, PrimitiveType, SceneProperties};

use super::uniforms::{CameraUniform, LightsArrayUniform};

/// GPU instance data for one camera: a uniform buffer plus its bind group.
pub(crate) struct CameraResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl CameraResources {
    /// Create a camera buffer and bind group against the shared camera layout.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> CameraResources {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        CameraResources { buffer, bind_group }
    }
}

/// GPU instance data for lighting uniforms: a buffer plus its bind group.
pub(crate) struct LightResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub synced_generation: u64,
}

impl LightResources {
    /// Create the light buffer and bind group against the shared light layout.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> LightResources {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<LightsArrayUniform>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("Light bind group"),
        });

        LightResources {
            buffer,
            bind_group,
            synced_generation: 0,
        }
    }
}

/// Canonical owner of every bind group layout used by the renderer.
///
/// A bind group layout is a *schema*: exactly one per kind, created once and
/// shared by every pipeline layout and every conforming bind group. Keeping them
/// all here (rather than bundled into the resource structs that happen to create
/// the first bind group of each kind) lets multiple instances — e.g. the main
/// camera plus per-sub-view camera slots — share a single layout, and gives
/// pipeline-layout construction one place to borrow from.
pub(crate) struct BindGroupLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub light: wgpu::BindGroupLayout,
    /// Color material layout.
    pub color: wgpu::BindGroupLayout,
    /// PBR material layout.
    pub pbr: wgpu::BindGroupLayout,
    pub ibl: wgpu::BindGroupLayout,
}

impl BindGroupLayouts {
    /// Create all bind group layouts.
    pub fn new(device: &wgpu::Device) -> BindGroupLayouts {
        let camera = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let light = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let ibl = ibl_bind_group_layout(device);

        BindGroupLayouts { camera, light, color, pbr, ibl }
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
    pub fn new(device: &wgpu::Device, layouts: &BindGroupLayouts) -> MaterialPipelineLayouts {
        let color = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Color Material Pipeline Layout"),
            bind_group_layouts: &[&layouts.camera, &layouts.light, &layouts.color],
            push_constant_ranges: &[],
        });

        let pbr = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Material Pipeline Layout"),
            bind_group_layouts: &[&layouts.camera, &layouts.light, &layouts.pbr],
            push_constant_ranges: &[],
        });

        let pbr_ibl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR IBL Material Pipeline Layout"),
            bind_group_layouts: &[&layouts.camera, &layouts.light, &layouts.pbr, &layouts.ibl],
            push_constant_ranges: &[],
        });

        MaterialPipelineLayouts { color, pbr, pbr_ibl }
    }
}

/// Fallback textures bound when a material has no texture of its own.
///
// NOTE: this struct existing as a Renderer-owned grab bag is a symptom of ownership
// confusion — fallback bindings are a concern of the material system, not the
// renderer. It should be dissolved into the material system as soon as that
// owner exists; do not add to it.
pub(crate) struct FallbackTextures {
    pub white: GpuTexture,
    pub default_normal: GpuTexture,
}

impl FallbackTextures {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> FallbackTextures {
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

        FallbackTextures { white, default_normal: normal }
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
