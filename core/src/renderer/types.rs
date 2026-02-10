use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    scene::{MaterialProperties, PrimitiveType, SceneProperties, Vertex},
};

use super::gpu_resources::{CameraUniform, GpuInstance, GpuTexture, LightsArrayUniform};

// Vertex shader attribute locations
pub(crate) enum VertexShaderLocations {
    VertexPosition = 0,
    TextureCoords,
    VertexNormal,
    InstanceTransformRow0,
    InstanceTransformRow1,
    InstanceTransformRow2,
    InstanceTransformRow3,
    InstanceNormalRow0,
    InstanceNormalRow1,
    InstanceNormalRow2,
}

/// Returns the vertex buffer layout for Vertex structs.
///
/// This describes how vertex data is laid out in GPU memory and maps to shader locations.
pub(crate) fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: VertexShaderLocations::VertexPosition as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: VertexShaderLocations::TextureCoords as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3 * 2]>() as wgpu::BufferAddress,
                shader_location: VertexShaderLocations::VertexNormal as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Returns the instance buffer layout for GpuInstance structs.
///
/// This describes how instance data is laid out in GPU memory for instanced rendering.
pub(crate) fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    use VertexShaderLocations as VSL;

    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuInstance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: VSL::InstanceTransformRow0 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 1]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow1 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 2]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow2 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 3]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceTransformRow3 as u32,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 4 * 4]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow0 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; (4 * 4) + (3 * 1)]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow1 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; (4 * 4) + (3 * 2)]>() as wgpu::BufferAddress,
                shader_location: VSL::InstanceNormalRow2 as u32,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// Maximum texture dimension for WebGL. When the canvas exceeds this size,
/// we scale down the surface while preserving aspect ratio.
#[cfg(target_arch = "wasm32")]
pub(super) const MAX_TEXTURE_DIMENSION: u32 = 2048;

/// Camera state and GPU resources for view/projection uniforms.
pub(super) struct CameraResources {
    pub(super) camera: Camera,
    pub(super) buffer: wgpu::Buffer,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) bind_group: wgpu::BindGroup,
}

impl CameraResources {
    /// Create camera resources including bind group layout.
    pub(super) fn new(device: &wgpu::Device, aspect: f32) -> CameraResources {
        let camera = Camera {
            eye: (0.0, 0.1, 0.2).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect,
            fovy: 45.0,
            znear: 0.001,
            zfar: 100.0,
            ortho: false,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
            camera,
            buffer,
            bind_group_layout,
            bind_group,
        }
    }
}

/// GPU resources for lighting uniform data.
pub(super) struct LightResources {
    pub(super) buffer: wgpu::Buffer,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) bind_group: wgpu::BindGroup,
}

impl LightResources {
    /// Create light resources including bind group layout.
    pub(super) fn new(device: &wgpu::Device) -> LightResources {
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
        }
    }
}

/// Bind group layouts for different material types.
pub(super) struct MaterialBindGroupLayouts {
    pub(super) color: wgpu::BindGroupLayout,
    pub(super) texture: wgpu::BindGroupLayout,
    pub(super) pbr: wgpu::BindGroupLayout,
}

impl MaterialBindGroupLayouts {
    /// Create bind group layouts for all material types.
    pub(super) fn new(device: &wgpu::Device) -> MaterialBindGroupLayouts {
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

        let texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Material Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
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

        MaterialBindGroupLayouts { color, texture, pbr }
    }
}

/// Pipeline layouts for different material types.
pub(super) struct MaterialPipelineLayouts {
    pub(super) color: wgpu::PipelineLayout,
    pub(super) texture: wgpu::PipelineLayout,
    pub(super) pbr: wgpu::PipelineLayout,
    /// PBR with IBL (includes environment bind group)
    pub(super) pbr_ibl: wgpu::PipelineLayout,
}

impl MaterialPipelineLayouts {
    /// Create pipeline layouts for all material types.
    pub(super) fn new(
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

        let texture = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Texture Material Pipeline Layout"),
            bind_group_layouts: &[
                camera_bind_group_layout,
                light_bind_group_layout,
                &material_layouts.texture,
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

        MaterialPipelineLayouts { color, texture, pbr, pbr_ibl }
    }
}

/// Default fallback textures for rendering.
pub(super) struct DefaultTextures {
    pub(super) depth: GpuTexture,
    pub(super) white: GpuTexture,
    pub(super) normal: GpuTexture,
}

impl DefaultTextures {
    /// Create default fallback textures for rendering.
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> DefaultTextures {
        use super::gpu_resources::{create_depth_texture, create_solid_color_texture};

        let depth = create_depth_texture(device, config, "depth_texture");
        let white = create_solid_color_texture(
            device,
            queue,
            [255, 255, 255, 255],
            "default_white_texture",
        );
        let normal = create_solid_color_texture(
            device,
            queue,
            [128, 128, 255, 255], // Neutral normal (0.5, 0.5, 1.0) in tangent space
            "default_normal_texture",
        );

        DefaultTextures { depth, white, normal }
    }
}

/// Cache key for render pipelines, combining all properties that require
/// a distinct compiled pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PipelineCacheKey {
    pub(super) material_props: MaterialProperties,
    pub(super) scene_props: SceneProperties,
    pub(super) primitive_type: PrimitiveType,
}

/// Clamp dimensions to the maximum texture size while preserving aspect ratio.
/// On native platforms, returns the input dimensions unchanged.
pub(super) fn clamp_surface_size(width: u32, height: u32) -> (u32, u32) {
    #[cfg(target_arch = "wasm32")]
    {
        if width <= MAX_TEXTURE_DIMENSION && height <= MAX_TEXTURE_DIMENSION {
            return (width, height);
        }

        let scale = if width >= height {
            MAX_TEXTURE_DIMENSION as f32 / width as f32
        } else {
            MAX_TEXTURE_DIMENSION as f32 / height as f32
        };

        let new_width = ((width as f32 * scale).round() as u32).max(1);
        let new_height = ((height as f32 * scale).round() as u32).max(1);
        (new_width, new_height)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        (width, height)
    }
}
