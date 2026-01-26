use std::collections::HashMap;

use anyhow::Result;
use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::{
    camera::{Camera, CameraUniform},
    ibl::IblResources,
    scene::{
        batch::partition_batches, GpuTexture, InstanceRaw, LightsArrayUniform,
        MaterialGpuResources, MaterialProperties, PrimitiveType, Scene, SceneProperties, Texture,
        Vertex,
    },
    selection::SelectionManager,
    shaders::ShaderGenerator,
};

/// Maximum texture dimension for WebGL. When the canvas exceeds this size,
/// we scale down the surface while preserving aspect ratio.
#[cfg(target_arch = "wasm32")]
const MAX_TEXTURE_DIMENSION: u32 = 2048;

// =============================================================================
// Grouping Structs
// =============================================================================

/// Camera state and GPU resources for view/projection uniforms.
struct CameraResources {
    camera: Camera,
    buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

/// GPU resources for lighting uniform data.
struct LightResources {
    buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

/// Bind group layouts for different material types.
struct MaterialBindGroupLayouts {
    color: wgpu::BindGroupLayout,
    texture: wgpu::BindGroupLayout,
    pbr: wgpu::BindGroupLayout,
}

/// Pipeline layouts for different material types.
struct MaterialPipelineLayouts {
    color: wgpu::PipelineLayout,
    texture: wgpu::PipelineLayout,
    pbr: wgpu::PipelineLayout,
    /// PBR with IBL (includes environment bind group)
    pbr_ibl: wgpu::PipelineLayout,
}

/// Default fallback textures for rendering.
struct DefaultTextures {
    depth: GpuTexture,
    white: GpuTexture,
    normal: GpuTexture,
}

/// GPU uniform data for screen-space outline rendering.
/// Must match the layout in outline_screenspace.wesl.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OutlineUniform {
    color: [f32; 4],
    width_pixels: f32,
    screen_width: f32,
    screen_height: f32,
    _padding: f32,
}

/// GPU resources for screen-space selection outline rendering.
struct OutlineResources {
    /// Mask texture for selected objects (R8Unorm)
    mask_texture: wgpu::Texture,
    mask_view: wgpu::TextureView,
    /// Pipeline for rendering selected objects to mask texture
    mask_pipeline: wgpu::RenderPipeline,
    /// Pipeline for fullscreen outline post-process
    outline_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer for outline settings
    uniform_buffer: wgpu::Buffer,
    /// Bind group layout for post-process shader
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group containing mask texture, sampler, and uniforms
    bind_group: wgpu::BindGroup,
    /// Sampler for mask texture
    mask_sampler: wgpu::Sampler,
}

/// Cache key for render pipelines, combining all properties that require
/// a distinct compiled pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineCacheKey {
    material_props: MaterialProperties,
    scene_props: SceneProperties,
    primitive_type: PrimitiveType,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Clamp dimensions to the maximum texture size while preserving aspect ratio.
/// On native platforms, returns the input dimensions unchanged.
fn clamp_surface_size(width: u32, height: u32) -> (u32, u32) {
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

pub(crate) struct Renderer<'a> {
    // Core GPU resources
    pub surface: wgpu::Surface<'a>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),
    /// Current cursor position in screen coordinates (x, y), or None if cursor is not over the window
    pub cursor_position: Option<(f32, f32)>,

    // Grouped resources
    camera_resources: CameraResources,
    lights: LightResources,
    material_layouts: MaterialBindGroupLayouts,
    pipelines: MaterialPipelineLayouts,
    default_textures: DefaultTextures,

    // Other
    shader_generator: ShaderGenerator,
    pipeline_cache: HashMap<PipelineCacheKey, wgpu::RenderPipeline>,
    ibl_resources: IblResources,
    outline_resources: OutlineResources,
}

impl<'a> Renderer<'a> {
    // Creating some of the wgpu types requires async code
    // The target parameter can be a Window, Canvas, or any type implementing the necessary traits
    pub async fn new<T>(target: T, width: u32, height: u32) -> Renderer<'a>
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let size = clamp_surface_size(width, height);

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(target).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Prefer Fifo (vsync) to avoid tearing/flickering, fallback to first available
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.0,
            height: size.1,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // Create grouped resources
        let material_layouts = Self::create_material_bind_group_layouts(&device);
        let camera_resources =
            Self::create_camera_resources(&device, config.width as f32 / config.height as f32);
        let lights = Self::create_light_resources(&device);
        let ibl_resources = IblResources::new(&device, &queue);
        let pipelines = Self::create_pipeline_layouts(
            &device,
            &camera_resources.bind_group_layout,
            &lights.bind_group_layout,
            &material_layouts,
            &ibl_resources.bind_group_layout,
        );
        let default_textures = Self::create_default_textures(&device, &queue, &config);
        let mut shader_generator = ShaderGenerator::new();
        let outline_resources = Self::create_outline_resources(
            &device,
            &config,
            &camera_resources.bind_group_layout,
            &mut shader_generator,
        );

        Self {
            surface,
            device,
            queue,
            config,
            size,
            cursor_position: None,
            camera_resources,
            lights,
            material_layouts,
            pipelines,
            default_textures,
            shader_generator,
            pipeline_cache: HashMap::new(),
            ibl_resources,
            outline_resources,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get a reference to the camera.
    pub fn camera(&self) -> &Camera {
        &self.camera_resources.camera
    }

    /// Get a mutable reference to the camera.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera_resources.camera
    }

    // =========================================================================
    // Initialization Helpers
    // =========================================================================

    /// Create bind group layouts for all material types.
    fn create_material_bind_group_layouts(device: &wgpu::Device) -> MaterialBindGroupLayouts {
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

    /// Create camera resources including bind group layout.
    fn create_camera_resources(device: &wgpu::Device, aspect: f32) -> CameraResources {
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

    /// Create light resources including bind group layout.
    fn create_light_resources(device: &wgpu::Device) -> LightResources {
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

    /// Create pipeline layouts for all material types.
    fn create_pipeline_layouts(
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

    /// Create default fallback textures for rendering.
    fn create_default_textures(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> DefaultTextures {
        let depth = Texture::create_depth_texture(device, config, "depth_texture");
        let white = Texture::create_solid_color_texture(
            device,
            queue,
            [255, 255, 255, 255],
            "default_white_texture",
        );
        let normal = Texture::create_solid_color_texture(
            device,
            queue,
            [128, 128, 255, 255], // Neutral normal (0.5, 0.5, 1.0) in tangent space
            "default_normal_texture",
        );

        DefaultTextures { depth, white, normal }
    }

    /// Create GPU resources for screen-space selection outline rendering.
    fn create_outline_resources(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
    ) -> OutlineResources {
        // Create mask texture (R8Unorm for storing selection mask)
        let mask_texture =
            Texture::create_mask(device, config.width, config.height, "Outline Mask Texture");
        let mask_view = mask_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create sampler for mask texture
        let mask_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Outline Mask Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create uniform buffer with default values
        let uniform = OutlineUniform {
            color: [1.0, 0.6, 0.0, 1.0], // Orange
            width_pixels: 3.0,
            screen_width: config.width as f32,
            screen_height: config.height as f32,
            _padding: 0.0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Outline Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout for screenspace outline shader
        // binding 0: mask texture, binding 1: sampler, binding 2: uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline Screenspace Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Outline Screenspace Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&mask_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Generate mask shader
        let mask_shader = shader_generator
            .generate_outline_mask_shader(device)
            .expect("Failed to generate outline mask shader");

        // Create pipeline layout for mask rendering (camera only)
        let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Outline Mask Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create mask pipeline - renders selected objects to R8Unorm texture
        let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Outline Mask Pipeline"),
            layout: Some(&mask_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mask_shader,
                entry_point: Some("vs_mask"),
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mask_shader,
                entry_point: Some("fs_mask"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: Texture::MASK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Generate screenspace outline shader
        let outline_shader = shader_generator
            .generate_outline_screenspace_shader(device)
            .expect("Failed to generate outline screenspace shader");

        // Create pipeline layout for screenspace outline (just the bind group)
        let outline_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Outline Screenspace Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create screenspace outline pipeline - fullscreen post-process
        let outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Outline Screenspace Pipeline"),
            layout: Some(&outline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[], // No vertex buffer - generates fullscreen triangle from vertex index
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_outline"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // No depth test for fullscreen pass
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        OutlineResources {
            mask_texture,
            mask_view,
            mask_pipeline,
            outline_pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            mask_sampler,
        }
    }

    // =========================================================================
    // Scene Preparation
    // =========================================================================

    /// Prepare all GPU resources for a scene before rendering.
    ///
    /// This method ensures all textures, materials, and meshes have their GPU resources
    /// created or updated as needed. It should be called before `render_scene_to_view()`.
    //
    // TODO: This iterates more or less everything in the scene. For performance in the future,
    // we should keep track of the need for these updates in the scene. I.e. mark things as
    // dirty if they need to be reified.
    pub fn prepare_scene(&mut self, scene: &mut Scene) -> Result<()> {
        // 0. Reify any unreified annotations (creates meshes/materials/nodes)
        scene.reify_annotations();

        // 1. Prepare all textures first (materials depend on them)
        for texture in scene.textures.values_mut() {
            if texture.needs_gpu_upload() {
                texture.ensure_gpu_resources(&self.device, &self.queue)?;
            }
        }

        // 2. Prepare all materials
        // We need to collect material IDs first to avoid borrow issues
        let material_ids: Vec<_> = scene.materials.keys().copied().collect();
        for mat_id in material_ids {
            // Check each primitive type
            for prim_type in [
                PrimitiveType::TriangleList,
                PrimitiveType::LineList,
                PrimitiveType::PointList,
            ] {
                let needs_update = scene
                    .materials
                    .get(&mat_id)
                    .map(|m| m.needs_gpu_resources(prim_type) && m.has_primitive_data(prim_type))
                    .unwrap_or(false);

                if needs_update {
                    self.prepare_material_primitive(scene, mat_id, prim_type)?;
                }
            }
        }

        // 3. Prepare all meshes
        for mesh in scene.meshes.values_mut() {
            if mesh.needs_gpu_upload() {
                mesh.ensure_gpu_resources(&self.device);
            }
        }

        // 4. Process environment maps for IBL
        if let Some(env_id) = scene.active_environment_map {
            if let Some(env_map) = scene.environment_maps.get_mut(&env_id) {
                if env_map.needs_generation() {
                    self.ibl_resources
                        .process_environment(&self.device, &self.queue, env_map)?;
                }
            }
        }

        Ok(())
    }

    /// Resolve a texture from the scene, falling back to a default texture if not found.
    fn resolve_texture_or_default<'b>(
        &'b self,
        scene: &'b Scene,
        texture_id: Option<u32>,
        default: &'b GpuTexture,
        name: &str,
    ) -> Result<(&'b wgpu::TextureView, &'b wgpu::Sampler)> {
        if let Some(tex_id) = texture_id {
            let tex = scene
                .textures
                .get(&tex_id)
                .ok_or_else(|| anyhow::anyhow!("{} texture {} not found", name, tex_id))?;
            let gpu = tex.gpu();
            Ok((&gpu.view, &gpu.sampler))
        } else {
            Ok((&default.view, &default.sampler))
        }
    }

    /// Create GPU resources for a color-based material (uniform buffer + bind group).
    fn create_color_material_resources(
        &self,
        color: &crate::common::RgbaColor,
        label: &str,
    ) -> MaterialGpuResources {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(color),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.material_layouts.color,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        MaterialGpuResources {
            bind_group,
            _buffer: Some(buffer),
        }
    }

    /// Prepare GPU resources for a specific material primitive type.
    fn prepare_material_primitive(
        &self,
        scene: &mut Scene,
        material_id: u32,
        primitive_type: PrimitiveType,
    ) -> Result<()> {
        match primitive_type {
            PrimitiveType::TriangleList => self.prepare_triangle_material(scene, material_id),
            PrimitiveType::LineList => self.prepare_line_material(scene, material_id),
            PrimitiveType::PointList => self.prepare_point_material(scene, material_id),
        }
    }

    /// Prepare GPU resources for triangle (face) rendering.
    fn prepare_triangle_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();
        let use_pbr =
            material.normal_texture().is_some() || material.metallic_roughness_texture().is_some();

        if use_pbr {
            self.prepare_pbr_material(scene, material_id)
        } else if material.base_color_texture().is_some() {
            self.prepare_textured_material(scene, material_id)
        } else {
            self.prepare_colored_material(scene, material_id)
        }
    }

    /// Prepare GPU resources for PBR material (normal/metallic-roughness textures).
    fn prepare_pbr_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        let pbr_uniform = material.build_pbr_uniform();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PBR Uniform Buffer"),
                contents: bytes_of(&pbr_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (base_color_view, base_color_sampler) = self.resolve_texture_or_default(
            scene,
            material.base_color_texture(),
            &self.default_textures.white,
            "Base color",
        )?;
        let (normal_view, normal_sampler) = self.resolve_texture_or_default(
            scene,
            material.normal_texture(),
            &self.default_textures.normal,
            "Normal",
        )?;
        let (mr_view, mr_sampler) = self.resolve_texture_or_default(
            scene,
            material.metallic_roughness_texture(),
            &self.default_textures.white,
            "Metallic-roughness",
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR Material Bind Group"),
            layout: &self.material_layouts.pbr,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(base_color_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(normal_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(mr_sampler),
                },
            ],
        });

        let material = scene.materials.get_mut(&material_id).unwrap();
        material.set_gpu(
            PrimitiveType::TriangleList,
            MaterialGpuResources {
                bind_group,
                _buffer: Some(buffer),
            },
        );
        material.mark_clean(PrimitiveType::TriangleList);
        Ok(())
    }

    /// Prepare GPU resources for texture-only material (base color texture, no PBR).
    fn prepare_textured_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();
        let texture_id = material.base_color_texture().unwrap();

        let texture = scene.textures.get(&texture_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Texture {} not found for material {}",
                texture_id,
                material_id
            )
        })?;
        let gpu_tex = texture.gpu();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Base Color Texture Bind Group"),
            layout: &self.material_layouts.texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&gpu_tex.sampler),
                },
            ],
        });

        let material = scene.materials.get_mut(&material_id).unwrap();
        material.set_gpu(
            PrimitiveType::TriangleList,
            MaterialGpuResources {
                bind_group,
                _buffer: None,
            },
        );
        material.mark_clean(PrimitiveType::TriangleList);
        Ok(())
    }

    /// Prepare GPU resources for color-only material (base_color_factor, no textures).
    fn prepare_colored_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();
        let gpu_resources =
            self.create_color_material_resources(&material.base_color_factor(), "Base Color Factor");

        let material = scene.materials.get_mut(&material_id).unwrap();
        material.set_gpu(PrimitiveType::TriangleList, gpu_resources);
        material.mark_clean(PrimitiveType::TriangleList);
        Ok(())
    }

    /// Prepare GPU resources for line rendering.
    fn prepare_line_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        if let Some(color) = material.line_color() {
            let gpu_resources = self.create_color_material_resources(&color, "Line Color");

            let material = scene.materials.get_mut(&material_id).unwrap();
            material.set_gpu(PrimitiveType::LineList, gpu_resources);
            material.mark_clean(PrimitiveType::LineList);
        }
        Ok(())
    }

    /// Prepare GPU resources for point rendering.
    fn prepare_point_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        if let Some(color) = material.point_color() {
            let gpu_resources = self.create_color_material_resources(&color, "Point Color");

            let material = scene.materials.get_mut(&material_id).unwrap();
            material.set_gpu(PrimitiveType::PointList, gpu_resources);
            material.mark_clean(PrimitiveType::PointList);
        }
        Ok(())
    }

    fn get_or_create_pipeline(
        &mut self,
        cache_key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        let material_props = cache_key.material_props.clone();
        let scene_props = cache_key.scene_props.clone();
        let primitive_type = cache_key.primitive_type;
        self.pipeline_cache.entry(cache_key).or_insert_with(|| {
            // Select pipeline layout based on material type and scene properties
            let use_pbr = material_props.has_normal_map || material_props.has_metallic_roughness_texture;
            let use_ibl = scene_props.has_ibl && use_pbr && material_props.has_lighting;
            let pipeline_layout = if use_ibl {
                &self.pipelines.pbr_ibl
            } else if use_pbr {
                &self.pipelines.pbr
            } else if material_props.has_base_color_texture {
                &self.pipelines.texture
            } else {
                &self.pipelines.color
            };

            let shader = self
                .shader_generator
                .generate_shader(&self.device, &material_props, &scene_props)
                .expect("Failed to generate shader");

            let topology = match primitive_type {
                PrimitiveType::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                PrimitiveType::LineList => wgpu::PrimitiveTopology::LineList,
                PrimitiveType::PointList => wgpu::PrimitiveTopology::PointList,
            };

            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Vertex::desc(), InstanceRaw::desc()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        // Only cull for triangles, not for lines or points
                        cull_mode: if topology == wgpu::PrimitiveTopology::TriangleList {
                            Some(wgpu::Face::Back)
                        } else {
                            None
                        },
                        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                        polygon_mode: wgpu::PolygonMode::Fill,
                        // Requires Features::DEPTH_CLIP_CONTROL
                        unclipped_depth: false,
                        // Requires Features::CONSERVATIVE_RASTERIZATION
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: Texture::DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                })
        })
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        let (width, height) = new_size;
        if width > 0 && height > 0 {
            let (clamped_width, clamped_height) = clamp_surface_size(width, height);
            self.size = (clamped_width, clamped_height);
            self.config.width = clamped_width;
            self.config.height = clamped_height;
            self.surface.configure(&self.device, &self.config);

            self.camera_resources.camera.aspect = width as f32 / height as f32;

            self.default_textures.depth =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");

            // Recreate mask texture at new size
            self.outline_resources.mask_texture = Texture::create_mask(
                &self.device,
                clamped_width,
                clamped_height,
                "Outline Mask Texture",
            );
            self.outline_resources.mask_view = self
                .outline_resources
                .mask_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate bind group with new texture view
            self.outline_resources.bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Outline Screenspace Bind Group"),
                    layout: &self.outline_resources.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.outline_resources.mask_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.outline_resources.mask_sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.outline_resources.uniform_buffer.as_entire_binding(),
                        },
                    ],
                });
        }
    }

    /// Render the scene to a specific view, updating uniforms and drawing all batches.
    /// If a selection is provided and not empty, selection outlines will be rendered.
    /// The encoder is not submitted - the caller is responsible for that.
    pub(crate) fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        selection: Option<&SelectionManager>,
    ) -> Result<()> {
        // Update camera uniform
        // TODO: only when needed
        let camera_uniform_slice = &[self.camera_resources.camera.to_uniform()];
        let camera_buffer_contents: &[u8] = bytemuck::cast_slice(camera_uniform_slice);
        self.queue
            .write_buffer(&self.camera_resources.buffer, 0, camera_buffer_contents);

        // Update lights uniform array
        // TODO: only when needed
        let lights_uniform = LightsArrayUniform::from_lights(&scene.lights);
        self.queue
            .write_buffer(&self.lights.buffer, 0, bytes_of(&lights_uniform));

        // Collect all instances into batches grouped by mesh and material
        let batches = scene.collect_draw_batches();

        // Partition batches by selection state if selection is provided
        let selected_batches = selection
            .filter(|sel| !sel.is_empty() && sel.config().outline_enabled)
            .map(|sel| {
                let (selected, _) =
                    partition_batches(&batches, |inst| sel.is_node_selected(inst.node_id));
                selected
            })
            .unwrap_or_default();
        let has_selection = !selected_batches.is_empty();

        // Update outline uniform if we have a selection
        if has_selection {
            if let Some(sel) = selection {
                let sel_config = sel.config();
                let outline_uniform = OutlineUniform {
                    color: sel_config.outline_color,
                    width_pixels: sel_config.outline_width,
                    screen_width: self.size.0 as f32,
                    screen_height: self.size.1 as f32,
                    _padding: 0.0,
                };
                self.queue.write_buffer(
                    &self.outline_resources.uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[outline_uniform]),
                );
            }
        }

        // =====================================================================
        // Pass 1: Main scene render (clears stencil to 0)
        // =====================================================================
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3D Scene Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.04,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.default_textures.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_bind_group(0, &self.camera_resources.bind_group, &[]);
            render_pass.set_bind_group(1, &self.lights.bind_group, &[]);

            // Build scene properties for shader generation and bind IBL if active
            let has_ibl = if let Some(env_id) = scene.active_environment_map {
                if let Some(processed) = self.ibl_resources.get_processed(env_id) {
                    render_pass.set_bind_group(3, &processed.bind_group, &[]);
                    true
                } else {
                    false
                }
            } else {
                false
            };

            let scene_props = SceneProperties { has_ibl };

            // Track current pipeline key to minimize pipeline changes
            let mut current_pipeline_key: Option<PipelineCacheKey> = None;

            // Render each batch
            for batch in &batches {
                let mesh = scene.meshes.get(&batch.mesh_id).unwrap();
                let material = scene.materials.get(&batch.material_id).unwrap();
                let material_props = material.get_properties(batch.primitive_type);

                // Bind material for this batch
                material.bind(&mut render_pass, batch.primitive_type);

                // Only change pipeline if material properties, scene properties, or primitive type changes
                let pipeline_key = PipelineCacheKey {
                    material_props: material_props.clone(),
                    scene_props: scene_props.clone(),
                    primitive_type: batch.primitive_type,
                };
                if current_pipeline_key.as_ref() != Some(&pipeline_key) {
                    let pipeline = self.get_or_create_pipeline(pipeline_key.clone());
                    render_pass.set_pipeline(pipeline);
                    current_pipeline_key = Some(pipeline_key);
                }

                // Draw all instances in this batch
                mesh.draw_instances(
                    &self.device,
                    &mut render_pass,
                    batch.primitive_type,
                    &batch.instances,
                );
            }
        }

        // =====================================================================
        // Pass 2 & 3: Screen-space selection outline (only if we have selected objects)
        // =====================================================================
        if has_selection {
            // Pass 2: Render selected objects to mask texture
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Selection Mask Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.outline_resources.mask_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.default_textures.depth.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                render_pass.set_pipeline(&self.outline_resources.mask_pipeline);
                render_pass.set_bind_group(0, &self.camera_resources.bind_group, &[]);

                for batch in &selected_batches {
                    if batch.primitive_type != PrimitiveType::TriangleList {
                        continue; // Only outline triangle meshes
                    }
                    let mesh = scene.meshes.get(&batch.mesh_id).unwrap();
                    mesh.draw_instances(
                        &self.device,
                        &mut render_pass,
                        batch.primitive_type,
                        &batch.instances,
                    );
                }
            }

            // Pass 3: Fullscreen post-process to draw outline
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Outline Screenspace Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None, // No depth test for fullscreen pass
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                render_pass.set_pipeline(&self.outline_resources.outline_pipeline);
                render_pass.set_bind_group(0, &self.outline_resources.bind_group, &[]);
                render_pass.draw(0..3, 0..1); // Fullscreen triangle
            }
        }

        Ok(())
    }

    pub fn render(
        &mut self,
        scene: &mut Scene,
        selection: Option<&SelectionManager>,
    ) -> Result<()> {
        // Prepare all GPU resources before rendering
        self.prepare_scene(scene)?;

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.render_scene_to_view(&view, &mut encoder, scene, selection)?;

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
