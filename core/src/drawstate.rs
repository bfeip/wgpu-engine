use std::collections::HashMap;

use anyhow::Result;
use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::{
    camera::{Camera, CameraUniform},
    scene::{
        GpuTexture, InstanceRaw, LightUniform, MaterialGpuResources, MaterialProperties,
        PrimitiveType, Scene, Texture, Vertex,
    },
    shaders::ShaderGenerator,
};

/// Maximum texture dimension for WebGL. When the canvas exceeds this size,
/// we scale down the surface while preserving aspect ratio.
#[cfg(target_arch = "wasm32")]
const MAX_TEXTURE_DIMENSION: u32 = 2048;

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

pub(crate) struct DrawState<'a> {
    pub surface: wgpu::Surface<'a>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: (u32, u32),
    pub camera: Camera,
    /// Current cursor position in screen coordinates (x, y), or None if cursor is not over the window
    pub cursor_position: Option<(f32, f32)>,

    // Bind group layouts for materials
    pub(crate) color_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) texture_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) pbr_bind_group_layout: wgpu::BindGroupLayout,

    shader_generator: ShaderGenerator,

    color_material_pipeline_layout: wgpu::PipelineLayout,
    texture_material_pipeline_layout: wgpu::PipelineLayout,
    pbr_material_pipeline_layout: wgpu::PipelineLayout,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    lights_buffer: wgpu::Buffer,
    lights_bind_group: wgpu::BindGroup,
    depth_texture: GpuTexture,

    // Default textures for PBR materials (used when specific textures are not provided)
    /// 1x1 white texture for missing base color textures
    pub(crate) default_white_texture: GpuTexture,
    /// 1x1 neutral normal texture (0.5, 0.5, 1.0) for missing normal maps
    pub(crate) default_normal_texture: GpuTexture,

    pipeline_cache: HashMap<(MaterialProperties, PrimitiveType), wgpu::RenderPipeline>,
}

impl<'a> DrawState<'a> {
    // Creating some of the wgpu types requires async code
    // The target parameter can be a Window, Canvas, or any type implementing the necessary traits
    pub async fn new<T>(target: T, width: u32, height: u32) -> DrawState<'a>
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

        // Create bind group layouts for materials
        let color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // PBR bind group layout: uniform + 3 textures with samplers
        let pbr_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let shader_generator = ShaderGenerator::new();

        let camera = Camera {
            eye: (0.0, 0.1, 0.2).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.001,
            zfar: 100.0,
            ortho: false,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<LightUniform>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let lights_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buffer.as_entire_binding(),
            }],
            label: Some("Light bind group"),
        });

        let color_material_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Color Material Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &color_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let texture_material_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Texture Material Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let pbr_material_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PBR Material Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                    &pbr_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        // Create default textures for PBR materials
        let default_white_texture = Texture::create_solid_color_texture(
            &device,
            &queue,
            [255, 255, 255, 255],
            "default_white_texture",
        );
        let default_normal_texture = Texture::create_solid_color_texture(
            &device,
            &queue,
            [128, 128, 255, 255], // Neutral normal (0.5, 0.5, 1.0) in tangent space
            "default_normal_texture",
        );

        Self {
            surface,
            device,
            queue,
            config,
            size,
            camera,
            cursor_position: None,
            color_bind_group_layout,
            texture_bind_group_layout,
            pbr_bind_group_layout,
            shader_generator,
            color_material_pipeline_layout,
            texture_material_pipeline_layout,
            pbr_material_pipeline_layout,
            camera_buffer,
            camera_bind_group,
            lights_buffer,
            lights_bind_group,
            depth_texture,
            default_white_texture,
            default_normal_texture,
            pipeline_cache: HashMap::new(),
        }
    }

    /// Prepare all GPU resources for a scene before rendering.
    ///
    /// This method ensures all textures, materials, and meshes have their GPU resources
    /// created or updated as needed. It should be called before `render_scene_to_view()`.
    pub fn prepare_scene(&mut self, scene: &mut Scene) -> Result<()> {
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

        Ok(())
    }

    /// Prepare GPU resources for a specific material primitive type.
    fn prepare_material_primitive(
        &self,
        scene: &mut Scene,
        material_id: u32,
        primitive_type: PrimitiveType,
    ) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        // Check if this material needs PBR path
        let use_pbr = material.normal_texture.is_some()
            || material.metallic_roughness_texture.is_some();

        match primitive_type {
            PrimitiveType::TriangleList => {
                if use_pbr {
                    // PBR material path - create bind group with all PBR textures
                    let pbr_uniform = material.build_pbr_uniform();
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("PBR Uniform Buffer"),
                        contents: bytes_of(&pbr_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    // Get base color texture or use default white
                    let (base_color_view, base_color_sampler) =
                        if let Some(tex_id) = material.base_color_texture {
                            let tex = scene.textures.get(&tex_id).ok_or_else(|| {
                                anyhow::anyhow!("Base color texture {} not found", tex_id)
                            })?;
                            let gpu = tex.gpu();
                            (&gpu.view, &gpu.sampler)
                        } else {
                            (&self.default_white_texture.view, &self.default_white_texture.sampler)
                        };

                    // Get normal texture or use default neutral normal
                    let (normal_view, normal_sampler) =
                        if let Some(tex_id) = material.normal_texture {
                            let tex = scene.textures.get(&tex_id).ok_or_else(|| {
                                anyhow::anyhow!("Normal texture {} not found", tex_id)
                            })?;
                            let gpu = tex.gpu();
                            (&gpu.view, &gpu.sampler)
                        } else {
                            (&self.default_normal_texture.view, &self.default_normal_texture.sampler)
                        };

                    // Get metallic-roughness texture or use default white
                    let (mr_view, mr_sampler) =
                        if let Some(tex_id) = material.metallic_roughness_texture {
                            let tex = scene.textures.get(&tex_id).ok_or_else(|| {
                                anyhow::anyhow!("Metallic-roughness texture {} not found", tex_id)
                            })?;
                            let gpu = tex.gpu();
                            (&gpu.view, &gpu.sampler)
                        } else {
                            (&self.default_white_texture.view, &self.default_white_texture.sampler)
                        };

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("PBR Material Bind Group"),
                        layout: &self.pbr_bind_group_layout,
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
                        primitive_type,
                        MaterialGpuResources {
                            bind_group,
                            _buffer: Some(buffer),
                        },
                    );
                    material.mark_clean(primitive_type);
                } else if let Some(texture_id) = material.base_color_texture {
                    // Legacy texture-based face rendering (base color texture only)
                    let texture = scene.textures.get(&texture_id).ok_or_else(|| {
                        anyhow::anyhow!("Texture {} not found for material {}", texture_id, material_id)
                    })?;
                    let gpu_tex = texture.gpu();

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Material Base Color Texture Bind Group"),
                        layout: &self.texture_bind_group_layout,
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
                        primitive_type,
                        MaterialGpuResources {
                            bind_group,
                            _buffer: None,
                        },
                    );
                    material.mark_clean(primitive_type);
                } else {
                    // Legacy color-based face rendering (using base_color_factor)
                    let color = material.base_color_factor;
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material Base Color Factor Buffer"),
                        contents: bytes_of(&color),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Material Base Color Factor Bind Group"),
                        layout: &self.color_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        }],
                    });

                    let material = scene.materials.get_mut(&material_id).unwrap();
                    material.set_gpu(
                        primitive_type,
                        MaterialGpuResources {
                            bind_group,
                            _buffer: Some(buffer),
                        },
                    );
                    material.mark_clean(primitive_type);
                }
            }
            PrimitiveType::LineList => {
                if let Some(color) = material.line_color {
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material Line Color Buffer"),
                        contents: bytes_of(&color),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Material Line Color Bind Group"),
                        layout: &self.color_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        }],
                    });

                    let material = scene.materials.get_mut(&material_id).unwrap();
                    material.set_gpu(
                        primitive_type,
                        MaterialGpuResources {
                            bind_group,
                            _buffer: Some(buffer),
                        },
                    );
                    material.mark_clean(primitive_type);
                }
            }
            PrimitiveType::PointList => {
                if let Some(color) = material.point_color {
                    let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Material Point Color Buffer"),
                        contents: bytes_of(&color),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Material Point Color Bind Group"),
                        layout: &self.color_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        }],
                    });

                    let material = scene.materials.get_mut(&material_id).unwrap();
                    material.set_gpu(
                        primitive_type,
                        MaterialGpuResources {
                            bind_group,
                            _buffer: Some(buffer),
                        },
                    );
                    material.mark_clean(primitive_type);
                }
            }
        }

        Ok(())
    }

    fn get_or_create_pipeline(
        &mut self,
        properties: MaterialProperties,
        primitive_type: PrimitiveType,
    ) -> &wgpu::RenderPipeline {
        let cache_key = (properties.clone(), primitive_type);
        self.pipeline_cache.entry(cache_key).or_insert_with(|| {
            // Select pipeline layout based on material type
            let use_pbr = properties.has_normal_map || properties.has_metallic_roughness_texture;
            let pipeline_layout = if use_pbr {
                &self.pbr_material_pipeline_layout
            } else if properties.has_base_color_texture {
                &self.texture_material_pipeline_layout
            } else {
                &self.color_material_pipeline_layout
            };

            let shader = self
                .shader_generator
                .generate_shader(&self.device, &properties)
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

            self.camera.aspect = width as f32 / height as f32;

            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    /// Render the scene to a specific view, updating uniforms and drawing all batches
    /// The encoder is not submitted - the caller is responsible for that
    pub(crate) fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
    ) -> Result<()> {
        // Update camera uniform
        // TODO: only when needed
        let camera_uniform_slice = &[self.camera.to_uniform()];
        let camera_buffer_contents: &[u8] = bytemuck::cast_slice(camera_uniform_slice);
        self.queue
            .write_buffer(&self.camera_buffer, 0, camera_buffer_contents);

        // Update light uniform
        // TODO: only when needed
        if !scene.lights.is_empty() {
            let light_uniform_slice = &[scene.lights[0].to_uniform()];
            let light_buffer_contents: &[u8] = bytemuck::cast_slice(light_uniform_slice);
            self.queue
                .write_buffer(&self.lights_buffer, 0, light_buffer_contents);
        }

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
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.lights_bind_group, &[]);

            // Collect all instances into batches grouped by mesh and material
            let batches = scene.collect_draw_batches();

            // Track current pipeline key to minimize pipeline changes
            let mut current_pipeline_key: Option<(MaterialProperties, PrimitiveType)> = None;

            // Render each batch
            for batch in batches {
                let mesh = scene.meshes.get(&batch.mesh_id).unwrap();
                let material = scene.materials.get(&batch.material_id).unwrap();
                let properties = material.get_properties(batch.primitive_type);

                // Bind material for this batch
                material.bind(&mut render_pass, batch.primitive_type);

                // Only change pipeline if material properties or primitive type changes
                let pipeline_key = (properties.clone(), batch.primitive_type);
                if current_pipeline_key.as_ref() != Some(&pipeline_key) {
                    let pipeline = self.get_or_create_pipeline(properties, batch.primitive_type);
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

        Ok(())
    }

    pub fn render(&mut self, scene: &mut Scene) -> Result<()> {
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

        self.render_scene_to_view(&view, &mut encoder, scene)?;

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
