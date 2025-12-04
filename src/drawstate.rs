use std::collections::HashMap;

use wgpu::{util::DeviceExt};

use crate::{
    camera::{
        Camera,
        CameraUniform
    },
    scene::{
        InstanceRaw,
        Vertex
    },
    light::LightUniform,
    material::{
        MaterialManager,
        MaterialType
    },
    scene::Scene,
    shaders::ShaderGenerator,
    texture,
    common::PhysicalSize
};

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
    pub size: PhysicalSize<u32>,
    pub camera: Camera,
    /// Current cursor position in screen coordinates (x, y), or None if cursor is not over the window
    pub cursor_position: Option<(f32, f32)>,

    pub material_manager: MaterialManager,
    shader_generator: ShaderGenerator,

    color_material_pipeline_layout: wgpu::PipelineLayout,
    texture_material_pipeline_layout: wgpu::PipelineLayout,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    lights_buffer: wgpu::Buffer,
    lights_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,

    pipeline_cache: HashMap<MaterialType, wgpu::RenderPipeline>,
}

impl<'a> DrawState<'a> {
    // Creating some of the wgpu types requires async code
    // The target parameter can be a Window, Canvas, or any type implementing the necessary traits
    pub async fn new<T>(target: T, width: u32, height: u32) -> DrawState<'a>
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let size = PhysicalSize::new(width, height);

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
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let material_manager = MaterialManager::new(&device);
        let shader_generator = ShaderGenerator::new();

        let camera = Camera {
            eye: (0.0, 0.1, 0.2).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.001,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<LightUniform>() as wgpu::BufferAddress,
            mapped_at_creation: false
        });

        let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                }
            ],
            label: Some("Light bind group layout")
        });

        let lights_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lights_buffer.as_entire_binding()
                }
            ],
            label: Some("Light bind group")
        });

        let color_material_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Color Material Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &light_bind_group_layout,
                &material_manager.color_bind_group_layout
            ],
            push_constant_ranges: &[],
        });

        let texture_material_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Texture Material Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &light_bind_group_layout,
                &material_manager.texture_bind_group_layout
            ],
            push_constant_ranges: &[]
        });

        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        Self {
            surface,
            device,
            queue,
            config,
            size,
            camera,
            cursor_position: None,
            material_manager,
            shader_generator,
            color_material_pipeline_layout,
            texture_material_pipeline_layout,
            camera_buffer,
            camera_bind_group,
            lights_buffer,
            lights_bind_group,
            depth_texture,
            pipeline_cache: HashMap::new(),
        }
    }

    fn get_or_create_pipeline(&mut self, material_type: MaterialType) -> &wgpu::RenderPipeline {
        self.pipeline_cache.entry(material_type).or_insert_with(|| {
            let (pipeline_layout, shader, topology) = match material_type {
                MaterialType::FaceColor => {
                    let layout = &self.color_material_pipeline_layout;
                    let shader = self.shader_generator.generate_shader(&self.device, material_type)
                        .expect("Failed to generate shader for FaceColor material");
                    (layout, shader, wgpu::PrimitiveTopology::TriangleList)
                },
                MaterialType::FaceTexture => {
                    let layout = &self.texture_material_pipeline_layout;
                    let shader = self.shader_generator.generate_shader(&self.device, material_type)
                        .expect("Failed to generate shader for FaceTexture material");
                    (layout, shader, wgpu::PrimitiveTopology::TriangleList)
                },
                MaterialType::LineColor => {
                    let layout = &self.color_material_pipeline_layout;
                    let shader = self.shader_generator.generate_shader(&self.device, material_type)
                        .expect("Failed to generate shader for LineColor material");
                    (layout, shader, wgpu::PrimitiveTopology::LineList)
                },
                MaterialType::PointColor => {
                    let layout = &self.color_material_pipeline_layout;
                    let shader = self.shader_generator.generate_shader(&self.device, material_type)
                        .expect("Failed to generate shader for PointColor material");
                    (layout, shader, wgpu::PrimitiveTopology::PointList)
                },
            };

            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[
                        Vertex::desc(),
                        InstanceRaw::desc()
                    ],
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
                    format: texture::Texture::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default()
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

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.camera.aspect = new_size.width as f32 / new_size.height as f32;

            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    /// Render the scene to a specific view, updating uniforms and drawing all batches
    /// The encoder is not submitted - the caller is responsible for that
    pub(crate) fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
    ) -> anyhow::Result<()> {
        // Update camera uniform
        let camera_uniform_slice = &[self.camera.to_uniform()];
        let camera_buffer_contents: &[u8] = bytemuck::cast_slice(camera_uniform_slice);
        self.queue
            .write_buffer(&self.camera_buffer, 0, camera_buffer_contents);

        // Update light uniform
        let light_uniform_slice = &[scene.lights[0].to_uniform()];
        let light_buffer_contents: &[u8] = bytemuck::cast_slice(light_uniform_slice);
        self.queue
            .write_buffer(&self.lights_buffer, 0, light_buffer_contents);

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

            // Track current material type to minimize pipeline changes
            let mut current_material_type: Option<crate::material::MaterialType> = None;

            // Render each batch
            for batch in batches {
                let mesh = scene.meshes.get(&batch.mesh_id).unwrap();
                let batch_material_type = {
                    let material = self.material_manager.get(batch.material_id).unwrap();
                    // Bind material for this batch
                    material.bind(&mut render_pass)?;
                    material.material_type()
                };

                // Only change pipeline if material type changes
                if current_material_type != Some(batch_material_type) {
                    let pipeline = self.get_or_create_pipeline(batch_material_type);
                    render_pass.set_pipeline(pipeline);
                    current_material_type = Some(batch_material_type);
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

    pub fn render(&mut self, scene: &mut Scene) -> anyhow::Result<()> {
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