mod gpu_resources;
mod outline;
mod pipeline;
mod prepare;
mod types;

use std::collections::HashMap;

use anyhow::Result;
use bytemuck::bytes_of;

use crate::{
    camera::Camera,
    ibl::IblResources,
    scene::{
        partition_batches, PrimitiveType, Scene, SceneProperties,
    },
    selection::SelectionManager,
    shaders::ShaderGenerator,
};

use gpu_resources::{
    create_depth_texture, create_mask_texture, CameraUniform, GpuResourceManager,
    LightsArrayUniform,
};
use outline::{OutlineResources, OutlineUniform};
use types::{
    clamp_surface_size, CameraResources, DefaultTextures, LightResources,
    MaterialBindGroupLayouts, MaterialPipelineLayouts, PipelineCacheKey,
};

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

    // Scene GPU resource tracking (for future use)
    gpu_resources: GpuResourceManager,
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
        let material_layouts = MaterialBindGroupLayouts::new(&device);
        let camera_resources =
            CameraResources::new(&device, config.width as f32 / config.height as f32);
        let lights = LightResources::new(&device);
        let ibl_resources = IblResources::new(&device, &queue);
        let pipelines = MaterialPipelineLayouts::new(
            &device,
            &camera_resources.bind_group_layout,
            &lights.bind_group_layout,
            &material_layouts,
            &ibl_resources.bind_group_layout,
        );
        let default_textures = DefaultTextures::new(&device, &queue, &config);
        let mut shader_generator = ShaderGenerator::new();
        let outline_resources = OutlineResources::new(
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
            gpu_resources: GpuResourceManager::new(),
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
                create_depth_texture(&self.device, &self.config, "depth_texture");

            // Recreate mask texture at new size
            self.outline_resources.mask_texture = create_mask_texture(
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
        let camera_uniform_slice = &[CameraUniform::from_camera(&self.camera_resources.camera)];
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
                let material_gpu = self.gpu_resources
                    .get_material(batch.material_id, batch.primitive_type)
                    .expect("Material GPU resources not initialized");
                render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

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
                let gpu_mesh = self.gpu_resources.get_mesh(batch.mesh_id)
                    .expect("Mesh GPU resources not initialized");
                gpu_resources::draw_mesh_instances(
                    &self.device,
                    &mut render_pass,
                    gpu_mesh,
                    batch.primitive_type,
                    &batch.instances,
                    mesh.index_count(batch.primitive_type),
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
                    let gpu_mesh = self.gpu_resources.get_mesh(batch.mesh_id)
                        .expect("Mesh GPU resources not initialized");
                    gpu_resources::draw_mesh_instances(
                        &self.device,
                        &mut render_pass,
                        gpu_mesh,
                        batch.primitive_type,
                        &batch.instances,
                        mesh.index_count(batch.primitive_type),
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
