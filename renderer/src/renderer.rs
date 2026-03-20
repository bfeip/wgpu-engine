mod batching;
mod gpu_resources;
mod outline;
mod pipeline;
mod prepare;

use std::collections::HashMap;

use anyhow::Result;

use crate::{
    ibl::IblResources,
    scene::{AlphaMode, Camera, MaterialProperties, PrimitiveType, Scene, SceneProperties},
    selection_query::SelectionQuery,
    shaders::ShaderGenerator,
};

use batching::{DrawBatch, DrawData};

use gpu_resources::{
    CameraResources, CameraUniform, DefaultTextures, GpuResourceManager,
    GpuTexture, HeadlessResources, LightResources, MaterialBindGroupLayouts,
    MaterialPipelineLayouts, PipelineCacheKey,
};
use outline::{OutlineResources, OutlineUniform};

pub struct Renderer {
    // Core GPU resources — public for Viewer access
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: (u32, u32),
    surface_format: wgpu::TextureFormat,

    // Internal config used by texture helpers (width/height/format)
    config: wgpu::SurfaceConfiguration,

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

    // Scene GPU resource tracking
    gpu_resources: GpuResourceManager,

    /// MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
    sample_count: u32,

    /// Cached GPU resources for headless rendering, reused across frames at the same size.
    headless_resources: Option<HeadlessResources>,
}

impl Renderer {
    /// Create a new renderer from pre-created device and queue.
    ///
    /// The caller (typically Viewer) is responsible for creating the wgpu instance,
    /// adapter, device, queue, and surface. The renderer only needs the device, queue,
    /// surface format, dimensions, and MSAA sample count.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        // Build an internal config for texture helpers
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Create grouped resources
        let material_layouts = MaterialBindGroupLayouts::new(&device);
        let camera_resources = CameraResources::new(&device);
        let lights = LightResources::new(&device);
        let ibl_resources = IblResources::new(&device, &queue, has_compute);
        let pipelines = MaterialPipelineLayouts::new(
            &device,
            &camera_resources.bind_group_layout,
            &lights.bind_group_layout,
            &material_layouts,
            &ibl_resources.bind_group_layout,
        );

        let default_textures = DefaultTextures::new(&device, &queue, &config, sample_count);
        let mut shader_generator = ShaderGenerator::new();
        let outline_resources = OutlineResources::new(
            &device,
            &config,
            &camera_resources.bind_group_layout,
            &mut shader_generator,
            sample_count,
        );

        Self {
            device,
            queue,
            size: (width, height),
            surface_format,
            config,
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
            sample_count,
            headless_resources: None,
        }
    }

    /// Create a new renderer for headless/offscreen rendering.
    ///
    /// This creates its own wgpu instance, adapter, device, and queue without
    /// requiring a surface. Useful for generating still images, thumbnails,
    /// or server-side rendering.
    ///
    /// Uses `Rgba8UnormSrgb` format and disables MSAA for simplicity.
    pub async fn new_headless(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter for headless rendering");

        let has_compute = adapter.get_info().backend != wgpu::Backend::Gl;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("Headless Renderer"),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .expect("Failed to create GPU device for headless rendering");

        let surface_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let sample_count = 1; // No MSAA for headless

        Self::new(device, queue, surface_format, width, height, sample_count, has_compute)
    }

    /// Get a reference to the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a reference to the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the current viewport size as (width, height).
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Get the surface texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Clear all scene-specific GPU resources.
    ///
    /// Call this when the scene is cleared or replaced to ensure stale GPU
    /// buffers (vertex data, textures, material bind groups) are not reused.
    pub fn clear_gpu_resources(&mut self) {
        self.gpu_resources.clear_scene_resources();
        self.lights.synced_generation = 0;
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        let (width, height) = new_size;
        if width > 0 && height > 0 {
            self.size = (width, height);
            self.config.width = width;
            self.config.height = height;

            self.default_textures.depth =
                GpuTexture::depth(&self.device, &self.config, self.sample_count, "depth_texture");

            self.default_textures.msaa_color_attachment = if self.sample_count > 1 {
                Some(GpuTexture::color_attachment(&self.device, &self.config, self.sample_count, "msaa_color_attachment"))
            } else {
                None
            };

            self.outline_resources
                .resize(&self.device, width, height, self.sample_count);

            self.headless_resources = None;
        }
    }

    /// Render the scene to an image buffer.
    ///
    /// This is the primary API for headless rendering. It renders the scene
    /// from the given camera viewpoint and returns the result as an RGBA image.
    ///
    /// The renderer must have been created with dimensions matching the desired
    /// output size, or resized beforehand.
    pub fn render_scene_to_image(
        &mut self,
        camera: &Camera,
        scene: &mut Scene,
        selection: Option<&dyn SelectionQuery>,
    ) -> Result<image::RgbaImage> {
        let (width, height) = self.size;

        // Ensure cached headless resources exist and match the current size
        if self
            .headless_resources
            .as_ref()
            .is_none_or(|r| r.size != (width, height))
        {
            self.headless_resources = Some(HeadlessResources::new(
                &self.device,
                width,
                height,
                self.surface_format,
            ));
        }

        let padded_bytes_per_row = self.headless_resources.as_ref().unwrap().padded_bytes_per_row;

        // Prepare and render — temporarily take resources out of self to avoid
        // borrow conflict with render_scene_to_view(&mut self)
        self.prepare_scene(camera, scene)?;
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Headless Render Encoder"),
            },
        );
        let resources = self.headless_resources.take().unwrap();
        self.render_scene_to_view(&resources.view, &mut encoder, camera, scene, selection)?;

        // Copy texture to staging buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &resources.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read the data
        let buffer_slice = resources.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        receiver.recv().unwrap()?;

        // Copy data, removing row padding
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let data = buffer_slice.get_mapped_range();
        let mut image_data = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (unpadded_bytes_per_row) as usize;
            image_data.extend_from_slice(&data[start..end]);
        }
        drop(data);
        resources.staging_buffer.unmap();

        // Put resources back for reuse
        self.headless_resources = Some(resources);

        image::RgbaImage::from_raw(width, height, image_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from rendered data"))
    }

    /// Render the scene to a specific view, updating uniforms and drawing all batches.
    /// If a selection is provided and not empty, selection outlines will be rendered.
    /// The encoder is not submitted - the caller is responsible for that.
    pub fn render_scene_to_view(
        &mut self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        scene: &Scene,
        selection: Option<&dyn SelectionQuery>,
    ) -> Result<()> {
        // Update camera uniform
        // TODO: only when needed
        let camera_uniform_slice = &[CameraUniform::from_camera(camera)];
        let camera_buffer_contents: &[u8] = bytemuck::cast_slice(camera_uniform_slice);
        self.queue
            .write_buffer(&self.camera_resources.buffer, 0, camera_buffer_contents);

        // Collect, sort, and partition draw batches for this frame
        let draw_data = DrawData::new(scene, camera.eye, selection);

        // Update outline uniform if we have a selection
        if draw_data.has_selection() {
            if let Some(sel) = selection {
                let outline_cfg = sel.outline_config();
                let outline_uniform = OutlineUniform {
                    color: outline_cfg.color,
                    width_pixels: outline_cfg.width_pixels,
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

        self.render_main_pass(encoder, view, draw_data.all_batches(), scene);

        if draw_data.has_selection() {
            self.render_selection_mask_pass(encoder, draw_data.selected_batches(), scene);
            self.render_outline_pass(encoder, view);
        }

        Ok(())
    }

    /// Pass 1: Main scene render - clears color/depth/stencil, binds camera/lights/IBL,
    /// and draws all batches with pipeline caching.
    fn render_main_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        batches: &[DrawBatch],
        scene: &Scene,
    ) {
        // When MSAA is active, render to the multisampled color attachment
        // and resolve to the swapchain view. Otherwise render directly to the swapchain.
        let (color_view, resolve_target) = match &self.default_textures.msaa_color_attachment {
            Some(msaa) => (&msaa.view, Some(view as &wgpu::TextureView)),
            None => (view, None),
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("3D Scene Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
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
        let has_ibl = if let Some(env_id) = scene.active_environment_map() {
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

        // Depth pre-pass for transparent objects: render depth-only with alpha test
        // so opaque portions of blend materials establish correct depth occlusion.
        let mut prepass_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in batches {
            let material = scene.get_material(batch.material_id).unwrap();
            let material_props = material.get_properties(batch.primitive_type);

            if material_props.alpha_mode != AlphaMode::Blend {
                continue;
            }

            let material_gpu = self.gpu_resources
                .get_material(batch.material_id, batch.primitive_type)
                .expect("Material GPU resources not initialized");
            render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

            // Normalize key: only has_lighting (pipeline layout), double_sided
            // (cull mode), and primitive_type matter for depth-only output.
            // alpha_mode is always Mask, IBL is unused.
            let pipeline_key = PipelineCacheKey {
                material_props: MaterialProperties {
                    alpha_mode: AlphaMode::Mask,
                    has_lighting: material_props.has_lighting,
                    double_sided: material_props.double_sided,
                },
                scene_props: SceneProperties { has_ibl: false },
                primitive_type: batch.primitive_type,
                depth_prepass: true,
            };
            if prepass_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = self.get_or_create_pipeline(pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                prepass_pipeline_key = Some(pipeline_key);
            }

            let mesh = scene.get_mesh(batch.mesh_id).unwrap();
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

        // Main rendering pass
        let mut current_pipeline_key: Option<PipelineCacheKey> = None;

        for batch in batches {
            let mesh = scene.get_mesh(batch.mesh_id).unwrap();
            let material = scene.get_material(batch.material_id).unwrap();
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
                depth_prepass: false,
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

    /// Pass 2: Render selected objects to mask texture for outline detection.
    fn render_selection_mask_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        selected_batches: &[DrawBatch],
        scene: &Scene,
    ) {
        let (mask_view, mask_resolve_target) = match &self.outline_resources.msaa_color_attachment {
            Some(msaa_mask) => (&msaa_mask.view, Some(&self.outline_resources.mask.view as &wgpu::TextureView)),
            None => (&self.outline_resources.mask.view, None),
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Selection Mask Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: mask_view,
                resolve_target: mask_resolve_target,
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

        for batch in selected_batches {
            if batch.primitive_type != PrimitiveType::TriangleList {
                continue; // Only outline triangle meshes
            }
            let mesh = scene.get_mesh(batch.mesh_id).unwrap();
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

    /// Pass 3: Fullscreen post-process to draw the selection outline.
    fn render_outline_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
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
