mod batching;
mod custom_pipeline;
mod gpu_resources;
mod outline;
mod pass_context;
mod pipeline;
mod prepare;
mod scene_pass;

pub use batching::{DrawBatch, DrawData};
pub use custom_pipeline::CustomPipelineBuilder;
pub use gpu_resources::{instance_buffer_layout, vertex_buffer_layout};
pub use pass_context::{FrameContext, SceneRenderPass};
pub use pipeline::PipelineCache;

use anyhow::Result;

use crate::{
    ibl::IblResources,
    scene::{Camera, Scene, SceneProperties},
    selection_query::SelectionQuery,
    shaders::ShaderGenerator,
};

use gpu_resources::{
    CameraResources, CameraUniform, GpuResourceManager, GpuTexture, HeadlessResources,
    LightResources, MaterialBindGroupLayouts, MaterialPipelineLayouts, RendererTextures,
};
use outline::{OutlineResources, OutlineUniform};
use scene_pass::{MainPass, OutlinePass, OverlayPass, SelectionMaskPass};

pub struct Renderer {
    // Core GPU resources
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
    renderer_textures: RendererTextures,

    // Other
    pipeline_cache: PipelineCache,
    ibl_resources: IblResources,
    outline_resources: OutlineResources,

    // Scene GPU resource tracking
    gpu_resources: GpuResourceManager,

    /// MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
    sample_count: u32,

    /// Cached GPU resources for headless rendering, reused across frames at the same size.
    headless_resources: Option<HeadlessResources>,

    /// Ordered list of render passes executed each frame.
    passes: Vec<Box<dyn SceneRenderPass>>,
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

        let renderer_textures = RendererTextures::new(&device, &queue, &config, sample_count);
        let mut shader_generator = ShaderGenerator::new();
        let outline_resources = OutlineResources::new(
            &device,
            &config,
            &camera_resources.bind_group_layout,
            &mut shader_generator,
            sample_count,
        );

        let pipeline_cache = PipelineCache::new(pipelines, shader_generator, sample_count, surface_format);

        Self {
            device,
            queue,
            size: (width, height),
            surface_format,
            config,
            camera_resources,
            lights,
            material_layouts,
            renderer_textures,
            pipeline_cache,
            ibl_resources,
            outline_resources,
            gpu_resources: GpuResourceManager::new(),
            sample_count,
            headless_resources: None,
            passes: vec![
                Box::new(MainPass),
                Box::new(OverlayPass),
                Box::new(SelectionMaskPass),
                Box::new(OutlinePass),
            ],
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

    /// Get the MSAA sample count (1 = no MSAA, 4 = 4× MSAA).
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Compile a user-supplied WESL shader with access to all engine shader modules.
    ///
    /// Engine modules available for import: `package::common`, `package::camera`,
    /// `package::lighting`, `package::constants`, `package::vertex`, `package::pbr`.
    /// See `docs/custom-passes.md` for the full module reference.
    pub fn compile_user_wesl(&self, source: &str) -> anyhow::Result<wgpu::ShaderModule> {
        crate::shaders::compile_user_wesl(&self.device, source)
    }

    /// Create a pipeline builder pre-configured with the engine's standard vertex
    /// and instance buffer layouts, surface format, and MSAA sample count.
    ///
    /// Camera (group 0) and lights (group 1) bind group layouts are included by
    /// default. See [`CustomPipelineBuilder`] for the full configuration API.
    pub fn custom_pipeline_builder(&self) -> CustomPipelineBuilder<'_> {
        CustomPipelineBuilder::new(
            &self.device,
            self.surface_format,
            self.sample_count,
            &self.camera_resources.bind_group_layout,
            &self.lights.bind_group_layout,
        )
    }

    /// Get the bind group layout for the camera uniform (group 0).
    ///
    /// Prefer [`custom_pipeline_builder`](Self::custom_pipeline_builder) for
    /// building custom pipelines — this method is a lower-level escape hatch.
    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.camera_resources.bind_group_layout
    }

    /// Get the bind group layout for the lights uniform (group 1).
    ///
    /// Prefer [`custom_pipeline_builder`](Self::custom_pipeline_builder) for
    /// building custom pipelines — this method is a lower-level escape hatch.
    pub fn lights_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.lights.bind_group_layout
    }

    /// Replace the ordered list of render passes executed each frame.
    ///
    /// Passes are executed in order; each pass receives the same [`FrameContext`]
    /// snapshot and can read or write the color/depth attachment as needed.
    /// Use this to inject custom passes (e.g. non-photorealistic shading) or
    /// to trim the default pass list.
    pub fn set_passes(&mut self, passes: Vec<Box<dyn SceneRenderPass>>) {
        self.passes = passes;
    }

    /// Preprocess an environment map into CPU-side IBL data via GPU compute shaders.
    ///
    /// Returns a [`PreprocessedIbl`] that can be attached to the environment map
    /// and serialized into a scene file. Requires compute shader support.
    pub fn preprocess_ibl(
        &self,
        env_map: &crate::scene::EnvironmentMap,
    ) -> anyhow::Result<crate::scene::PreprocessedIbl> {
        self.ibl_resources
            .preprocess_ibl(&self.device, &self.queue, env_map)
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

            self.renderer_textures.depth =
                GpuTexture::depth(&self.device, &self.config, self.sample_count, "depth_texture");
            self.renderer_textures.overlay_depth =
                GpuTexture::depth(&self.device, &self.config, self.sample_count, "overlay_depth_texture");

            self.renderer_textures.msaa_color_attachment = if self.sample_count > 1 {
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

        // Resolve IBL bind group once — shared by all geometry passes this frame.
        let ibl_bind_group = scene
            .active_environment_map()
            .and_then(|env_id| self.ibl_resources.get_processed(env_id))
            .map(|processed| &processed.bind_group);

        // Build a frame context from named field borrows. Because `pipeline_cache` is
        // not included here, the borrow checker allows `&mut self.pipeline_cache` to
        // coexist with `&ctx` in the pass calls below.
        let ctx = FrameContext {
            device: &self.device,
            queue: &self.queue,
            scene,
            gpu_resources: &self.gpu_resources,
            camera_bind_group: &self.camera_resources.bind_group,
            lights_bind_group: &self.lights.bind_group,
            ibl_bind_group,
            scene_props: SceneProperties { has_ibl: ibl_bind_group.is_some() },
            renderer_textures: &self.renderer_textures,
            outline_resources: &self.outline_resources,
            sample_count: self.sample_count,
            surface_format: self.surface_format,
            size: self.size,
        };

        let passes = &mut self.passes;
        let pipeline_cache = &mut self.pipeline_cache;
        for pass in passes.iter_mut() {
            if pass.is_active(&draw_data) {
                pass.execute(encoder, view, &ctx, pipeline_cache, &draw_data);
            }
        }

        Ok(())
    }

}
