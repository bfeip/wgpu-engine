use crate::scene::{AlphaMode, MaterialProperties, PrimitiveType, SceneProperties};

use super::batching::{DrawBatch, DrawData};
use super::gpu_resources::{self, GpuTexture, PipelineCacheKey};
use super::pass_context::{FrameContext, SceneRenderPass};
use super::pipeline::PipelineCache;

/// Bind the scene-level bind groups shared by all geometry passes:
/// - Group 0: Camera (view/proj + eye position)
/// - Group 1: Lights
/// - Group 3: IBL environment (when active)
fn bind_scene_groups<'r>(render_pass: &mut wgpu::RenderPass<'r>, ctx: &'r FrameContext<'r>) {
    render_pass.set_bind_group(0, ctx.camera_bind_group, &[]);
    render_pass.set_bind_group(1, ctx.lights_bind_group, &[]);
    if let Some(ibl) = ctx.ibl_bind_group {
        render_pass.set_bind_group(3, ibl, &[]);
    }
}

/// Draw a list of batches through the scene geometry pipeline.
///
/// If `with_depth_prepass` is true, a depth-only pre-pass for `Blend`-mode materials
/// is executed first so their opaque portions establish correct depth occlusion before
/// the main draw loop renders them with blending.
fn draw_batches(
    render_pass: &mut wgpu::RenderPass<'_>,
    batches: &[DrawBatch],
    ctx: &FrameContext<'_>,
    pipeline_cache: &mut PipelineCache,
    with_depth_prepass: bool,
) {
    if with_depth_prepass {
        // Depth pre-pass for transparent objects: render depth-only with alpha test
        // so opaque portions of blend materials establish correct depth occlusion.
        let mut prepass_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in batches {
            let material = ctx.scene.get_material(batch.material_id).unwrap();
            let material_props = material.get_properties(batch.primitive_type);

            if material_props.alpha_mode != AlphaMode::Blend {
                continue;
            }

            let material_gpu = ctx
                .gpu_resources
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
                    always_on_top: material_props.always_on_top,
                },
                scene_props: SceneProperties { has_ibl: false },
                primitive_type: batch.primitive_type,
                depth_prepass: true,
            };
            if prepass_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = pipeline_cache.get_or_create(ctx.device, pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                prepass_pipeline_key = Some(pipeline_key);
            }

            let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
            let gpu_mesh = ctx
                .gpu_resources
                .get_mesh(batch.mesh_id)
                .expect("Mesh GPU resources not initialized");
            gpu_resources::draw_mesh_instances(
                ctx.device,
                render_pass,
                gpu_mesh,
                batch.primitive_type,
                &batch.instances,
                mesh.index_count(batch.primitive_type),
            );
        }
    }

    // Main draw loop
    let mut current_pipeline_key: Option<PipelineCacheKey> = None;
    for batch in batches {
        let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
        let material = ctx.scene.get_material(batch.material_id).unwrap();
        let material_props = material.get_properties(batch.primitive_type);

        let material_gpu = ctx
            .gpu_resources
            .get_material(batch.material_id, batch.primitive_type)
            .expect("Material GPU resources not initialized");
        render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

        // Only change pipeline if material properties, scene properties, or primitive type changes
        let pipeline_key = PipelineCacheKey {
            material_props: material_props.clone(),
            scene_props: ctx.scene_props.clone(),
            primitive_type: batch.primitive_type,
            depth_prepass: false,
        };
        if current_pipeline_key.as_ref() != Some(&pipeline_key) {
            let pipeline = pipeline_cache.get_or_create(ctx.device, pipeline_key.clone());
            render_pass.set_pipeline(pipeline);
            current_pipeline_key = Some(pipeline_key);
        }

        let gpu_mesh = ctx
            .gpu_resources
            .get_mesh(batch.mesh_id)
            .expect("Mesh GPU resources not initialized");
        gpu_resources::draw_mesh_instances(
            ctx.device,
            render_pass,
            gpu_mesh,
            batch.primitive_type,
            &batch.instances,
            mesh.index_count(batch.primitive_type),
        );
    }
}

/// Pass 1: Main scene render.
///
/// Clears color/depth/stencil, binds camera/lights/IBL, runs a depth pre-pass for
/// `Blend`-mode materials, then draws all batches with pipeline caching.
pub(crate) struct MainPass;

impl SceneRenderPass for MainPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);

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
                view: &ctx.renderer_textures.depth.view,
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

        bind_scene_groups(&mut render_pass, ctx);
        draw_batches(&mut render_pass, draw_data.all_batches(), ctx, pipeline_cache, true);
    }
}

/// Pass 2 (conditional): Overlay render.
///
/// Loads the existing color attachment, clears a separate depth buffer, and draws
/// always-on-top geometry so it depth-tests among itself but not against the scene.
/// Owns its own depth buffer so it can be independently resized.
pub(crate) struct OverlayPass {
    depth: super::gpu_resources::GpuTexture,
}

impl OverlayPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        Self {
            depth: super::gpu_resources::GpuTexture::depth_sized(
                device,
                width,
                height,
                sample_count,
                "overlay_depth_texture",
            ),
        }
    }
}

impl SceneRenderPass for OverlayPass {
    fn is_active(&self, draw_data: &DrawData) -> bool {
        draw_data.has_overlay()
    }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        self.depth = super::gpu_resources::GpuTexture::depth_sized(
            device,
            size.0,
            size.1,
            sample_count,
            "overlay_depth_texture",
        );
    }

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Overlay Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
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

        bind_scene_groups(&mut render_pass, ctx);
        draw_batches(&mut render_pass, draw_data.overlay_batches(), ctx, pipeline_cache, false);
    }
}

// ---------------------------------------------------------------------------
// HiddenLineSolidPass — flat-white solid for the hidden-line workflow
// ---------------------------------------------------------------------------

/// Hidden-line workflow solid pass.
///
/// Clears to white and renders all `TriangleList` geometry flat-white using
/// a minimal camera-only pipeline. Writes depth so the subsequent edge pass
/// can occlude lines that are behind solid faces.
pub(crate) struct HiddenLineSolidPass {
    pipeline: wgpu::RenderPipeline,
}

impl HiddenLineSolidPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
    ) -> Self {
        use super::gpu_resources::{instance_buffer_layout, vertex_buffer_layout};
        let shader = shader_generator
            .generate_hidden_line_solid_shader(device)
            .expect("Failed to generate hidden line solid shader");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hidden Line Solid Pipeline Layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Hidden Line Solid Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GpuTexture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                // Push faces slightly away so coplanar edges pass depth test.
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        Self { pipeline }
    }
}

impl SceneRenderPass for HiddenLineSolidPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        _pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Hidden Line Solid Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        for batch in draw_data.all_batches() {
            if batch.primitive_type != PrimitiveType::TriangleList { continue; }
            ctx.draw_batch(&mut render_pass, batch);
        }
    }
}

// ---------------------------------------------------------------------------
// HiddenLineEdgesPass — depth-tested edges for the hidden-line workflow
// ---------------------------------------------------------------------------

/// Hidden-line workflow edge pass.
///
/// Loads the existing color (written by [`HiddenLineSolidPass`]) and renders all
/// `LineList` primitives using the standard material pipeline. Lines whose depth
/// is greater than the solid geometry depth are occluded and not drawn.
pub(crate) struct HiddenLineEdgesPass;

impl SceneRenderPass for HiddenLineEdgesPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let has_lines = draw_data.all_batches().iter().any(|b| b.primitive_type == PrimitiveType::LineList);
        if !has_lines { return; }

        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Hidden Line Edges Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        bind_scene_groups(&mut render_pass, ctx);

        let mut current_pipeline_key: Option<PipelineCacheKey> = None;
        for batch in draw_data.all_batches().iter().filter(|b| b.primitive_type == PrimitiveType::LineList) {
            let material = ctx.scene.get_material(batch.material_id).unwrap();
            let material_props = material.get_properties(batch.primitive_type);
            let material_gpu = ctx
                .gpu_resources
                .get_material(batch.material_id, batch.primitive_type)
                .expect("Material GPU resources not initialized");
            render_pass.set_bind_group(2, &material_gpu.bind_group, &[]);

            let pipeline_key = PipelineCacheKey {
                material_props: material_props.clone(),
                scene_props: ctx.scene_props.clone(),
                primitive_type: batch.primitive_type,
                depth_prepass: false,
            };
            if current_pipeline_key.as_ref() != Some(&pipeline_key) {
                let pipeline = pipeline_cache.get_or_create(ctx.device, pipeline_key.clone());
                render_pass.set_pipeline(pipeline);
                current_pipeline_key = Some(pipeline_key);
            }
            ctx.draw_batch(&mut render_pass, batch);
        }
    }
}

// ---------------------------------------------------------------------------
// SelectionOutlinePass — mask + screenspace composite in one SceneRenderPass
// ---------------------------------------------------------------------------

/// Creates the pipeline that renders selected geometry into the R8Unorm mask texture.
fn build_mask_pipeline(
    device: &wgpu::Device,
    camera_bgl: &wgpu::BindGroupLayout,
    shader_generator: &mut crate::shaders::ShaderGenerator,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    use super::gpu_resources::{GpuTexture, instance_buffer_layout, vertex_buffer_layout};

    let shader = shader_generator
        .generate_outline_mask_shader(device)
        .expect("Failed to generate outline mask shader");
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Outline Mask Pipeline Layout"),
        bind_group_layouts: &[camera_bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Outline Mask Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_mask"),
            buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_mask"),
            targets: &[Some(wgpu::ColorTargetState {
                format: GpuTexture::MASK_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: GpuTexture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
        cache: None,
    })
}

struct ScreenspaceResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

/// Creates the fullscreen screenspace composite pipeline and its bind group.
fn build_screenspace_resources(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    shader_generator: &mut crate::shaders::ShaderGenerator,
    sample_count: u32,
    mask_texture: &super::gpu_resources::GpuTexture,
    uniform_buffer: &wgpu::Buffer,
) -> ScreenspaceResources {
    let multisampled = sample_count > 1;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Outline Screenspace Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled,
                },
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
    let bind_group = make_screenspace_bind_group(device, &bind_group_layout, &mask_texture.view, uniform_buffer);

    let shader = shader_generator
        .generate_outline_screenspace_shader(device, multisampled)
        .expect("Failed to generate outline screenspace shader");
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Outline Screenspace Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Outline Screenspace Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_fullscreen"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_outline"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: sample_count, mask: !0, alpha_to_coverage_enabled: false },
        multiview: None,
        cache: None,
    });

    ScreenspaceResources { pipeline, bind_group_layout, bind_group }
}

fn make_screenspace_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    mask_view: &wgpu::TextureView,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Outline Screenspace Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(mask_view) },
            wgpu::BindGroupEntry { binding: 2, resource: uniform_buffer.as_entire_binding() },
        ],
    })
}

/// Pass 3 (conditional): Selection outline.
///
/// Runs two wgpu render passes in sequence:
/// 1. Renders selected triangle geometry into an R8Unorm mask texture (depth-tested
///    against the main scene buffer so occluded geometry is not outlined).
/// 2. Reads the mask in a fullscreen screenspace pass and composites the outline
///    color over the scene.
///
/// Owns all GPU resources needed for this effect so neither `FrameContext` nor
/// `Renderer` need to know about outline-specific state.
pub(crate) struct SelectionOutlinePass {
    mask_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    mask_texture: super::gpu_resources::GpuTexture,
    screenspace: ScreenspaceResources,
}

impl SelectionOutlinePass {
    pub(crate) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
        sample_count: u32,
    ) -> Self {
        use wgpu::util::DeviceExt;
        use super::gpu_resources::{GpuTexture, OutlineUniform};

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Outline Uniform Buffer"),
            contents: bytemuck::cast_slice(&[OutlineUniform {
                color: [1.0, 0.6, 0.0, 1.0],
                width_pixels: 3.0,
                screen_width: config.width as f32,
                screen_height: config.height as f32,
                _padding: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mask_pipeline = build_mask_pipeline(device, camera_bgl, shader_generator, sample_count);

        let multisampled = sample_count > 1;
        let mask_label = if multisampled { "Outline Mask Color Attachment" } else { "Outline Mask Texture" };
        let mask_texture = GpuTexture::mask(device, config.width, config.height, sample_count, mask_label);

        let screenspace = build_screenspace_resources(
            device, config.format, shader_generator, sample_count, &mask_texture, &uniform_buffer,
        );

        Self { mask_pipeline, uniform_buffer, mask_texture, screenspace }
    }
}

impl SceneRenderPass for SelectionOutlinePass {
    fn is_active(&self, draw_data: &DrawData) -> bool {
        draw_data.has_selection()
    }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        let multisampled = sample_count > 1;
        let label = if multisampled { "Outline Mask Color Attachment" } else { "Outline Mask Texture" };
        self.mask_texture = super::gpu_resources::GpuTexture::mask(device, size.0, size.1, sample_count, label);
        self.screenspace.bind_group = make_screenspace_bind_group(
            device,
            &self.screenspace.bind_group_layout,
            &self.mask_texture.view,
            &self.uniform_buffer,
        );
    }

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        _pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        use super::gpu_resources::OutlineUniform;

        if let Some(cfg) = draw_data.outline_config() {
            ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[OutlineUniform {
                color: cfg.color,
                width_pixels: cfg.width_pixels,
                screen_width: ctx.size.0 as f32,
                screen_height: ctx.size.1 as f32,
                _padding: 0.0,
            }]));
        }

        // Mask pass — render selected triangles into the R8Unorm mask texture.
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Selection Mask Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mask_texture.view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &ctx.renderer_textures.depth.view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rp.set_pipeline(&self.mask_pipeline);
            rp.set_bind_group(0, ctx.camera_bind_group, &[]);
            for batch in draw_data.selected_batches() {
                if batch.primitive_type != PrimitiveType::TriangleList { continue; }
                let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
                let gpu_mesh = ctx.gpu_resources.get_mesh(batch.mesh_id).expect("Mesh GPU resources not initialized");
                gpu_resources::draw_mesh_instances(ctx.device, &mut rp, gpu_mesh, batch.primitive_type, &batch.instances, mesh.index_count(batch.primitive_type));
            }
            for batch in draw_data.selection_sub_geom_batches() {
                if batch.primitive_type != PrimitiveType::TriangleList { continue; }
                let gpu_mesh = ctx.gpu_resources.get_mesh(batch.mesh_id).expect("Mesh GPU resources not initialized");
                gpu_resources::draw_mesh_subgeom(ctx.device, &mut rp, gpu_mesh, batch.primitive_type, &batch.instance_transform, batch.first_index, batch.index_count);
            }
        }

        // Screenspace pass — composite outline color over scene.
        {
            let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Outline Screenspace Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rp.set_pipeline(&self.screenspace.pipeline);
            rp.set_bind_group(0, &self.screenspace.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}

