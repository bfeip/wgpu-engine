use crate::scene::PrimitiveType;

use super::super::batching::DrawData;
use super::super::gpu_resources::{self, GpuTexture, OutlineUniform, instance_buffer_layout, vertex_buffer_layout};
use super::super::pass_context::{FrameContext, SceneRenderPass};
use super::super::pipeline::MaterialPipelineCache;

/// Creates the pipeline that renders highlighted geometry into the R8Unorm mask texture.
fn build_mask_pipeline(
    device: &wgpu::Device,
    camera_bgl: &wgpu::BindGroupLayout,
    shader_generator: &mut crate::shaders::ShaderGenerator,
    sample_count: u32,
) -> wgpu::RenderPipeline {
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
    mask_texture: &GpuTexture,
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

/// Runs two wgpu render passes in sequence:
/// 1. Renders highlighted triangle geometry into an R8Unorm mask texture (depth-tested
///    against the main scene buffer so occluded geometry is not outlined).
/// 2. Reads the mask in a fullscreen screenspace pass and composites the outline
///    color over the scene.
///
/// Owns all GPU resources needed for this effect so neither `FrameContext` nor
/// `Renderer` need to know about outline-specific state.
pub(crate) struct OutlinePass {
    mask_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    mask_texture: GpuTexture,
    screenspace: ScreenspaceResources,
}

impl OutlinePass {
    pub(crate) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
        sample_count: u32,
    ) -> Self {
        use wgpu::util::DeviceExt;

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

impl SceneRenderPass for OutlinePass {
    fn is_active(&self, draw_data: &DrawData) -> bool {
        draw_data.has_highlights()
    }

    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32), sample_count: u32) {
        let multisampled = sample_count > 1;
        let label = if multisampled { "Outline Mask Color Attachment" } else { "Outline Mask Texture" };
        self.mask_texture = GpuTexture::mask(device, size.0, size.1, sample_count, label);
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
        _pipeline_cache: &mut MaterialPipelineCache,
        draw_data: &DrawData,
    ) {
        if let Some(cfg) = draw_data.outline_config() {
            ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[OutlineUniform {
                color: cfg.color,
                width_pixels: cfg.width_pixels,
                screen_width: ctx.size.0 as f32,
                screen_height: ctx.size.1 as f32,
                _padding: 0.0,
            }]));
        }

        // Mask pass — render highlighted triangles into the R8Unorm mask texture.
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Highlight Mask Pass"),
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
            for batch in draw_data.highlighted_batches() {
                if batch.primitive_type != PrimitiveType::TriangleList { continue; }
                let mesh = ctx.scene.get_mesh(batch.mesh_id).unwrap();
                let gpu_mesh = ctx.gpu_resources.get_mesh(batch.mesh_id).expect("Mesh GPU resources not initialized");
                gpu_resources::draw_mesh_instances(ctx.device, &mut rp, gpu_mesh, batch.primitive_type, &batch.instances, mesh.index_count(batch.primitive_type));
            }
            for batch in draw_data.highlight_sub_geom_batches() {
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
