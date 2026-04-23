use wgpu::util::DeviceExt;

use crate::shaders::ShaderGenerator;

use super::gpu_resources::{instance_buffer_layout, vertex_buffer_layout, GpuTexture};

/// GPU uniform data for screen-space outline rendering.
/// Must match the layout in outline_screenspace.wesl.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct OutlineUniform {
    pub(super) color: [f32; 4],
    pub(super) width_pixels: f32,
    pub(super) screen_width: f32,
    pub(super) screen_height: f32,
    pub(super) _padding: f32,
}

/// Resources for the fullscreen screenspace outline pass.
/// Created once with either a single-sample or multisampled texture depending on
/// `sample_count`; the two cases are otherwise identical in structure.
pub(super) struct OutlineScreenspaceResources {
    /// The texture rendered into by the mask pass and sampled by the outline shader.
    /// Single-sample (R8Unorm) when sample_count == 1; multisampled when sample_count > 1.
    pub(super) texture: GpuTexture,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) bind_group: wgpu::BindGroup,
}

/// GPU resources for screen-space selection outline rendering.
pub(super) struct OutlineResources {
    /// Pipeline for rendering selected objects to the mask texture.
    pub(super) mask_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer for outline settings (shared with the screenspace pipeline).
    pub(super) uniform_buffer: wgpu::Buffer,
    /// Screenspace outline pipeline resources — single-sample or MSAA depending on
    /// the renderer's sample_count.
    pub(super) screenspace: OutlineScreenspaceResources,
}

impl OutlineResources {
    pub(super) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
        sample_count: u32,
    ) -> OutlineResources {
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

        let mask_shader = shader_generator
            .generate_outline_mask_shader(device)
            .expect("Failed to generate outline mask shader");
        let mask_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Outline Mask Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });
        let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Outline Mask Pipeline"),
            layout: Some(&mask_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mask_shader,
                entry_point: Some("vs_mask"),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mask_shader,
                entry_point: Some("fs_mask"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GpuTexture::MASK_FORMAT,
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
                format: GpuTexture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let screenspace = Self::create_screenspace_resources(
            device, config, shader_generator, &uniform_buffer, sample_count,
        );

        OutlineResources { mask_pipeline, uniform_buffer, screenspace }
    }

    /// Recreate size-dependent resources (mask texture and bind group) after a resize.
    pub(super) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, sample_count: u32) {
        let multisampled = sample_count > 1;
        let label = if multisampled { "Outline Mask Color Attachment" } else { "Outline Mask Texture" };
        self.screenspace.texture = GpuTexture::mask(device, width, height, sample_count, label);
        self.screenspace.bind_group = Self::create_bind_group(
            device,
            &self.screenspace.bind_group_layout,
            &self.screenspace.texture.view,
            &self.uniform_buffer,
        );
    }

    fn create_screenspace_resources(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        shader_generator: &mut ShaderGenerator,
        uniform_buffer: &wgpu::Buffer,
        sample_count: u32,
    ) -> OutlineScreenspaceResources {
        let multisampled = sample_count > 1;
        let label = if multisampled { "Outline Mask Color Attachment" } else { "Outline Mask Texture" };
        let texture = GpuTexture::mask(device, config.width, config.height, sample_count, label);

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

        let bind_group = Self::create_bind_group(device, &bind_group_layout, &texture.view, uniform_buffer);

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
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        OutlineScreenspaceResources { texture, pipeline, bind_group_layout, bind_group }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Outline Screenspace Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
