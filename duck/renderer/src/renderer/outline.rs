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

/// GPU resources for screen-space selection outline rendering.
pub(super) struct OutlineResources {
    /// Mask texture for selected objects (R8Unorm, always sample_count 1).
    /// When MSAA is active, this serves as the resolve target for the
    /// multisampled color attachment. The outline screenspace shader
    /// always samples from this resolved texture.
    pub(super) mask: GpuTexture,
    /// Multisampled mask color attachment for MSAA rendering.
    /// None when sample_count == 1.
    pub(super) msaa_color_attachment: Option<GpuTexture>,
    /// Pipeline for rendering selected objects to mask texture
    pub(super) mask_pipeline: wgpu::RenderPipeline,
    /// Pipeline for fullscreen outline post-process
    pub(super) outline_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer for outline settings
    pub(super) uniform_buffer: wgpu::Buffer,
    /// Bind group layout for post-process shader
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group containing mask texture, sampler, and uniforms
    pub(super) bind_group: wgpu::BindGroup,
}

impl OutlineResources {
    /// Create GPU resources for screen-space selection outline rendering.
    pub(super) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_generator: &mut ShaderGenerator,
        sample_count: u32,
    ) -> OutlineResources {
        // Create mask texture (R8Unorm, always 1x — serves as resolve target when MSAA is active)
        let mask = GpuTexture::mask(device, config.width, config.height, 1, "Outline Mask Texture");

        // Create multisampled mask color attachment when MSAA is active
        let msaa_color_attachment = if sample_count > 1 {
            Some(GpuTexture::mask(device, config.width, config.height, sample_count, "Outline Mask Color Attachment"))
        } else {
            None
        };

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
        let bind_group = Self::create_bind_group(device, &bind_group_layout, &mask, &uniform_buffer);

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

        // Create mask pipeline - renders selected objects to R8Unorm texture.
        // Uses the renderer's sample_count to match the shared depth buffer.
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
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        OutlineResources {
            mask,
            msaa_color_attachment,
            mask_pipeline,
            outline_pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Recreate size-dependent resources (mask texture and bind group) after a resize.
    pub(super) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, sample_count: u32) {
        self.mask = GpuTexture::mask(device, width, height, 1, "Outline Mask Texture");
        self.msaa_color_attachment = if sample_count > 1 {
            Some(GpuTexture::mask(device, width, height, sample_count, "Outline Mask Color Attachment"))
        } else {
            None
        };
        self.bind_group = Self::create_bind_group(device, &self.bind_group_layout, &self.mask, &self.uniform_buffer);
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        mask: &GpuTexture,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Outline Screenspace Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&mask.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&mask.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }
}
