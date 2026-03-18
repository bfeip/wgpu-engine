use crate::scene::{AlphaMode, MaterialProperties, PrimitiveType};

use super::gpu_resources::{instance_buffer_layout, vertex_buffer_layout, GpuTexture, PipelineCacheKey};
use super::Renderer;

impl Renderer {
    pub(super) fn get_or_create_pipeline(
        &mut self,
        cache_key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        let material_props = cache_key.material_props.clone();
        let scene_props = cache_key.scene_props.clone();
        let primitive_type = cache_key.primitive_type;
        let depth_prepass = cache_key.depth_prepass;
        self.pipeline_cache.entry(cache_key).or_insert_with(|| {
            // Select pipeline layout based on material type and scene properties
            let use_ibl = scene_props.has_ibl && material_props.has_lighting;
            let pipeline_layout = if use_ibl {
                &self.pipelines.pbr_ibl
            } else if material_props.has_lighting {
                &self.pipelines.pbr
            } else {
                &self.pipelines.color
            };

            // For depth prepass, force alpha_mask feature so the shader includes
            // discard logic, reusing the existing alpha_cutoff uniform.
            let shader_material_props = if depth_prepass {
                MaterialProperties {
                    alpha_mode: AlphaMode::Mask,
                    ..material_props.clone()
                }
            } else {
                material_props.clone()
            };

            let shader = self
                .shader_generator
                .generate_shader(&self.device, &shader_material_props, &scene_props)
                .expect("Failed to generate shader");

            let topology = match primitive_type {
                PrimitiveType::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                PrimitiveType::LineList => wgpu::PrimitiveTopology::LineList,
                PrimitiveType::PointList => wgpu::PrimitiveTopology::PointList,
            };

            // Depth prepass: write depth only, no color output
            let (color_write_mask, depth_write_enabled) = if depth_prepass {
                (wgpu::ColorWrites::empty(), true)
            } else {
                (
                    wgpu::ColorWrites::ALL,
                    material_props.alpha_mode != AlphaMode::Blend,
                )
            };

            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(if depth_prepass {
                        "Depth Prepass Pipeline"
                    } else {
                        "Render Pipeline"
                    }),
                    layout: Some(pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.config.format,
                            blend: Some(match material_props.alpha_mode {
                                AlphaMode::Blend => wgpu::BlendState::ALPHA_BLENDING,
                                _ => wgpu::BlendState::REPLACE,
                            }),
                            write_mask: color_write_mask,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        // Only cull for single-sided triangles, not for lines, points, or double-sided materials
                        cull_mode: if topology == wgpu::PrimitiveTopology::TriangleList
                            && !material_props.double_sided
                        {
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
                        format: GpuTexture::DEPTH_FORMAT,
                        depth_write_enabled,
                        // Blend materials use LessEqual so the main blending pass can
                        // render at the same depth the prepass established.
                        depth_compare: if !depth_prepass
                            && material_props.alpha_mode == AlphaMode::Blend
                        {
                            wgpu::CompareFunction::LessEqual
                        } else {
                            wgpu::CompareFunction::Less
                        },
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: self.sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                })
        })
    }
}
