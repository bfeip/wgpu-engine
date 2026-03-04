use crate::scene::PrimitiveType;

use super::gpu_resources::{instance_buffer_layout, vertex_buffer_layout, GpuTexture, PipelineCacheKey};
use super::Renderer;

impl<'a> Renderer<'a> {
    pub(super) fn get_or_create_pipeline(
        &mut self,
        cache_key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        let material_props = cache_key.material_props.clone();
        let scene_props = cache_key.scene_props.clone();
        let primitive_type = cache_key.primitive_type;
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

            let shader = self
                .shader_generator
                .generate_shader(&self.device, &material_props, &scene_props)
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
                        buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
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
}
