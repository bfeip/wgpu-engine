use std::collections::HashMap;

use crate::scene::{AlphaMode, MaterialProperties, PrimitiveType};
use crate::shaders::ShaderGenerator;

use super::gpu_resources::{
    instance_buffer_layout, vertex_buffer_layout, GpuTexture, MaterialPipelineLayouts,
    PipelineCacheKey,
};

/// Cached render pipelines and the resources needed to create new ones.
pub(super) struct PipelineCache {
    cache: HashMap<PipelineCacheKey, wgpu::RenderPipeline>,
    pipelines: MaterialPipelineLayouts,
    shader_generator: ShaderGenerator,
    sample_count: u32,
    surface_format: wgpu::TextureFormat,
}

impl PipelineCache {
    pub(super) fn new(
        pipelines: MaterialPipelineLayouts,
        shader_generator: ShaderGenerator,
        sample_count: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            cache: HashMap::new(),
            pipelines,
            shader_generator,
            sample_count,
            surface_format,
        }
    }

    pub(super) fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        cache_key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        let material_props = cache_key.material_props.clone();
        let scene_props = cache_key.scene_props.clone();
        let primitive_type = cache_key.primitive_type;
        let depth_prepass = cache_key.depth_prepass;
        self.cache.entry(cache_key).or_insert_with(|| {
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
                .generate_shader(device, &shader_material_props, &scene_props, depth_prepass)
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

            // Lines/points use LessEqual so they render correctly on coplanar triangles
            // (drawn after triangles, same depth buffer value). Blend materials also use
            // LessEqual so the main pass can render at the depth the prepass established.
            let is_blend = material_props.alpha_mode == AlphaMode::Blend;
            let is_non_triangle =
                matches!(primitive_type, PrimitiveType::LineList | PrimitiveType::PointList);
            let depth_compare = if !depth_prepass && (is_blend || is_non_triangle) {
                wgpu::CompareFunction::LessEqual
            } else {
                wgpu::CompareFunction::Less
            };

            // Push triangle faces slightly away from the camera so edge lines on curved
            // surfaces remain visible. Tessellated chords sit above the true curve at
            // chord midpoints, so without a bias the face depth genuinely wins over the
            // line. slope_scale accounts for grazing angles where chord error is worst.
            // clamp=0 avoids the DEPTH_BIAS_CLAMP feature requirement.
            let depth_bias = if primitive_type == PrimitiveType::TriangleList {
                wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                }
            } else {
                wgpu::DepthBiasState::default()
            };

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        format: self.surface_format,
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
                    depth_compare,
                    stencil: wgpu::StencilState::default(),
                    bias: depth_bias,
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
