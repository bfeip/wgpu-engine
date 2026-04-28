use crate::scene::PrimitiveType;

use super::common::bind_scene_groups;
use super::super::batching::DrawData;
use super::super::gpu_resources::{GpuTexture, PipelineCacheKey, instance_buffer_layout, vertex_buffer_layout};
use super::super::pass_context::{FrameContext, SceneRenderPass};
use super::super::pipeline::PipelineCache;

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
    color_bind_group: wgpu::BindGroup,
}

impl HiddenLineSolidPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
    ) -> Self {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let shader = shader_generator
            .generate_flat_color_shader(device)
            .expect("Failed to generate flat color shader");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hidden Line Solid Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, lights_bgl, material_color_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Hidden Line Solid Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_flat_color"),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_flat_color"),
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
        let color_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Hidden Line Solid Color Buffer"),
            contents: bytemuck::cast_slice(&[1.0f32, 1.0, 1.0, 1.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hidden Line Solid Color Bind Group"),
            layout: material_color_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
        Self { pipeline, color_bind_group }
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
        render_pass.set_bind_group(2, &self.color_bind_group, &[]);
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
// HiddenLineOccludedPass — gray occluded lines for the hidden-line workflow
// ---------------------------------------------------------------------------

/// Hidden-line workflow occluded pass.
///
/// Renders all `LineList` geometry with depth compare `Greater` and no depth write,
/// outputting flat gray. Only line fragments that lie *behind* the solid geometry
/// (i.e. occluded) survive the depth test and are drawn. Because this pass runs
/// before [`HiddenLineEdgesPass`], visible (black) lines will paint over the gray
/// wherever the two coincide.
pub(crate) struct HiddenLineOccludedPass {
    pipeline: wgpu::RenderPipeline,
    color_bind_group: wgpu::BindGroup,
}

impl HiddenLineOccludedPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        material_color_bgl: &wgpu::BindGroupLayout,
        shader_generator: &mut crate::shaders::ShaderGenerator,
    ) -> Self {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        let shader = shader_generator
            .generate_flat_color_shader(device)
            .expect("Failed to generate flat color shader");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hidden Line Occluded Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, lights_bgl, material_color_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Hidden Line Occluded Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_flat_color"),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_flat_color"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: GpuTexture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        let color_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Hidden Line Occluded Color Buffer"),
            contents: bytemuck::cast_slice(&[0.6f32, 0.6, 0.6, 1.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hidden Line Occluded Color Bind Group"),
            layout: material_color_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
        Self { pipeline, color_bind_group }
    }
}

impl SceneRenderPass for HiddenLineOccludedPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        _pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let has_lines = draw_data.all_batches().iter().any(|b| b.primitive_type == PrimitiveType::LineList);
        if !has_lines { return; }

        let (color_view, resolve_target) = ctx.renderer_textures.msaa_views(view);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Hidden Line Occluded Pass"),
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

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, ctx.camera_bind_group, &[]);
        render_pass.set_bind_group(2, &self.color_bind_group, &[]);
        for batch in draw_data.all_batches() {
            if batch.primitive_type != PrimitiveType::LineList { continue; }
            ctx.draw_batch(&mut render_pass, batch);
        }
    }
}
