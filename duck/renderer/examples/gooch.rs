use duck_engine_renderer::{
    DrawData, FrameContext, PipelineCache, Renderer, SceneRenderPass,
};
use duck_engine_renderer::scene::{
    Camera, Light, Material, Mesh, PrimitiveType, Scene,
    common::RgbaColor,
};

use cgmath::{Point3, Vector3};

const GOOCH_SHADER: &str = include_str!("gooch.wgsl");

struct GoochPass {
    pipeline: wgpu::RenderPipeline,
}

impl GoochPass {
    fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        camera_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gooch Shader"),
            source: wgpu::ShaderSource::Wgsl(GOOCH_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gooch Pipeline Layout"),
            bind_group_layouts: &[camera_bgl, lights_bgl],
            push_constant_ranges: &[],
        });

        let vertex_attrs = [
            wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
        ];
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: 36,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };

        let instance_attrs = [
            wgpu::VertexAttribute { offset: 0,  shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 16, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 48, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 64, shader_location: 7, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 76, shader_location: 8, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 88, shader_location: 9, format: wgpu::VertexFormat::Float32x3 },
        ];
        let instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: 100,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attrs,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gooch Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout, instance_buffer_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
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

        GoochPass { pipeline }
    }
}

impl SceneRenderPass for GoochPass {
    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        _pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Gooch Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.08, b: 0.12, a: 1.0 }),
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
        render_pass.set_bind_group(1, ctx.lights_bind_group, &[]);

        for batch in draw_data.all_batches() {
            if batch.primitive_type == PrimitiveType::TriangleList {
                ctx.draw_batch(&mut render_pass, batch);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let width = 800u32;
    let height = 600u32;

    let mut renderer = pollster::block_on(Renderer::new_headless(width, height));

    // Build scene: UV sphere with a plain unlit material.
    // The Gooch pass ignores material bind groups and drives colour purely from
    // normal direction, so any valid material works here.
    let mut scene = Scene::new();
    let mesh_id = scene.add_mesh(Mesh::sphere(1.0, 48, 24, PrimitiveType::TriangleList));
    let mat_id = scene.add_material(Material::new());
    scene.add_instance_node(
        None, mesh_id, mat_id,
        Some("sphere".to_string()),
        Default::default(),
    )?;

    // A warm directional light from upper-left so the Gooch interpolation has
    // a meaningful gradient across the sphere.
    scene.add_light(Light::directional(
        Vector3::new(-0.5, -1.0, -0.5),
        RgbaColor { r: 1.0, g: 0.95, b: 0.8, a: 1.0 },
        1.0,
    ));

    let camera = Camera {
        eye: Point3::new(0.0, 0.0, 3.5),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect: width as f32 / height as f32,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
        ortho: false,
    };

    // Build and install the Gooch pass, replacing the default pass list.
    let gooch = GoochPass::new(
        renderer.device(),
        renderer.surface_format(),
        renderer.sample_count(),
        renderer.camera_bind_group_layout(),
        renderer.lights_bind_group_layout(),
    );
    renderer.set_passes(vec![Box::new(gooch)]);

    let image = renderer.render_scene_to_image(&camera, &mut scene, None)?;
    image.save("gooch.png")?;
    println!("Saved gooch.png ({width}×{height})");

    Ok(())
}
