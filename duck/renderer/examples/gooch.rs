use duck_engine_renderer::{
    DrawData, FrameContext, PipelineCache, Renderer, RenderWorkflow, SceneRenderPass,
};
use duck_engine_renderer::scene::{
    Camera, Light, Material, Mesh, PrimitiveType, Scene,
    common::RgbaColor,
};

use cgmath::{Point3, Vector3};

const GOOCH_WESL: &str = include_str!("gooch.wesl");

struct GoochPass {
    pipeline: wgpu::RenderPipeline,
}

impl GoochPass {
    fn new(renderer: &Renderer) -> Self {
        let shader = renderer.compile_user_wesl(GOOCH_WESL)
            .expect("failed to compile gooch shader");
        let pipeline = renderer.custom_pipeline_builder()
            .shader(&shader, "vs_main", "fs_main")
            .label("Gooch")
            .build();
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

struct GoochWorkflow {
    pass: GoochPass,
}

impl GoochWorkflow {
    fn new(renderer: &Renderer) -> Self {
        Self { pass: GoochPass::new(renderer) }
    }
}

impl RenderWorkflow for GoochWorkflow {
    fn name(&self) -> &'static str { "Gooch" }

    fn resize(&mut self, _device: &wgpu::Device, _size: (u32, u32), _sample_count: u32) {}

    fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        ctx: &FrameContext<'_>,
        pipeline_cache: &mut PipelineCache,
        draw_data: &DrawData,
    ) {
        self.pass.execute(encoder, view, ctx, pipeline_cache, draw_data);
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

    renderer.set_workflow(Box::new(GoochWorkflow::new(&renderer)));

    let image = renderer.render_scene_to_image(&camera, &mut scene, None)?;
    image.save("gooch.png")?;
    println!("Saved gooch.png ({width}×{height})");

    Ok(())
}
