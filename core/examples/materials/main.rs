use std::sync::Arc;

use cgmath::{Point3, Quaternion, Vector3};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use wgpu_engine::common::RgbaColor;
use wgpu_engine::scene::{Light, Material, MaterialFlags, Mesh};
use wgpu_engine::{Viewer, winit_support};

const SPHERE_RADIUS: f32 = 0.4;
const SPHERE_SEGMENTS: u32 = 32;
const SPHERE_RINGS: u32 = 16;
const COLS: usize = 5;
const ROWS: usize = 5;
const SPACING: f32 = 1.2;

/// Compute the grid position for a (row, col) cell, centered at the origin.
fn grid_position(row: usize, col: usize) -> Point3<f32> {
    let x = (col as f32 - (COLS as f32 - 1.0) / 2.0) * SPACING;
    let z = (row as f32 - (ROWS as f32 - 1.0) / 2.0) * SPACING;
    Point3::new(x, 0.0, z)
}

/// Build the material showcase scene: a 5x5 grid of spheres demonstrating PBR properties.
fn build_material_scene(viewer: &mut Viewer) {
    let scene = viewer.scene_mut();

    // Replace default lighting with a three-light setup
    scene.lights.clear();
    scene.lights.push(Light::directional(
        Vector3::new(-1.0, 1.0, -1.0),
        RgbaColor::WHITE,
        3.0,
    ));
    //scene.lights.push(Light::point(
    //    Vector3::new(-4.0, 3.0, -2.0),
    //    RgbaColor { r: 0.7, g: 0.8, b: 1.0, a: 1.0 },
    //    50.0,
    //));
    //scene.lights.push(Light::point(
    //    Vector3::new(2.0, 4.0, 4.0),
    //    RgbaColor { r: 1.0, g: 0.95, b: 0.85, a: 1.0 },
    //    40.0,
    //));

    // Shared sphere mesh for all instances
    let mesh_id = scene.add_mesh(Mesh::sphere(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_RINGS));

    let identity_rot = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let unit_scale = Vector3::new(1.0, 1.0, 1.0);

    // Row 0: Roughness gradient (dielectric blue, roughness 0.0 -> 1.0)
    for col in 0..COLS {
        let roughness = col as f32 / (COLS - 1) as f32;
        let mat = Material::new()
            .with_base_color_factor(RgbaColor::BLUE)
            .with_metallic_factor(0.0)
            .with_roughness_factor(roughness);
        let mat_id = scene.add_material(mat);
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some(format!("Roughness {roughness:.2}")),
                grid_position(0, col),
                identity_rot,
                unit_scale,
            )
            .unwrap();
    }

    // Row 1: Metallic gradient (blue, roughness 0.3, metallic 0.0 -> 1.0)
    for col in 0..COLS {
        let metallic = col as f32 / (COLS - 1) as f32;
        let mat = Material::new()
            .with_base_color_factor(RgbaColor::BLUE)
            .with_roughness_factor(0.3)
            .with_metallic_factor(metallic);
        let mat_id = scene.add_material(mat);
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some(format!("Metallic {metallic:.2}")),
                grid_position(1, col),
                identity_rot,
                unit_scale,
            )
            .unwrap();
    }

    // Row 2: Base colors (dielectric, roughness 0.4)
    let colors = [
        ("Red", RgbaColor::RED),
        ("Green", RgbaColor::GREEN),
        ("Blue", RgbaColor::BLUE),
        ("White", RgbaColor::WHITE),
        ("Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in colors.iter().enumerate() {
        let mat = Material::new()
            .with_base_color_factor(*color)
            .with_metallic_factor(0.0)
            .with_roughness_factor(0.4);
        let mat_id = scene.add_material(mat);
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some(name.to_string()),
                grid_position(2, col),
                identity_rot,
                unit_scale,
            )
            .unwrap();
    }

    // Row 3: Metallic colors (metallic 1.0, roughness 0.25)
    let metals = [
        ("Red", RgbaColor::RED),
        ("Green", RgbaColor::GREEN),
        ("Blue", RgbaColor::BLUE),
        ("White", RgbaColor::WHITE),
        ("Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in metals.iter().enumerate() {
        let mat = Material::new()
            .with_base_color_factor(*color)
            .with_metallic_factor(1.0)
            .with_roughness_factor(0.25);
        let mat_id = scene.add_material(mat);
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some(name.to_string()),
                grid_position(3, col),
                identity_rot,
                unit_scale,
            )
            .unwrap();
    }

    // Row 4: Unlit (DO_NOT_LIGHT) vs lit
    let unlit_colors = [
        ("Unlit Red", RgbaColor::RED),
        ("Unlit Green", RgbaColor::GREEN),
        ("Unlit Blue", RgbaColor::BLUE),
        ("Unlit White", RgbaColor::WHITE),
        ("Unlit Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in unlit_colors.iter().enumerate() {
        let mat = Material::new()
            .with_base_color_factor(*color)
            .with_flags(MaterialFlags::DO_NOT_LIGHT);
        let mat_id = scene.add_material(mat);
        scene
            .add_instance_node(
                None,
                mesh_id,
                mat_id,
                Some(name.to_string()),
                grid_position(4, col),
                identity_rot,
                unit_scale,
            )
            .unwrap();
    }

    // Camera: elevated view looking down at the grid
    let camera = viewer.camera_mut();
    camera.eye = Point3::new(0.0, 6.0, 8.0);
    camera.target = Point3::new(0.0, 0.0, 0.0);

    if let Some(bounds) = viewer.scene().bounding() {
        viewer.camera_mut().fit_to_bounds(&bounds);
    }
}

/// Application state for the winit event loop.
struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<Viewer<'a>>,
}

impl<'a> App<'a> {
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs =
            Window::default_attributes().with_title("WGPU Engine - PBR Materials Showcase");

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        let size = window.inner_size();
        let mut viewer = pollster::block_on(Viewer::new(
            Arc::clone(&window),
            size.width,
            size.height,
        ));

        build_material_scene(&mut viewer);

        window.request_redraw();
        self.window = Some(window);
        self.viewer = Some(viewer);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.initialize(event_loop);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let viewer = self.viewer.as_mut().unwrap();
                viewer.update();
                if let Err(e) = viewer.render() {
                    log::error!("Render error: {}", e);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }

        if let Some(app_event) = winit_support::convert_window_event(event) {
            let viewer = self.viewer.as_mut().unwrap();
            viewer.handle_event(&app_event);

            if let wgpu_engine::event::Event::KeyboardInput {
                event: key_event, ..
            } = &app_event
            {
                if matches!(
                    key_event.logical_key,
                    wgpu_engine::input::Key::Named(wgpu_engine::input::NamedKey::Escape)
                ) {
                    event_loop.exit();
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(app_event) = winit_support::convert_device_event(event) {
            self.viewer.as_mut().unwrap().handle_event(&app_event);
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        window: None,
        viewer: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
