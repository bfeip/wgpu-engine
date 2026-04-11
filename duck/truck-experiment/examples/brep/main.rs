use std::sync::Arc;

use cgmath::{Point3, Vector3};
use truck_modeling::builder;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_truck_experiment::{add_body_to_scene, tessellate_body, Body, TessellationOptions};
use duck_engine_viewer::common::RgbaColor;
use duck_engine_viewer::input::{ElementState, Key, NamedKey};
use duck_engine_viewer::scene::{CoordinateSpace, Light, Material};
use duck_engine_viewer::{Viewer, winit_support};

/// Build a B-rep cube using the Truck CAD kernel, tessellate it, and add it to the scene.
fn build_brep_scene(viewer: &mut Viewer) {
    let scene = viewer.scene_mut();

    // Lighting
    scene.clear_lights();
    scene.add_light(
        Light::directional(Vector3::new(-0.3, -1.0, -0.5), RgbaColor::WHITE, 3.0)
            .in_space(CoordinateSpace::Camera),
    );

    // Create a unit cube using Truck's sweep API:
    // vertex → edge (sweep X) → face (sweep Y) → solid (sweep Z)
    let v = builder::vertex(truck_modeling::Point3::new(-0.5, -0.5, -0.5));
    let edge = builder::tsweep(&v, truck_modeling::Vector3::unit_x());
    let face = builder::tsweep(&edge, truck_modeling::Vector3::unit_y());
    let solid = builder::tsweep(&face, truck_modeling::Vector3::unit_z());

    // Convert to our Body type and tessellate
    let body = Body::from_truck_solid(0, &solid);
    let tessellated = tessellate_body(&body, &TessellationOptions::default());

    // Materials
    let face_material = scene.add_material(
        Material::new()
            .with_base_color_factor(RgbaColor {
                r: 0.4,
                g: 0.6,
                b: 0.9,
                a: 1.0,
            })
            .with_metallic_factor(0.3)
            .with_roughness_factor(0.4),
    );
    let line_material = scene.add_material(Material::new().with_line_color(RgbaColor::BLACK));

    // Add tessellated body to scene
    let _map =
        add_body_to_scene(&body, &tessellated, scene, face_material, line_material).unwrap();

    // Fit camera
    let camera = viewer.camera_mut();
    camera.eye = Point3::new(1.5, 1.5, 1.5);
    camera.target = Point3::new(0.0, 0.0, 0.0);

    if let Some(bounds) = viewer.scene().bounding() {
        viewer.camera_mut().fit_to_bounds(&bounds);
    }
}

struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<Viewer<'a>>,
}

impl<'a> App<'a> {
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs =
            Window::default_attributes().with_title("Duck Engine - B-Rep Geometry");

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        let size = window.inner_size();
        let mut viewer = pollster::block_on(Viewer::new(
            Arc::clone(&window),
            size.width,
            size.height,
        ));

        build_brep_scene(&mut viewer);

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
                    eprintln!("Render error: {}", e);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }

        let Some(app_event) = winit_support::convert_window_event(event) else {
            return;
        };
        let viewer = self.viewer.as_mut().unwrap();
        viewer.handle_event(&app_event);

        let duck_engine_viewer::event::Event::KeyboardInput {
            event: key_event, ..
        } = &app_event
        else {
            return;
        };
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return;
        }
        if let Key::Named(NamedKey::Escape) = &key_event.logical_key {
            event_loop.exit();
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
