#![allow(dead_code)]

mod texture;
mod camera;
mod light;
mod common;
mod scene;
mod drawstate;
mod material;
mod shaders;

use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::KeyCode,
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::scene::Scene;
use crate::drawstate::DrawState;


enum VertexShaderLocations {
    VertexPosition = 0,
    TextureCoords,
    VertexNormal,
    InstanceTransformRow0,
    InstanceTransformRow1,
    InstanceTransformRow2,
    InstanceTransformRow3,
    InstanceNormalRow0,
    InstanceNormalRow1,
    InstanceNormalRow2,
}


fn handle_window_event(
    event: &WindowEvent,
    control_flow: &winit::event_loop::EventLoopWindowTarget<()>,
    state: &mut DrawState,
    scene: &mut Scene
) {
    match event {
        WindowEvent::CloseRequested => control_flow.exit(),
        WindowEvent::KeyboardInput { event, .. } => handle_key_input(event, control_flow, state),
        WindowEvent::Resized(physical_size) => state.resize(*physical_size),
        WindowEvent::RedrawRequested => {
            if !state.window.is_visible().unwrap() {
                state.window.set_visible(true);
            }
            // This tells winit that we want another frame after this one
            state.window.request_redraw();

            match state.render(scene) {
                Ok(_) => {}
                Err(err) => {
                    // Check if the error is a surface error that we can handle
                    if let Some(surface_err) = err.downcast_ref::<wgpu::SurfaceError>() {
                        match surface_err {
                            wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                                state.resize(state.size);
                            }
                            wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other => {
                                log::error!("OutOfMemory: {}", surface_err);
                                control_flow.exit();
                            }
                            wgpu::SurfaceError::Timeout => {
                                log::warn!("Surface timeout: {}", surface_err);
                            }
                        }
                    } else {
                        // Handle other types of errors
                        log::error!("Render error: {}", err);
                        control_flow.exit();
                    }
                }
            }
        }
        _ => {}
    }
}

fn handle_key_input(
    event: &KeyEvent,
    control_flow: &winit::event_loop::EventLoopWindowTarget<()>,
    state: &mut DrawState
) {
    let mut update_camera = |angle| {
        let x = f32::sin(angle) * 5.;
        let y = state.camera.eye.y;
        let z = f32::cos(angle) * 5.;
        state.camera.eye = cgmath::point3(x, y, z);
    };

    use winit::keyboard::PhysicalKey;
    match event.physical_key {
        PhysicalKey::Code(KeyCode::Escape) => control_flow.exit(),
        PhysicalKey::Code(KeyCode::ArrowLeft) => {
            state.camera_rotation_radians = state.camera_rotation_radians - 0.1 % std::f32::consts::TAU;
            update_camera(state.camera_rotation_radians);
        },
        PhysicalKey::Code(KeyCode::ArrowRight) => {
            state.camera_rotation_radians = state.camera_rotation_radians + 0.1 % std::f32::consts::TAU;
            update_camera(state.camera_rotation_radians);
        }
        _ => {}
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
    .with_title("WGPU")
    .with_visible(false)
    .build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    // DEMO CODE
    let mut state = DrawState::new(&window).await;
    let texture = state.material_manager.create_texture_material_from_path(
        &state.device,
        &state.queue,
        "/home/zachary/src/wgpu-engine/src/happy-tree.png"
    ).unwrap();
    let mut scene = Scene::demo(&state.device, texture);

    event_loop
        .run(move |event, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window.id() => {
                    handle_window_event(event, control_flow, &mut state, &mut scene);
                }
                _ => {}
            }
        })
        .unwrap();
}
