#![allow(dead_code)]

mod texture;
mod camera;
mod light;
mod common;
mod scene;
mod drawstate;
mod material;
mod shaders;
mod gltf;
mod event;
mod operator;

use winit::{
    event_loop::EventLoop,
    keyboard::KeyCode,
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::scene::Scene;
use crate::drawstate::DrawState;
use crate::event::{EventDispatcher, EventKind, EventContext};
use crate::operator::{OperatorManager, NavigationOperator, BuiltinOperatorId};


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

    // Set up event dispatcher
    let mut dispatcher = EventDispatcher::new();

    // Set up operator manager and add navigation operator
    let mut operator_manager = OperatorManager::new();
    let nav_operator = Box::new(NavigationOperator::new(BuiltinOperatorId::Navigation.into()));
    operator_manager.add_operator(nav_operator, 1, &mut dispatcher);

    // Register CloseRequested handler
    dispatcher.register(EventKind::CloseRequested, |_event, ctx| {
        ctx.control_flow.exit();
        true
    });

    // Register Resized handler
    dispatcher.register(EventKind::Resized, |event, ctx| {
        if let crate::event::Event::Resized(physical_size) = event {
            ctx.state.resize(*physical_size);
        }
        true
    });

    // Register RedrawRequested handler
    dispatcher.register(EventKind::RedrawRequested, |_event, ctx| {
        if !ctx.state.window.is_visible().unwrap() {
            ctx.state.window.set_visible(true);
        }
        // This tells winit that we want another frame after this one
        ctx.state.window.request_redraw();

        match ctx.state.render(ctx.scene) {
            Ok(_) => {}
            Err(err) => {
                // Check if the error is a surface error that we can handle
                if let Some(surface_err) = err.downcast_ref::<wgpu::SurfaceError>() {
                    match surface_err {
                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                            ctx.state.resize(ctx.state.size);
                        }
                        wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other => {
                            log::error!("OutOfMemory: {}", surface_err);
                            ctx.control_flow.exit();
                        }
                        wgpu::SurfaceError::Timeout => {
                            log::warn!("Surface timeout: {}", surface_err);
                        }
                    }
                } else {
                    // Handle other types of errors
                    log::error!("Render error: {}", err);
                    ctx.control_flow.exit();
                }
            }
        }
        true
    });

    // Register KeyboardInput handler
    dispatcher.register(EventKind::KeyboardInput, |event, ctx| {
        if let crate::event::Event::KeyboardInput { event: key_event, .. } = event {
            use winit::keyboard::PhysicalKey;

            match key_event.physical_key {
                PhysicalKey::Code(KeyCode::Escape) => {
                    ctx.control_flow.exit();
                },
                _ => return false,
            }
            true
        } else {
            false
        }
    });

    event_loop
        .run(move |event, control_flow| {
            if let Some(app_event) = crate::event::Event::from_winit_event(event) {
                let mut ctx = EventContext {
                    state: &mut state,
                    scene: &mut scene,
                    control_flow,
                };
                dispatcher.dispatch(&app_event, &mut ctx);
            }
        })
        .unwrap();
}
