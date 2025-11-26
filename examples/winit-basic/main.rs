use std::sync::Arc;
use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
};

use wgpu_engine::{Viewer, winit_support};

async fn run() {
    // Initialize logging
    env_logger::init();

    // Create event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("WGPU Engine - Winit Basic Example")
            .with_visible(false)
            .build(&event_loop)
            .unwrap(),
    );

    let size = window.inner_size();

    // Create viewer with the window as surface target
    let mut viewer = Viewer::new(Arc::clone(&window), size.width, size.height).await;

    // Make window visible and trigger initial render
    window.set_visible(true);
    window.request_redraw();

    // Run the event loop
    event_loop
        .run(move |event, control_flow| {
            // Handle window close directly
            if matches!(
                &event,
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::CloseRequested,
                    ..
                }
            ) {
                control_flow.exit();
                return;
            }

            // Request next frame after each redraw
            if matches!(
                &event,
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::RedrawRequested,
                    ..
                }
            ) {
                window.request_redraw();
            }

            // Convert winit event to our event type
            if let Some(app_event) = winit_support::convert_event(event) {
                // Handle the event through the viewer
                viewer.handle_event(&app_event);

                // Check for exit on Escape key
                if let wgpu_engine::Event::KeyboardInput { event: key_event, .. } = &app_event {
                    if matches!(
                        key_event.logical_key,
                        wgpu_engine::Key::Named(wgpu_engine::NamedKey::Escape)
                    ) {
                        control_flow.exit();
                    }
                }
            }
        })
        .unwrap();
}

fn main() {
    pollster::block_on(run());
}
