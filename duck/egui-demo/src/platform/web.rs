//! Web bootstrap: canvas creation, async wgpu init via the event-loop proxy,
//! and the wasm entry point.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use winit::dpi::PhysicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::Window;

use crate::{App, UserEvent, ViewerState, ui};

pub(crate) fn run() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let proxy = event_loop.create_proxy();

    let app = App {
        state: None,
        ui: ui::UiState::default(),
        workflow_index: 0,
        proxy,
        pending_scene_bytes: Rc::new(RefCell::new(None)),
        pending_hdr_bytes: Rc::new(RefCell::new(None)),
    };

    event_loop.spawn_app(app);
}

/// On web, wgpu init is async and cannot block. Create the canvas
/// synchronously, build the viewer state off the event loop, then deliver it
/// back through the proxy as a [`UserEvent::Initialized`].
pub(crate) fn resume(app: &mut App, event_loop: &ActiveEventLoop) {
    if app.state.is_some() {
        return;
    }

    let size = web_canvas_size();
    let window_attrs = Window::default_attributes()
        .with_title("Duck Engine - egui Example")
        .with_append(true)
        .with_inner_size(size);
    let window = Arc::new(
        event_loop
            .create_window(window_attrs)
            .expect("Failed to create window"),
    );

    // Queue the embedded default scene so it loads once the viewer is ready.
    let default_scene = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/default-scene.duck"
    ));
    *app.pending_scene_bytes.borrow_mut() = Some(default_scene.to_vec());

    let proxy = app.proxy.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let state = ViewerState::from_window(window, Some(size)).await;
        let _ = proxy.send_event(UserEvent::Initialized(state));
    });
}

/// Size the canvas to the browser window (in physical pixels).
fn web_canvas_size() -> PhysicalSize<u32> {
    let win = web_sys::window().expect("no window");
    let dpr = win.device_pixel_ratio();
    let width = win.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(1280.0);
    let height = win.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(720.0);
    PhysicalSize::new((width * dpr) as u32, (height * dpr) as u32)
}
