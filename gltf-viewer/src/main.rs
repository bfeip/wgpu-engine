mod app;
mod ui;

use winit::event_loop::EventLoop;

use app::{App, AppEvent};

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let event_loop = EventLoop::<AppEvent>::with_user_event().build().unwrap();
    let mut app = App::new(event_loop.create_proxy());
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use winit::platform::web::EventLoopExtWebSys;

    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");

    let event_loop = EventLoop::<AppEvent>::with_user_event().build().unwrap();
    let app = App::new(event_loop.create_proxy());
    event_loop.spawn_app(app);
}
