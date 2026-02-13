use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::event::Event;
use crate::input::{ElementState, Key, KeyEvent, MouseButton, MouseScrollDelta, NamedKey, PhysicalKey};
use crate::viewer::Viewer;

#[wasm_bindgen]
pub struct WebViewer {
    viewer: Viewer<'static>,
}

#[wasm_bindgen]
impl WebViewer {
    /// Create a new WebViewer from an HTML canvas element.
    ///
    /// Initializes the wgpu rendering pipeline on the given canvas.
    /// Call this once, then use `update_and_render()` in a requestAnimationFrame loop.
    ///
    /// Usage from JS: `const viewer = await WebViewer.create(canvas);`
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WebViewer, JsValue> {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();

        let viewer = Viewer::from_canvas(canvas).await;
        Ok(WebViewer { viewer })
    }

    /// Call once per frame from requestAnimationFrame.
    /// Updates delta time and renders the scene.
    pub fn update_and_render(&mut self) {
        self.viewer.update();
        if let Err(e) = self.viewer.render() {
            log::error!("Render error: {}", e);
        }
    }

    /// Notify the viewer that the canvas was resized.
    /// Pass the canvas pixel dimensions (CSS size * devicePixelRatio).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewer.handle_event(&Event::Resized((width, height)));
    }

    /// Forward a mousemove event.
    /// - `x`, `y`: cursor position in canvas pixel coordinates
    /// - `dx`, `dy`: movement delta (from MouseEvent.movementX/Y)
    pub fn on_mouse_move(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        self.viewer
            .handle_event(&Event::CursorMoved { position: (x, y) });
        self.viewer
            .handle_event(&Event::MouseMotion { delta: (dx, dy) });
    }

    /// Forward a mousedown event. `button` is the DOM MouseEvent.button value.
    pub fn on_mouse_down(&mut self, button: i32) {
        self.viewer.handle_event(&Event::MouseInput {
            state: ElementState::Pressed,
            button: dom_button(button),
        });
    }

    /// Forward a mouseup event. `button` is the DOM MouseEvent.button value.
    pub fn on_mouse_up(&mut self, button: i32) {
        self.viewer.handle_event(&Event::MouseInput {
            state: ElementState::Released,
            button: dom_button(button),
        });
    }

    /// Forward a wheel event. `delta_x` and `delta_y` are in pixels.
    pub fn on_wheel(&mut self, delta_x: f32, delta_y: f32) {
        self.viewer.handle_event(&Event::MouseWheel {
            delta: MouseScrollDelta::PixelDelta(delta_x, delta_y),
        });
    }

    /// Forward a keydown event.
    /// - `key`: the KeyboardEvent.key string (e.g. "w", "Escape")
    /// - `code`: the KeyboardEvent.keyCode numeric value
    /// - `repeat`: whether this is a repeat event
    pub fn on_key_down(&mut self, key: &str, code: u32, repeat: bool) {
        self.viewer.handle_event(&Event::KeyboardInput {
            event: make_key_event(key, code, ElementState::Pressed, repeat),
            is_synthetic: false,
        });
    }

    /// Forward a keyup event.
    pub fn on_key_up(&mut self, key: &str, code: u32) {
        self.viewer.handle_event(&Event::KeyboardInput {
            event: make_key_event(key, code, ElementState::Released, false),
            is_synthetic: false,
        });
    }

    /// Load a glTF/glb model from raw bytes.
    /// The scene and camera will be set automatically.
    pub fn load_gltf(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let aspect = self.viewer.camera().aspect;
        match crate::load_gltf_scene_from_slice(data, aspect) {
            Ok(result) => {
                self.viewer.set_scene(result.scene);
                let camera = result.camera.unwrap_or_else(|| {
                    camera_for_scene(self.viewer.scene(), aspect)
                });
                self.viewer.set_camera(camera);
                Ok(())
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Load a scene from .wgsc format bytes.
    /// The scene and camera will be set automatically.
    pub fn load_scene(&mut self, data: &[u8]) -> Result<(), JsValue> {
        let scene = crate::Scene::from_bytes(data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let aspect = self.viewer.camera().aspect;
        let camera = camera_for_scene(&scene, aspect);
        self.viewer.set_scene(scene);
        self.viewer.set_camera(camera);
        Ok(())
    }

    /// Clear the current scene.
    pub fn clear_scene(&mut self) {
        self.viewer.set_scene(crate::Scene::new());
    }

    /// Get the number of root nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.viewer.scene().root_nodes().len()
    }

    /// Get the number of meshes in the scene.
    pub fn mesh_count(&self) -> usize {
        self.viewer.scene().meshes.len()
    }
}

/// Convert a DOM MouseEvent.button value to our MouseButton type.
fn dom_button(button: i32) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        3 => MouseButton::Back,
        4 => MouseButton::Forward,
        n => MouseButton::Other(n as u16),
    }
}

/// Convert a DOM KeyboardEvent.key string to our Key type.
fn dom_key(key: &str) -> Key {
    match key {
        "Escape" => Key::Named(NamedKey::Escape),
        "Enter" => Key::Named(NamedKey::Enter),
        "Tab" => Key::Named(NamedKey::Tab),
        "Backspace" => Key::Named(NamedKey::Backspace),
        "Delete" => Key::Named(NamedKey::Delete),
        " " => Key::Named(NamedKey::Space),
        "ArrowLeft" => Key::Named(NamedKey::ArrowLeft),
        "ArrowRight" => Key::Named(NamedKey::ArrowRight),
        "ArrowUp" => Key::Named(NamedKey::ArrowUp),
        "ArrowDown" => Key::Named(NamedKey::ArrowDown),
        "Home" => Key::Named(NamedKey::Home),
        "End" => Key::Named(NamedKey::End),
        "PageUp" => Key::Named(NamedKey::PageUp),
        "PageDown" => Key::Named(NamedKey::PageDown),
        "Control" => Key::Named(NamedKey::Control),
        "Alt" => Key::Named(NamedKey::Alt),
        "Shift" => Key::Named(NamedKey::Shift),
        "Meta" => Key::Named(NamedKey::Super),
        "F1" => Key::Named(NamedKey::F1),
        "F2" => Key::Named(NamedKey::F2),
        "F3" => Key::Named(NamedKey::F3),
        "F4" => Key::Named(NamedKey::F4),
        "F5" => Key::Named(NamedKey::F5),
        "F6" => Key::Named(NamedKey::F6),
        "F7" => Key::Named(NamedKey::F7),
        "F8" => Key::Named(NamedKey::F8),
        "F9" => Key::Named(NamedKey::F9),
        "F10" => Key::Named(NamedKey::F10),
        "F11" => Key::Named(NamedKey::F11),
        "F12" => Key::Named(NamedKey::F12),
        s if s.len() == 1 => {
            let c = s.chars().next().unwrap();
            Key::Character(c)
        }
        _ => Key::Unidentified,
    }
}

/// Build a KeyEvent from DOM keyboard event properties.
fn make_key_event(key: &str, code: u32, state: ElementState, repeat: bool) -> KeyEvent {
    KeyEvent {
        physical_key: PhysicalKey::Code(code),
        logical_key: dom_key(key),
        state,
        repeat,
    }
}

/// Create a camera that fits the scene bounds (same logic as gltf-viewer).
fn camera_for_scene(scene: &crate::Scene, aspect: f32) -> crate::Camera {
    use cgmath::{Point3, Vector3};

    let mut camera = crate::Camera {
        eye: Point3::new(1.0, 1.0, 1.0),
        target: Point3::new(0.0, 0.0, 0.0),
        up: Vector3::new(0.0, 1.0, 0.0),
        aspect,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
        ortho: false,
    };

    if let Some(bounds) = scene.bounding() {
        camera.fit_to_bounds(&bounds);
    }

    camera
}
