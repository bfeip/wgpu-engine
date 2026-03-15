use std::cell::RefCell;
use std::rc::Rc;

use cgmath::{InnerSpace, MetricSpace, Vector3};

use crate::scene::Camera;
use crate::event::{CallbackId, Event, EventDispatcher, EventKind};
use crate::input::{ElementState, Key, MouseButton};
use crate::operator::{Operator, OperatorId};
use crate::scene_scale;

// ── Navigation mode ─────────────────────────────────────────────────────────

/// The navigation interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavigationMode {
    /// Orbit/pan/zoom around a target point (default).
    #[default]
    Orbit,
    /// First-person walk with WASD movement and mouse look.
    Walk,
}

// ======== Orbit State ========

/// Internal state for orbit-style navigation (orbit / pan / zoom).
struct OrbitState {
    /// Azimuth angle in radians (horizontal rotation around target).
    azimuth: f32,
    /// Elevation angle in radians (vertical rotation).
    elevation: f32,
    /// Base distance from camera to target.
    radius: f32,
}

impl OrbitState {
    fn new() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            radius: 5.0,
        }
    }

    /// Initialize parameters from current camera state.
    fn init_from_camera(&mut self, camera: &Camera) {
        self.radius = camera.length();

        // Calculate direction vector from target to eye
        let direction = camera.eye - camera.target;

        // Calculate azimuth (horizontal angle around Y-axis)
        self.azimuth = f32::atan2(direction.x, direction.z);

        // Calculate elevation (vertical angle from horizontal plane)
        let horizontal_distance =
            f32::sqrt(direction.x * direction.x + direction.z * direction.z);
        self.elevation = f32::atan2(direction.y, horizontal_distance);
    }

    /// Update camera position based on current orbit parameters.
    fn update_camera_position(&self, camera: &mut Camera) {
        // Convert spherical coordinates to Cartesian
        let x = camera.target.x + self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = camera.target.y + self.radius * self.elevation.sin();
        let z = camera.target.z + self.radius * self.elevation.cos() * self.azimuth.cos();

        camera.eye = cgmath::point3(x, y, z);

        // Update up vector to maintain proper orientation
        let forward = (camera.target - camera.eye).normalize();
        let world_up = cgmath::vec3(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        camera.up = forward.cross(right).normalize();
    }

    /// Handle zoom via mouse wheel by adjusting camera distance.
    fn handle_zoom(&mut self, delta: f32, camera: &mut Camera, model_radius: f32) {
        let zoom_factor = scene_scale::zoom_factor();
        let factor = if delta > 0.0 {
            1.0 - zoom_factor // zoom in
        } else {
            1.0 + zoom_factor // zoom out
        };

        self.radius *= factor.powf(delta.abs());
        self.radius = self.radius.clamp(
            scene_scale::min_camera_radius(model_radius),
            scene_scale::max_camera_radius(model_radius),
        );

        self.update_camera_position(camera);
    }

    /// Handle orbit rotation based on mouse movement.
    fn handle_orbit(&mut self, dx: f64, dy: f64, camera: &mut Camera) {
        const ORBIT_SENSITIVITY: f32 = 0.005;

        let dx = dx as f32 * ORBIT_SENSITIVITY;
        let dy = dy as f32 * ORBIT_SENSITIVITY;

        self.azimuth -= dx;

        self.elevation += dy;
        const MAX_ELEVATION: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        self.elevation = self.elevation.clamp(-MAX_ELEVATION, MAX_ELEVATION);

        self.update_camera_position(camera);
    }

    /// Handle panning based on mouse movement.
    fn handle_pan(&self, dx: f64, dy: f64, camera: &mut Camera, model_radius: f32) {
        let dx = dx as f32;
        let dy = dy as f32;

        let right = camera.right();
        let up_view = camera.up;

        let pan_scale = scene_scale::pan_sensitivity(model_radius);

        let offset = right * (-dx * pan_scale) + up_view * (dy * pan_scale);
        camera.eye += offset;
        camera.target += offset;
    }
}

// ======== Walk State ========

/// Mouse look sensitivity in radians per pixel of mouse movement.
const LOOK_SENSITIVITY: f32 = 0.003;

/// Maximum pitch angle (looking up/down) in radians. Just under 90 degrees.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Internal state for walk-style navigation (WASD + mouse look).
struct WalkState {
    /// Current yaw angle (horizontal rotation) in radians.
    yaw: f32,
    /// Current pitch angle (vertical rotation) in radians.
    pitch: f32,
    /// Whether the W key is pressed.
    forward_pressed: bool,
    /// Whether the S key is pressed.
    backward_pressed: bool,
    /// Whether the A key is pressed.
    left_pressed: bool,
    /// Whether the D key is pressed.
    right_pressed: bool,
}

impl WalkState {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            forward_pressed: false,
            backward_pressed: false,
            left_pressed: false,
            right_pressed: false,
        }
    }

    /// Initialize yaw and pitch from current camera orientation.
    fn init_from_camera(&mut self, camera: &Camera) {
        let forward = camera.forward();

        // Calculate yaw from forward vector projected onto XZ plane
        self.yaw = f32::atan2(forward.x, forward.z);

        // Calculate pitch from forward vector's Y component
        let horizontal_length = (forward.x * forward.x + forward.z * forward.z).sqrt();
        self.pitch = f32::atan2(forward.y, horizontal_length);
    }

    /// Handle mouse look based on mouse drag delta.
    fn handle_look(&mut self, dx: f32, dy: f32, camera: &mut Camera) {
        // Update yaw (horizontal rotation) - inverted so mouse left turns view left
        self.yaw -= dx * LOOK_SENSITIVITY;

        // Update pitch (vertical rotation) with clamping
        self.pitch -= dy * LOOK_SENSITIVITY;
        self.pitch = self.pitch.clamp(-MAX_PITCH, MAX_PITCH);

        // Update camera target based on new orientation
        self.update_camera_target(camera);
    }

    /// Update camera target based on current yaw and pitch.
    fn update_camera_target(&self, camera: &mut Camera) {
        // Preserve the current eye-to-target distance
        let distance = (camera.target - camera.eye).magnitude();

        // Calculate new forward direction from yaw and pitch
        let forward = Vector3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        );

        // Set target at the preserved distance in front of eye
        camera.target = camera.eye + forward * distance;

        // Update up vector to maintain proper orientation
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        let right = forward.cross(world_up).normalize();
        camera.up = right.cross(forward).normalize();
    }

    /// Apply movement based on currently pressed keys.
    /// Returns true if any movement was applied.
    fn apply_movement(&self, camera: &mut Camera, delta_time: f32, model_radius: f32) -> bool {
        let mut movement = Vector3::new(0.0, 0.0, 0.0);

        // Get horizontal forward direction (project onto XZ plane)
        let forward_flat = Vector3::new(self.yaw.sin(), 0.0, self.yaw.cos()).normalize();

        // Get right direction (perpendicular to forward on XZ plane)
        let right = Vector3::new(-self.yaw.cos(), 0.0, self.yaw.sin());

        if self.forward_pressed {
            movement += forward_flat;
        }
        if self.backward_pressed {
            movement -= forward_flat;
        }
        if self.right_pressed {
            movement += right;
        }
        if self.left_pressed {
            movement -= right;
        }

        if movement.magnitude2() > 0.0 {
            let walk_speed = scene_scale::walk_speed(model_radius);
            movement = movement.normalize() * walk_speed * delta_time;
            camera.eye += movement;
            camera.target += movement;
            return true;
        }

        false
    }

    /// Handle key press/release for movement keys.
    /// Returns true if the key was handled.
    fn handle_key(&mut self, key: &Key, state: ElementState) -> bool {
        let pressed = state == ElementState::Pressed;

        match key {
            Key::Character('w') | Key::Character('W') => {
                self.forward_pressed = pressed;
                true
            }
            Key::Character('s') | Key::Character('S') => {
                self.backward_pressed = pressed;
                true
            }
            Key::Character('a') | Key::Character('A') => {
                self.left_pressed = pressed;
                true
            }
            Key::Character('d') | Key::Character('D') => {
                self.right_pressed = pressed;
                true
            }
            _ => false,
        }
    }

    /// Check if any movement key is currently pressed.
    fn is_moving(&self) -> bool {
        self.forward_pressed || self.backward_pressed || self.left_pressed || self.right_pressed
    }

    /// Reset all pressed key state.
    fn reset_keys(&mut self) {
        self.forward_pressed = false;
        self.backward_pressed = false;
        self.left_pressed = false;
        self.right_pressed = false;
    }
}

// ── Combined state ──────────────────────────────────────────────────────────

/// Combined internal state for the navigation operator.
struct NavigationState {
    mode: Rc<RefCell<NavigationMode>>,
    orbit: OrbitState,
    walk: WalkState,
}

impl NavigationState {
    fn new(mode: Rc<RefCell<NavigationMode>>) -> Self {
        Self {
            mode,
            orbit: OrbitState::new(),
            walk: WalkState::new(),
        }
    }

    fn mode(&self) -> NavigationMode {
        *self.mode.borrow()
    }

    fn handle_drag_start(&mut self, button: &MouseButton, camera: &Camera) -> bool {
        match (self.mode(), button) {
            (NavigationMode::Orbit, MouseButton::Left | MouseButton::Right) => {
                self.orbit.init_from_camera(camera);
                true
            }
            (NavigationMode::Walk, MouseButton::Left) => {
                self.walk.init_from_camera(camera);
                true
            }
            _ => false,
        }
    }

    fn handle_drag(
        &mut self,
        button: &MouseButton,
        delta: &(f32, f32),
        camera: &mut Camera,
        model_radius: f32,
    ) -> bool {
        match (self.mode(), button) {
            (NavigationMode::Orbit, MouseButton::Left) => {
                self.orbit
                    .handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Orbit, MouseButton::Right) => {
                self.orbit
                    .handle_pan(delta.0 as f64, delta.1 as f64, camera, model_radius);
                true
            }
            (NavigationMode::Walk, MouseButton::Left) => {
                self.walk.handle_look(delta.0, delta.1, camera);
                true
            }
            _ => false,
        }
    }

    fn handle_wheel(&mut self, scroll_amount: f32, camera: &mut Camera, model_radius: f32) -> bool {
        if self.mode() != NavigationMode::Orbit {
            return false;
        }
        self.orbit.init_from_camera(camera);
        self.orbit.handle_zoom(scroll_amount, camera, model_radius);
        true
    }

    fn handle_keyboard(
        &mut self,
        key_event: &crate::input::KeyEvent,
        camera: &Camera,
    ) -> bool {
        if self.mode() != NavigationMode::Walk {
            return false;
        }
        if key_event.state == ElementState::Pressed && !key_event.repeat {
            self.walk.init_from_camera(camera);
        }
        self.walk.handle_key(&key_event.logical_key, key_event.state)
    }

    fn handle_update(
        &mut self,
        delta_time: f32,
        camera: &mut Camera,
        model_radius: f32,
    ) -> bool {
        if self.mode() != NavigationMode::Walk {
            self.walk.reset_keys();
            return false;
        }
        if delta_time > 1.0 {
            return false;
        }
        if self.walk.is_moving() {
            self.walk.apply_movement(camera, delta_time, model_radius);
            return true;
        }
        false
    }
}

// ── Operator ────────────────────────────────────────────────────────────────

/// Operator for camera navigation with two modes:
///
/// **Orbit mode** (default):
/// - Left mouse button + drag: Orbit camera around target
/// - Right mouse button + drag: Pan camera perpendicular to view direction
/// - Mouse wheel: Zoom in/out (adjust camera distance from target)
///
/// **Walk mode**:
/// - W/S: Move forward/backward
/// - A/D: Strafe left/right
/// - Left mouse drag: Look around (rotate view)
///
/// Use [`NavigationOperator::mode_handle`] to get a shared handle for
/// reading or changing the active mode.
pub struct NavigationOperator {
    id: OperatorId,
    state: Rc<RefCell<NavigationState>>,
    mode: Rc<RefCell<NavigationMode>>,
    callback_ids: Vec<CallbackId>,
}

impl NavigationOperator {
    /// Creates a new navigation operator with the given ID.
    pub fn new(id: OperatorId) -> Self {
        let mode = Rc::new(RefCell::new(NavigationMode::default()));
        Self {
            id,
            state: Rc::new(RefCell::new(NavigationState::new(mode.clone()))),
            mode,
            callback_ids: Vec::new(),
        }
    }

    /// Returns a shared handle to the navigation mode.
    pub fn mode_handle(&self) -> Rc<RefCell<NavigationMode>> {
        self.mode.clone()
    }
}

impl Operator for NavigationOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        let s = self.state.clone();
        let drag_start_cb = dispatcher.register(EventKind::MouseDragStart, move |event, ctx| {
            let Event::MouseDragStart { button, .. } = event else { return false };
            s.borrow_mut().handle_drag_start(button, ctx.renderer.camera())
        });

        let s = self.state.clone();
        let drag_cb = dispatcher.register(EventKind::MouseDrag, move |event, ctx| {
            let Event::MouseDrag { button, delta, .. } = event else { return false };
            let model_radius = scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());
            s.borrow_mut().handle_drag(button, delta, ctx.renderer.camera_mut(), model_radius)
        });

        let s = self.state.clone();
        let wheel_cb = dispatcher.register(EventKind::MouseWheel, move |event, ctx| {
            let Event::MouseWheel { delta } = event else { return false };
            use crate::input::MouseScrollDelta;
            let scroll_amount = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(_x, y) => *y / 100.0,
            };
            let model_radius = scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());
            s.borrow_mut().handle_wheel(scroll_amount, ctx.renderer.camera_mut(), model_radius)
        });

        let s = self.state.clone();
        let keyboard_cb = dispatcher.register(EventKind::KeyboardInput, move |event, ctx| {
            let Event::KeyboardInput { event: key_event, .. } = event else { return false };
            s.borrow_mut().handle_keyboard(key_event, ctx.renderer.camera())
        });

        let s = self.state.clone();
        let update_cb = dispatcher.register(EventKind::Update, move |event, ctx| {
            let Event::Update { delta_time } = event else { return false };
            let model_radius = scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());
            s.borrow_mut().handle_update(*delta_time, ctx.renderer.camera_mut(), model_radius)
        });

        self.callback_ids = vec![drag_start_cb, drag_cb, wheel_cb, keyboard_cb, update_cb];
    }

    fn deactivate(&mut self, dispatcher: &mut EventDispatcher) {
        for id in &self.callback_ids {
            dispatcher.unregister(*id);
        }
        self.callback_ids.clear();
    }

    fn id(&self) -> OperatorId {
        self.id
    }

    fn name(&self) -> &str {
        "Navigation"
    }

    fn callback_ids(&self) -> &[CallbackId] {
        &self.callback_ids
    }

    fn is_active(&self) -> bool {
        !self.callback_ids.is_empty()
    }
}
