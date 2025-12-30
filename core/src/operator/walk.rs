use std::cell::RefCell;
use std::rc::Rc;

use cgmath::{InnerSpace, Vector3};

use crate::camera::Camera;
use crate::event::{CallbackId, Event, EventDispatcher, EventKind};
use crate::input::{ElementState, Key, MouseButton};
use crate::operator::{Operator, OperatorId};
use crate::scene_scale;

/// Mouse look sensitivity in radians per pixel of mouse movement.
const LOOK_SENSITIVITY: f32 = 0.003;

/// Maximum pitch angle (looking up/down) in radians. Just under 90 degrees.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Internal state for the walk operator.
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
    /// Uses delta_time for frame-rate independent movement.
    /// Returns true if any movement was applied.
    fn apply_movement(&self, camera: &mut Camera, delta_time: f32, model_radius: f32) -> bool {
        let mut movement = Vector3::new(0.0, 0.0, 0.0);

        // Get horizontal forward direction (project onto XZ plane for ground-based movement)
        let forward_flat = Vector3::new(
            self.yaw.sin(),
            0.0,
            self.yaw.cos(),
        ).normalize();

        // Get right direction (perpendicular to forward on XZ plane)
        // right = forward_flat × up = (sin(yaw), 0, cos(yaw)) × (0, 1, 0)
        let right = Vector3::new(
            -self.yaw.cos(),
            0.0,
            self.yaw.sin(),
        );

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

        // Normalize and apply speed scaled to model size
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
}

/// Operator for first-person "walk" style navigation.
///
/// Controls:
/// - W: Move forward
/// - S: Move backward
/// - A: Strafe left
/// - D: Strafe right
/// - Left mouse drag: Look around (rotate view)
///
/// Movement is constrained to the horizontal plane (no flying).
/// Looking around uses mouse drag rather than mouse movement to match
/// BIM/engineering application conventions.
pub struct WalkOperator {
    id: OperatorId,
    state: Rc<RefCell<WalkState>>,
    callback_ids: Vec<CallbackId>,
}

impl WalkOperator {
    /// Creates a new walk operator with the given ID.
    pub fn new(id: OperatorId) -> Self {
        Self {
            id,
            state: Rc::new(RefCell::new(WalkState::new())),
            callback_ids: Vec::new(),
        }
    }
}

impl Operator for WalkOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        // Register MouseDragStart handler to initialize look parameters
        let operator_state = self.state.clone();
        let drag_start_callback = dispatcher.register(EventKind::MouseDragStart, move |event, ctx| {
            if let Event::MouseDragStart { button: MouseButton::Left, .. } = event {
                let mut s = operator_state.borrow_mut();
                s.init_from_camera(&ctx.state.camera);
                true
            } else {
                false
            }
        });

        // Register MouseDrag handler for looking around
        let operator_state = self.state.clone();
        let drag_callback = dispatcher.register(EventKind::MouseDrag, move |event, ctx| {
            if let Event::MouseDrag { button: MouseButton::Left, delta, .. } = event {
                let mut s = operator_state.borrow_mut();
                s.handle_look(delta.0, delta.1, &mut ctx.state.camera);
                true
            } else {
                false
            }
        });

        // Register KeyboardInput handler to track WASD key state
        let operator_state = self.state.clone();
        let keyboard_callback = dispatcher.register(EventKind::KeyboardInput, move |event, ctx| {
            if let Event::KeyboardInput { event: key_event, .. } = event {
                let mut s = operator_state.borrow_mut();

                // Initialize camera orientation on first key press
                if key_event.state == ElementState::Pressed && !key_event.repeat {
                    s.init_from_camera(&ctx.state.camera);
                }

                // Update held key state (movement applied in Update handler)
                s.handle_key(&key_event.logical_key, key_event.state)
            } else {
                false
            }
        });

        // Register Update handler for continuous movement while keys are held
        let operator_state = self.state.clone();
        let update_callback = dispatcher.register(EventKind::Update, move |event, ctx| {
            if let Event::Update { delta_time } = event {
                if *delta_time > 1.0 {
                    // Do not apply movement if there's more than a second
                    // between updates.
                    return false;
                }
                let s = operator_state.borrow();
                if s.is_moving() {
                    let model_radius = scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());
                    s.apply_movement(&mut ctx.state.camera, *delta_time, model_radius);
                    return true;
                }
            }
            false
        });

        self.callback_ids = vec![drag_start_callback, drag_callback, keyboard_callback, update_callback];
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
        "Walk"
    }

    fn callback_ids(&self) -> &[CallbackId] {
        &self.callback_ids
    }

    fn is_active(&self) -> bool {
        !self.callback_ids.is_empty()
    }
}
