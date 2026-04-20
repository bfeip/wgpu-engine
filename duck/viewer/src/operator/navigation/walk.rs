use cgmath::{InnerSpace, Vector3};

use crate::scene::Camera;
use crate::input::{ElementState, Key};
use crate::scene_scale;

/// Mouse look sensitivity in radians per pixel of mouse movement.
const LOOK_SENSITIVITY: f32 = 0.003;

/// Maximum pitch angle (looking up/down) in radians. Just under 90 degrees.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Internal state for walk-style navigation (WASD + mouse look).
pub(super) struct WalkState {
    /// Current yaw angle (horizontal rotation) in radians.
    pub yaw: f32,
    /// Current pitch angle (vertical rotation) in radians.
    pub pitch: f32,
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
    pub fn new() -> Self {
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
    pub fn init_from_camera(&mut self, camera: &Camera) {
        let forward = camera.forward();

        // Calculate yaw from forward vector projected onto XZ plane
        self.yaw = f32::atan2(forward.x, forward.z);

        // Calculate pitch from forward vector's Y component
        let horizontal_length = (forward.x * forward.x + forward.z * forward.z).sqrt();
        self.pitch = f32::atan2(forward.y, horizontal_length);
    }

    /// Handle mouse look based on mouse drag delta.
    pub fn handle_look(&mut self, dx: f32, dy: f32, camera: &mut Camera) {
        // Update yaw (horizontal rotation) - inverted so mouse left turns view left
        self.yaw -= dx * LOOK_SENSITIVITY;

        // Update pitch (vertical rotation) with clamping
        self.pitch -= dy * LOOK_SENSITIVITY;
        self.pitch = self.pitch.clamp(-MAX_PITCH, MAX_PITCH);

        // Update camera target based on new orientation
        self.update_camera_target(camera);
    }

    /// Update camera target based on current yaw and pitch.
    pub fn update_camera_target(&self, camera: &mut Camera) {
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
    pub fn apply_movement(&self, camera: &mut Camera, delta_time: f32, model_radius: f32) -> bool {
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
    pub fn handle_key(&mut self, key: &Key, state: ElementState) -> bool {
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
    pub fn is_moving(&self) -> bool {
        self.forward_pressed || self.backward_pressed || self.left_pressed || self.right_pressed
    }

    /// Reset all pressed key state.
    pub fn reset_keys(&mut self) {
        self.forward_pressed = false;
        self.backward_pressed = false;
        self.left_pressed = false;
        self.right_pressed = false;
    }
}
