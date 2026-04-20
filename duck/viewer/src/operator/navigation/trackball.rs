use cgmath::{InnerSpace, Point3, Rotation};

use crate::scene::{Camera, common::quaternion_from_axis_angle_safe};
use crate::scene_scale;

use super::ORBIT_SENSITIVITY;

/// Internal state for trackball-style orbit (camera-local axes, unrestricted roll).
pub(super) struct TrackballState {
    pub radius: f32,
    /// Custom orbit pivot point. When set, orbit rotates the camera around
    /// this point instead of `camera.target`.
    pub pivot: Option<Point3<f32>>,
}

impl TrackballState {
    pub fn new() -> Self {
        Self { radius: 5.0, pivot: None }
    }

    pub fn init(&mut self, camera: &Camera) {
        self.pivot = None;
        self.radius = camera.length();
    }

    pub fn init_with_pivot(&mut self, camera: &Camera, pivot: Point3<f32>) {
        self.pivot = Some(pivot);
        self.radius = camera.length();
    }

    /// Orbit around `camera.target` using camera-local yaw and pitch axes.
    /// `camera.up` is carried forward by the same rotation, accumulating roll.
    pub fn handle_orbit(&mut self, dx: f64, dy: f64, camera: &mut Camera) {
        if self.pivot.is_some() {
            return self.handle_pivot_orbit(dx, dy, camera);
        }

        let dx = dx as f32 * ORBIT_SENSITIVITY;
        let dy = dy as f32 * ORBIT_SENSITIVITY;

        let right = camera.right();
        let yaw_rot = quaternion_from_axis_angle_safe(camera.up, -dx);
        let pitch_rot = quaternion_from_axis_angle_safe(right, dy);
        let rotation = pitch_rot * yaw_rot;

        let offset = camera.eye - camera.target;
        camera.eye = camera.target + rotation.rotate_vector(offset);
        camera.up = rotation.rotate_vector(camera.up);
    }

    /// Handle orbit around the pivot point using incremental rotations.
    ///
    /// Rotates both eye and target around the pivot by the same rotation,
    /// keeping the pivot visually stationary on screen.
    fn handle_pivot_orbit(&mut self, dx: f64, dy: f64, camera: &mut Camera) {
        let pivot = self.pivot.unwrap();

        let dx = dx as f32 * ORBIT_SENSITIVITY;
        let dy = dy as f32 * ORBIT_SENSITIVITY;

        let right = camera.right();
        let yaw_rot = quaternion_from_axis_angle_safe(camera.up, -dx);
        let pitch_rot = quaternion_from_axis_angle_safe(right, dy);
        let rotation = pitch_rot * yaw_rot;

        camera.eye = pivot + rotation.rotate_vector(camera.eye - pivot);
        camera.target = pivot + rotation.rotate_vector(camera.target - pivot);
        camera.up = rotation.rotate_vector(camera.up);
    }

    /// Handle zoom via mouse wheel by adjusting camera distance.
    pub fn handle_zoom(&mut self, delta: f32, camera: &mut Camera, model_radius: f32) {
        let zoom_factor = scene_scale::zoom_factor();
        let factor = if delta > 0.0 { 1.0 - zoom_factor } else { 1.0 + zoom_factor };
        self.radius = camera.length();
        self.radius *= factor.powf(delta.abs());
        self.radius = self.radius.clamp(
            scene_scale::min_camera_radius(model_radius),
            scene_scale::max_camera_radius(model_radius),
        );
        let dir = (camera.eye - camera.target).normalize();
        camera.eye = camera.target + dir * self.radius;
    }

    /// Handle panning based on mouse movement.
    pub fn handle_pan(&self, dx: f64, dy: f64, camera: &mut Camera, model_radius: f32) {
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
