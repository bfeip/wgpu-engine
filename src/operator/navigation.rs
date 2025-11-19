use std::cell::RefCell;
use std::rc::Rc;

use cgmath::MetricSpace;

use crate::camera::Camera;
use crate::event::{CallbackId, Event, EventDispatcher, EventKind};
use crate::operator::{Operator, OperatorId};

/// Internal state for the navigation operator.
struct NavState {
    /// Azimuth angle in radians (horizontal rotation around target).
    azimuth: f32,
    /// Elevation angle in radians (vertical rotation).
    elevation: f32,
    /// Base distance from camera to target.
    radius: f32,
}

// TODO: in the future these should be proportional to the model bounds
// Camera distance bounds for zoom
const MIN_RADIUS: f32 = 0.01; // Minimum camera distance
const MAX_RADIUS: f32 = 50.0; // Maximum camera distance

impl NavState {
    fn new() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            radius: 5.0,
        }
    }

    /// Initialize parameters from current camera state.
    fn init_from_camera(&mut self, camera: &Camera) {
        // Calculate radius (distance from eye to target)
        self.radius = camera.eye.distance(camera.target);

        // Calculate direction vector from target to eye
        let direction = camera.eye - camera.target;

        // Calculate azimuth (horizontal angle around Y-axis)
        self.azimuth = f32::atan2(direction.x, direction.z);

        // Calculate elevation (vertical angle from horizontal plane)
        let horizontal_distance = f32::sqrt(direction.x * direction.x + direction.z * direction.z);
        self.elevation = f32::atan2(direction.y, horizontal_distance);
    }

    /// Update camera position based on current orbit parameters.
    fn update_camera_position(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;

        // Convert spherical coordinates to Cartesian
        let x = camera.target.x + self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = camera.target.y + self.radius * self.elevation.sin();
        let z = camera.target.z + self.radius * self.elevation.cos() * self.azimuth.cos();

        camera.eye = cgmath::point3(x, y, z);

        // Update up vector to maintain proper orientation
        // Calculate forward (view direction) and right vectors
        let forward = (camera.target - camera.eye).normalize();
        let world_up = cgmath::vec3(0.0, 1.0, 0.0);

        // Calculate right vector (perpendicular to forward and world up)
        let right = world_up.cross(forward).normalize();

        // Recalculate up vector to ensure it's perpendicular to both forward and right
        camera.up = forward.cross(right).normalize();
    }

    /// Handle zoom via mouse wheel by adjusting camera distance.
    /// Positive delta = zoom in (decrease radius)
    /// Negative delta = zoom out (increase radius)
    fn handle_zoom(&mut self, delta: f32, camera: &mut Camera) {
        // TODO: in the future zoom amount should be proportional to model radius.
        const ZOOM_SENSITIVITY: f32 = 0.01; // Distance units per wheel unit

        // Adjust zoom offset and clamp (positive delta decreases radius = zoom in)
        self.radius -= delta * ZOOM_SENSITIVITY;
        self.radius = self.radius.clamp(MIN_RADIUS, MAX_RADIUS);

        // Update camera position with new radius
        self.update_camera_position(camera);
    }

    /// Handle orbit rotation based on mouse movement.
    /// Updates azimuth (horizontal) and elevation (vertical) angles.
    fn handle_orbit(&mut self, dx: f64, dy: f64, camera: &mut Camera) {
        const ORBIT_SENSITIVITY: f32 = 0.005;

        let dx = dx as f32 * ORBIT_SENSITIVITY;
        let dy = dy as f32 * ORBIT_SENSITIVITY;

        // Update azimuth (horizontal rotation)
        self.azimuth -= dx;

        // Update elevation (vertical rotation) with clamping to avoid gimbal lock
        self.elevation += dy;
        const MAX_ELEVATION: f32 = std::f32::consts::FRAC_PI_2 - 0.01; // Just under 90 degrees
        self.elevation = self.elevation.clamp(-MAX_ELEVATION, MAX_ELEVATION);

        // Update camera position based on new angles
        self.update_camera_position(camera);
    }

    /// Handle panning based on mouse movement.
    /// Moves both camera eye and target perpendicular to the view direction.
    fn handle_pan(&self, dx: f64, dy: f64, camera: &mut Camera) {
        const PAN_SENSITIVITY: f32 = 1.0;

        let dx = dx as f32 * PAN_SENSITIVITY;
        let dy = dy as f32 * PAN_SENSITIVITY;

        // Get camera right and up vectors
        let right = camera.right();
        let up_view = camera.up;

        // Scale pan speed by distance from target
        let pan_scale = camera.length() * 0.001;

        // Move both eye and target perpendicular to view direction
        let offset = right * (-dx * pan_scale) + up_view * (dy * pan_scale);
        camera.eye += offset;
        camera.target += offset;
    }
}

/// Operator for camera navigation via the mouse.
///
/// - Left mouse button + drag: Orbit camera around target
/// - Right mouse button + drag: Pan camera perpendicular to view direction
/// - Mouse wheel: Zoom in/out (adjust camera distance from target)
pub struct NavigationOperator {
    id: OperatorId,
    state: Rc<RefCell<NavState>>,
    callback_ids: Vec<CallbackId>,
}

impl NavigationOperator {
    /// Creates a new navigation operator with the given ID.
    pub fn new(id: OperatorId) -> Self {
        Self {
            id,
            state: Rc::new(RefCell::new(NavState::new())),
            callback_ids: Vec::new(),
        }
    }
}

impl Operator for NavigationOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        // Register MouseDragStart handler to initialize orbit/pan parameters
        let operator_state = self.state.clone();
        let drag_start_callback = dispatcher.register(EventKind::MouseDragStart, move |event, ctx| {
            if let Event::MouseDragStart { button, .. } = event {
                use winit::event::MouseButton;

                match button {
                    MouseButton::Left | MouseButton::Right => {
                        // Initialize orbit/pan parameters from current camera state
                        let mut s = operator_state.borrow_mut();
                        s.init_from_camera(&ctx.state.camera);
                        true
                    }
                    _ => false,
                }
            } else {
                false
            }
        });

        // Register MouseDrag handler to update camera during drag
        let operator_state = self.state.clone();
        let drag_callback = dispatcher.register(EventKind::MouseDrag, move |event, ctx| {
            if let Event::MouseDrag { button, delta, .. } = event {
                use winit::event::MouseButton;

                let mut s = operator_state.borrow_mut();

                match button {
                    MouseButton::Left => {
                        // Orbit camera
                        s.handle_orbit(delta.0 as f64, delta.1 as f64, &mut ctx.state.camera);
                        true
                    }
                    MouseButton::Right => {
                        // Pan camera
                        s.handle_pan(delta.0 as f64, delta.1 as f64, &mut ctx.state.camera);
                        true
                    }
                    _ => false,
                }
            } else {
                false
            }
        });

        // Register MouseWheel handler for zoom
        let operator_state = self.state.clone();
        let mouse_wheel_callback = dispatcher.register(EventKind::MouseWheel, move |event, ctx| {
            if let Event::MouseWheel { delta } = event {
                use winit::event::MouseScrollDelta;

                let mut s = operator_state.borrow_mut();

                // Extract the scroll amount (positive = zoom in, negative = zoom out)
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32, // Scale pixel delta
                };

                s.init_from_camera(&ctx.state.camera);
                s.handle_zoom(scroll_amount, &mut ctx.state.camera);

                true
            } else {
                false
            }
        });

        self.callback_ids = vec![drag_start_callback, drag_callback, mouse_wheel_callback];
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
