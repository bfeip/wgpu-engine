use std::cell::RefCell;
use std::rc::Rc;

use cgmath::MetricSpace;

use crate::camera::Camera;
use crate::event::{CallbackId, Event, EventDispatcher, EventKind};
use crate::operator::{Operator, OperatorId};

/// Internal state for the navigation operator.
struct NavState {
    /// Whether the user is currently dragging (mouse button held down).
    is_dragging: bool,
    /// Azimuth angle in radians (horizontal rotation around target).
    azimuth: f32,
    /// Elevation angle in radians (vertical rotation, clamped to avoid gimbal lock).
    elevation: f32,
    /// Distance from camera to target (orbit radius).
    radius: f32,
}

impl NavState {
    fn new() -> Self {
        Self {
            is_dragging: false,
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

        // Calculate azimuth (horizontal angle around Y-axis, measured from +Z toward +X)
        self.azimuth = direction.x.atan2(direction.z);

        // Calculate elevation (vertical angle from horizontal plane)
        let horizontal_distance = (direction.x * direction.x + direction.z * direction.z).sqrt();
        self.elevation = direction.y.atan2(horizontal_distance);
    }

    /// Update camera position based on current orbit parameters.
    fn update_camera_position(&self, camera: &mut Camera) {
        // Convert spherical coordinates to Cartesian
        let x = camera.target.x + self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = camera.target.y + self.radius * self.elevation.sin();
        let z = camera.target.z + self.radius * self.elevation.cos() * self.azimuth.cos();

        camera.eye = cgmath::point3(x, y, z);
    }
}

/// Operator for camera navigation via the mouse.
///
/// When the left mouse button is held down, moving the mouse will orbit the camera
/// around the scene origin.
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
        // Register MouseInput handler to track dragging state
        let operator_state = self.state.clone();
        let mouse_input_callback = dispatcher.register(EventKind::MouseInput, move |event, ctx| {
            if let Event::MouseInput {
                state: button_state,
                button,
            } = event
            {
                use winit::event::{ElementState, MouseButton};

                let mut s = operator_state.borrow_mut();
                match (button, button_state) {
                    (MouseButton::Left, ElementState::Pressed) => {
                        // Initialize orbit parameters from current camera state
                        s.init_from_camera(&ctx.state.camera);
                        s.is_dragging = true;
                    }
                    (MouseButton::Left, ElementState::Released) => {
                        s.is_dragging = false;
                    }
                    _ => return false,
                }
                true
            } else {
                false
            }
        });

        // Register MouseMotion handler to update camera during drag
        let operator_state = self.state.clone();
        let mouse_motion_callback = dispatcher.register(EventKind::MouseMotion, move |event, ctx| {
            if let Event::MouseMotion { delta } = event {
                let mut s = operator_state.borrow_mut();

                if s.is_dragging {
                    // Update orbit angles based on mouse movement
                    let sensitivity = 0.005;
                    let dx = delta.0 as f32 * sensitivity;
                    let dy = delta.1 as f32 * sensitivity;

                    // Update azimuth (horizontal rotation)
                    s.azimuth -= dx;

                    // Update elevation (vertical rotation) with clamping to avoid gimbal lock
                    s.elevation += dy;
                    const MAX_ELEVATION: f32 = std::f32::consts::FRAC_PI_2 - 0.01; // Just under 90 degrees
                    s.elevation = s.elevation.clamp(-MAX_ELEVATION, MAX_ELEVATION);

                    // Update camera position based on new angles
                    s.update_camera_position(&mut ctx.state.camera);

                    true
                } else {
                    false
                }
            } else {
                false
            }
        });

        self.callback_ids = vec![mouse_input_callback, mouse_motion_callback];
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
