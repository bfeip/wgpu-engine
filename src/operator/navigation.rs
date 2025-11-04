use std::cell::RefCell;
use std::rc::Rc;

use crate::event::{CallbackId, Event, EventDispatcher, EventKind};
use crate::operator::{Operator, OperatorId};

/// Internal state for the navigation operator.
struct NavState {
    /// Whether the user is currently dragging (mouse button held down).
    is_dragging: bool,
}

impl NavState {
    fn new() -> Self {
        Self {
            is_dragging: false,
        }
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
        let mouse_input_callback = dispatcher.register(EventKind::MouseInput, move |event, _ctx| {
            if let Event::MouseInput {
                state: button_state,
                button,
            } = event
            {
                use winit::event::{ElementState, MouseButton};

                let mut s = operator_state.borrow_mut();
                match (button, button_state) {
                    (MouseButton::Left, ElementState::Pressed) => {
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
                let s = operator_state.borrow();

                if s.is_dragging {
                    // Update camera rotation based on mouse movement
                    let sensitivity = 0.005;
                    let dx = delta.0 as f32 * sensitivity;

                    ctx.state.camera_rotation_radians -= dx;
                    ctx.state.camera_rotation_radians %= std::f32::consts::TAU;

                    // Update camera position in an orbit around the origin
                    let angle = ctx.state.camera_rotation_radians;
                    let radius = 5.0;
                    let x = f32::sin(angle) * radius;
                    let y = ctx.state.camera.eye.y;
                    let z = f32::cos(angle) * radius;
                    ctx.state.camera.eye = cgmath::point3(x, y, z);

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
