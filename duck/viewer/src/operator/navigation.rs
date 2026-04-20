use std::cell::RefCell;
use std::rc::Rc;

use crate::common;
use crate::scene::{Camera, geom_query::pick_all_from_ray};
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::input::MouseButton;
use crate::operator::{Operator, OperatorId};
use crate::scene_scale;

mod turntable;
mod trackball;
mod walk;

use turntable::TurntableState;
use trackball::TrackballState;
use walk::WalkState;

pub(super) const ORBIT_SENSITIVITY: f32 = 0.005;

pub(super) fn pan(dx: f32, dy: f32, camera: &mut Camera, viewport: (u32, u32)) {
    let (width, height) = viewport;
    let pivot = camera.target;
    let movement_plane = common::Plane::from_point(camera.forward(), pivot);
    let screen = camera.project_point_screen(pivot, width, height);
    let diff_ray = camera.ray_from_screen_point(screen.x - dx, screen.y - dy, width, height);
    if let Some((_, new_pivot)) = movement_plane.intersect_ray(&diff_ray) {
        let offset = new_pivot - pivot;
        camera.eye += offset;
        camera.target += offset;
    }
}

pub(super) fn zoom_radius(radius: f32, delta: f32, model_radius: f32) -> f32 {
    let zoom_factor = scene_scale::zoom_factor();
    let factor = if delta > 0.0 { 1.0 - zoom_factor } else { 1.0 + zoom_factor };
    (radius * factor.powf(delta.abs())).clamp(
        scene_scale::min_camera_radius(model_radius),
        scene_scale::max_camera_radius(model_radius),
    )
}

// ── Navigation mode ─────────────────────────────────────────────────────────

/// The navigation interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavigationMode {
    /// Orbit/pan/zoom around a target point with the camera kept upright (default).
    #[default]
    Turntable,
    /// Orbit/pan/zoom allowing unrestricted camera roll. Good for CAD workflows
    /// where parts may be in arbitrary orientations.
    Trackball,
    /// First-person walk with WASD movement and mouse look.
    Walk,
}

// ── Combined state ──────────────────────────────────────────────────────────

/// Combined internal state for the navigation operator.
struct NavigationState {
    mode: Rc<RefCell<NavigationMode>>,
    turntable: TurntableState,
    trackball: TrackballState,
    walk: WalkState,
}

impl NavigationState {
    fn new(mode: Rc<RefCell<NavigationMode>>) -> Self {
        Self {
            mode,
            turntable: TurntableState::new(),
            trackball: TrackballState::new(),
            walk: WalkState::new(),
        }
    }

    fn mode(&self) -> NavigationMode {
        *self.mode.borrow()
    }

    fn handle_drag_start(
        &mut self,
        button: &MouseButton,
        start_pos: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let camera = &*ctx.camera;
        match (self.mode(), button) {
            (NavigationMode::Turntable, MouseButton::Left | MouseButton::Right) => {
                let pivot = ctx.scene.bounding().map(|b| b.center()).unwrap_or(camera.target);
                self.turntable.init_with_pivot(ctx.camera, pivot);
                true
            }
            (NavigationMode::Trackball, MouseButton::Left | MouseButton::Right) => {
                let pivot = ctx.scene.bounding().map(|b| b.center()).unwrap_or(camera.target);
                self.trackball.init(ctx.camera, pivot);
                true
            }
            (_, MouseButton::Middle) => {
                let ray = camera.ray_from_screen_point(
                    start_pos.0,
                    start_pos.1,
                    ctx.size.0,
                    ctx.size.1,
                );
                let hits = pick_all_from_ray(&ray, ctx.scene);
                let pivot = if let Some(hit) = hits.first() {
                    hit.hit_point
                } else {
                    ctx.scene
                        .bounding()
                        .map(|b| b.center())
                        .unwrap_or(camera.target)
                };
                match self.mode() {
                    NavigationMode::Turntable => self.turntable.init_with_pivot(ctx.camera, pivot),
                    NavigationMode::Trackball => self.trackball.init(ctx.camera, pivot),
                    NavigationMode::Walk => {}
                }
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
        viewport: (u32, u32),
    ) -> bool {
        match (self.mode(), button) {
            (NavigationMode::Turntable, MouseButton::Left) => {
                self.turntable.handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Turntable, MouseButton::Right) => {
                self.turntable.handle_pan(delta.0, delta.1, camera, viewport);
                true
            }
            (NavigationMode::Trackball, MouseButton::Left) => {
                self.trackball.handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Trackball, MouseButton::Right) => {
                self.trackball.handle_pan(delta.0, delta.1, camera, viewport);
                true
            }
            (_, MouseButton::Middle) => {
                match self.mode() {
                    NavigationMode::Turntable => {
                        self.turntable.handle_orbit(delta.0 as f64, delta.1 as f64, camera)
                    }
                    NavigationMode::Trackball => {
                        self.trackball.handle_orbit(delta.0 as f64, delta.1 as f64, camera)
                    }
                    NavigationMode::Walk => {}
                }
                true
            }
            (NavigationMode::Walk, MouseButton::Left) => {
                self.walk.handle_look(delta.0, delta.1, camera);
                true
            }
            _ => false,
        }
    }

    fn handle_drag_end(&mut self, button: &MouseButton) {
        if *button == MouseButton::Middle {
            self.turntable.pivot = None;
            self.trackball.pivot = None;
        }
    }

    fn handle_wheel(&mut self, scroll_amount: f32, camera: &mut Camera, model_radius: f32) -> bool {
        match self.mode() {
            NavigationMode::Turntable => {
                self.turntable.init(camera);
                self.turntable.handle_zoom(scroll_amount, camera, model_radius);
                true
            }
            NavigationMode::Trackball => {
                self.trackball.handle_zoom(scroll_amount, camera, model_radius);
                true
            }
            NavigationMode::Walk => false,
        }
    }

    fn handle_keyboard(
        &mut self,
        key_event: &crate::input::KeyEvent,
        camera: &Camera,
    ) -> bool {
        if self.mode() != NavigationMode::Walk {
            return false;
        }
        if key_event.state == crate::input::ElementState::Pressed && !key_event.repeat {
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

/// Operator for camera navigation. See [`NavigationMode`] for available modes.
///
/// **Turntable / Trackball mode** (default: Turntable):
/// - Left mouse button + drag: Orbit camera around target
/// - Right mouse button + drag: Pan camera perpendicular to view direction
/// - Middle mouse button + drag: Orbit around point under cursor (hit geometry or scene center)
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
            let Event::MouseDragStart { button, start_pos, .. } = event else { return false };
            s.borrow_mut().handle_drag_start(button, *start_pos, ctx)
        });

        let s = self.state.clone();
        let drag_cb = dispatcher.register(EventKind::MouseDrag, move |event, ctx| {
            let Event::MouseDrag { button, delta, .. } = event else { return false };
            s.borrow_mut().handle_drag(button, delta, ctx.camera, ctx.size)
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
            s.borrow_mut().handle_wheel(scroll_amount, ctx.camera, model_radius)
        });

        let s = self.state.clone();
        let keyboard_cb = dispatcher.register(EventKind::KeyboardInput, move |event, ctx| {
            let Event::KeyboardInput { event: key_event, .. } = event else { return false };
            s.borrow_mut().handle_keyboard(key_event, ctx.camera)
        });

        let s = self.state.clone();
        let update_cb = dispatcher.register(EventKind::Update, move |event, ctx| {
            let Event::Update { delta_time } = event else { return false };
            let model_radius = scene_scale::model_radius_from_bounds(ctx.scene.bounding().as_ref());
            s.borrow_mut().handle_update(*delta_time, ctx.camera, model_radius)
        });

        let s = self.state.clone();
        let drag_end_cb = dispatcher.register(EventKind::MouseDragEnd, move |event, _ctx| {
            let Event::MouseDragEnd { button, .. } = event else { return false };
            s.borrow_mut().handle_drag_end(button);
            false
        });

        self.callback_ids = vec![drag_start_cb, drag_cb, drag_end_cb, wheel_cb, keyboard_cb, update_cb];
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
