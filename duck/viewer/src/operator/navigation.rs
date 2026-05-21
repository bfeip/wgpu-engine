use std::cell::RefCell;
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::common;
use crate::scene::{PositionedCamera, geom_query::{pick_all_from_ray, RayPickQuery}};
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::input::{Key, Modifiers, MouseButton, MouseScrollDelta};
use crate::operator::Operator;
use crate::scene_scale;

mod turntable;
mod trackball;
mod walk;

use turntable::TurntableState;
use trackball::TrackballState;
use walk::WalkState;

pub(super) const ORBIT_SENSITIVITY: f32 = 0.005;

pub(super) fn pan(dx: f32, dy: f32, camera: &mut PositionedCamera, viewport: (u32, u32)) {
    let (width, height) = viewport;
    let pivot = camera.target;
    let movement_plane = common::Plane::from_point(camera.forward(), pivot);
    let screen = camera.project_point_screen(pivot, width, height);
    let diff_ray = camera.ray_from_screen_point(screen.x - dx, screen.y - dy, width, height);
    if let Some((_, new_pivot)) = diff_ray.intersect_plane(&movement_plane) {
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
    /// First-person walk with movement keys and mouse look.
    Walk,
}

// ── Navigation actions ───────────────────────────────────────────────────────

/// Semantic actions for the navigation operator across all modes.
///
/// Mouse drag actions apply to turntable/trackball and walk modes:
/// - [`Orbit`] rotates around the target (or free-looks in walk mode)
/// - [`Pan`] moves the camera parallel to the view plane
/// - [`PivotOrbit`] picks a new pivot under the cursor before orbiting
/// - [`Zoom`] adjusts camera distance (scroll or pinch)
///
/// Key actions apply exclusively to walk mode:
/// - [`MoveForward`], [`MoveBackward`], [`MoveLeft`], [`MoveRight`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NavigationAction {
    /// Orbit (turntable/trackball) or free-look (walk).
    Orbit,
    /// Pan the camera parallel to the view plane.
    Pan,
    /// Pick a new pivot point under the cursor, then orbit around it.
    PivotOrbit,
    /// Zoom in/out via scroll wheel or pinch gesture.
    Zoom,
    /// Walk mode: move forward.
    MoveForward,
    /// Walk mode: move backward.
    MoveBackward,
    /// Walk mode: strafe left.
    MoveLeft,
    /// Walk mode: strafe right.
    MoveRight,
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
        action: NavigationAction,
        start_pos: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let camera = ctx.camera();
        let scene_center =
            ctx.scene.bounding().bounds.map(|b| b.center()).unwrap_or(camera.target);

        match (self.mode(), action) {
            (NavigationMode::Turntable, NavigationAction::Orbit | NavigationAction::Pan) => {
                self.turntable.init_with_pivot(&camera, scene_center);
                true
            }
            (NavigationMode::Trackball, NavigationAction::Orbit | NavigationAction::Pan) => {
                self.trackball.init(&camera, scene_center);
                true
            }
            (_, NavigationAction::PivotOrbit) => {
                let ray = camera.ray_from_screen_point(
                    start_pos.0, start_pos.1, ctx.size.0, ctx.size.1,
                );
                let hits = pick_all_from_ray(&RayPickQuery::faces(ray), ctx.scene);
                let pivot = hits.first().map(|h| h.hit_point).unwrap_or(scene_center);
                match self.mode() {
                    NavigationMode::Turntable => self.turntable.init_with_pivot(&camera, pivot),
                    NavigationMode::Trackball => self.trackball.init(&camera, pivot),
                    NavigationMode::Walk => {}
                }
                true
            }
            (NavigationMode::Walk, NavigationAction::Orbit) => {
                self.walk.init_from_camera(&camera);
                true
            }
            _ => false,
        }
    }

    fn handle_drag(
        &mut self,
        action: NavigationAction,
        delta: &(f32, f32),
        camera: &mut PositionedCamera,
        viewport: (u32, u32),
    ) -> bool {
        match (self.mode(), action) {
            (NavigationMode::Turntable, NavigationAction::Orbit) => {
                self.turntable.handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Turntable, NavigationAction::Pan) => {
                self.turntable.handle_pan(delta.0, delta.1, camera, viewport);
                true
            }
            (NavigationMode::Trackball, NavigationAction::Orbit) => {
                self.trackball.handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Trackball, NavigationAction::Pan) => {
                self.trackball.handle_pan(delta.0, delta.1, camera, viewport);
                true
            }
            (NavigationMode::Turntable | NavigationMode::Trackball, NavigationAction::PivotOrbit) => {
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
            (NavigationMode::Walk, NavigationAction::Orbit) => {
                self.walk.handle_look(delta.0, delta.1, camera);
                true
            }
            _ => false,
        }
    }

    fn handle_drag_end(&mut self, action: NavigationAction) {
        if action == NavigationAction::PivotOrbit {
            self.turntable.pivot = None;
            self.trackball.pivot = None;
        }
    }

    fn handle_wheel(
        &mut self,
        scroll_amount: f32,
        camera: &mut PositionedCamera,
        model_radius: f32,
    ) -> bool {
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
        camera: &PositionedCamera,
        bindings: &InputMap<NavigationAction>,
    ) -> bool {
        if self.mode() != NavigationMode::Walk {
            return false;
        }
        if key_event.state == crate::input::ElementState::Pressed && !key_event.repeat {
            self.walk.init_from_camera(camera);
        }
        self.walk.handle_key(&key_event.logical_key, key_event.state, bindings)
    }

    fn handle_update(
        &mut self,
        delta_time: f32,
        camera: &mut PositionedCamera,
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
/// - Orbit drag: Orbit camera around target
/// - Pan drag: Pan camera perpendicular to view direction
/// - Pivot-orbit drag: Orbit around point under cursor
/// - Mouse wheel: Zoom in/out
///
/// **Walk mode**:
/// - Movement key bindings: Move forward/backward/strafe
/// - Orbit drag: Look around
///
/// Use [`NavigationOperator::mode_handle`] to get a shared handle for
/// reading or changing the active mode. Use [`NavigationOperator::bindings`]
/// to remap any action.
pub struct NavigationOperator {
    state: Rc<RefCell<NavigationState>>,
    mode: Rc<RefCell<NavigationMode>>,
    /// All navigation bindings: orbit/pan/zoom drags and walk movement keys.
    pub bindings: Rc<RefCell<InputMap<NavigationAction>>>,
    callback_ids: Vec<CallbackId>,
}

impl NavigationOperator {
    /// Creates a new navigation operator with default bindings.
    pub fn new() -> Self {
        let mode = Rc::new(RefCell::new(NavigationMode::default()));
        let bindings = InputMap::new()
            // Orbit/pan/pivot drag
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Left, modifiers: Modifiers::default() },
                NavigationAction::Orbit,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Left, modifiers: Modifiers::default() },
                NavigationAction::Orbit,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Right, modifiers: Modifiers::default() },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Right, modifiers: Modifiers::default() },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDragStart {
                    button: MouseButton::Left,
                    modifiers: Modifiers { shift: true, ..Default::default() },
                },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDrag {
                    button: MouseButton::Left,
                    modifiers: Modifiers { shift: true, ..Default::default() },
                },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Middle, modifiers: Modifiers::default() },
                NavigationAction::PivotOrbit,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Middle, modifiers: Modifiers::default() },
                NavigationAction::PivotOrbit,
            )
            .bind(
                InputBinding::MouseDragEnd { button: MouseButton::Middle, modifiers: Modifiers::default() },
                NavigationAction::PivotOrbit,
            )
            // Zoom
            .bind(InputBinding::MouseScroll, NavigationAction::Zoom)
            // Walk movement keys
            .bind(
                InputBinding::Key { key: Key::Character('w'), modifiers: Modifiers::default() },
                NavigationAction::MoveForward,
            )
            .bind(
                InputBinding::Key { key: Key::Character('s'), modifiers: Modifiers::default() },
                NavigationAction::MoveBackward,
            )
            .bind(
                InputBinding::Key { key: Key::Character('a'), modifiers: Modifiers::default() },
                NavigationAction::MoveLeft,
            )
            .bind(
                InputBinding::Key { key: Key::Character('d'), modifiers: Modifiers::default() },
                NavigationAction::MoveRight,
            );

        Self {
            state: Rc::new(RefCell::new(NavigationState::new(mode.clone()))),
            mode,
            bindings: Rc::new(RefCell::new(bindings)),
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
        let b = self.bindings.clone();
        let drag_start_cb = dispatcher.register(EventKind::MouseDragStart, move |event, ctx| {
            let Event::MouseDragStart { button, start_pos, .. } = event else { return false };
            let actions = b.borrow().actions_for_drag_start(*button, ctx.modifiers).to_vec();
            let mut handled = false;
            for action in actions {
                handled |= s.borrow_mut().handle_drag_start(action, *start_pos, ctx);
            }
            handled
        });

        let s = self.state.clone();
        let b = self.bindings.clone();
        let drag_cb = dispatcher.register(EventKind::MouseDrag, move |event, ctx| {
            let Event::MouseDrag { button, delta, .. } = event else { return false };
            let actions = b.borrow().actions_for_drag(*button, ctx.modifiers).to_vec();
            let size = ctx.size;
            let mut handled = false;
            ctx.with_camera_mut(|cam| {
                for action in &actions {
                    handled |= s.borrow_mut().handle_drag(*action, delta, cam, size);
                }
            });
            handled
        });

        let s = self.state.clone();
        let b = self.bindings.clone();
        let wheel_cb = dispatcher.register(EventKind::MouseWheel, move |event, ctx| {
            let Event::MouseWheel { delta } = event else { return false };
            if !b.borrow().actions_for_scroll().contains(&NavigationAction::Zoom) {
                return false;
            }
            let scroll_amount = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(_x, y) => *y / 100.0,
            };
            let model_radius =
                scene_scale::model_radius_from_bounds(ctx.scene.bounding().bounds.as_ref());
            ctx.with_camera_mut(|cam| {
                s.borrow_mut().handle_wheel(scroll_amount, cam, model_radius);
            });
            true
        });

        let s = self.state.clone();
        let b = self.bindings.clone();
        let keyboard_cb = dispatcher.register(EventKind::KeyboardInput, move |event, ctx| {
            let Event::KeyboardInput { event: key_event, .. } = event else { return false };
            let mut handled = false;
            let b_guard = b.borrow();
            ctx.with_camera_mut(|cam| {
                handled = s.borrow_mut().handle_keyboard(key_event, cam, &b_guard);
            });
            handled
        });

        let s = self.state.clone();
        let update_cb = dispatcher.register(EventKind::Update, move |event, ctx| {
            let Event::Update { delta_time } = event else { return false };
            let model_radius =
                scene_scale::model_radius_from_bounds(ctx.scene.bounding().bounds.as_ref());
            ctx.with_camera_mut(|cam| {
                s.borrow_mut().handle_update(*delta_time, cam, model_radius);
            });
            false
        });

        let s = self.state.clone();
        let b = self.bindings.clone();
        let drag_end_cb = dispatcher.register(EventKind::MouseDragEnd, move |event, _ctx| {
            let Event::MouseDragEnd { button, .. } = event else { return false };
            // Modifiers may not be held on release; match with no modifiers.
            let actions =
                b.borrow().actions_for_drag_end(*button, Modifiers::default()).to_vec();
            for action in actions {
                s.borrow_mut().handle_drag_end(action);
            }
            false
        });

        self.callback_ids =
            vec![drag_start_cb, drag_cb, drag_end_cb, wheel_cb, keyboard_cb, update_cb];
    }

    fn deactivate(&mut self, dispatcher: &mut EventDispatcher) {
        for id in &self.callback_ids {
            dispatcher.unregister(*id);
        }
        self.callback_ids.clear();
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
