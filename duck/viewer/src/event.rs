use std::collections::HashMap;
use web_time::Instant;

use crate::input::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, TouchId, TouchPhase};
use crate::scene::{NodePayload, PositionedCamera, Scene};
use crate::selection::SelectionManager;

/// Movement threshold in pixels before a mouse button hold becomes a drag.
const DRAG_THRESHOLD_PIXELS: f32 = 4.0;

/// Maximum time in milliseconds for a button press/release to be considered a click.
const CLICK_TIME_THRESHOLD_MS: u64 = 300;

/// Context passed to event callbacks, providing mutable access to application state.
pub struct EventContext<'c> {
    /// Current viewport size (width, height)
    pub size: (u32, u32),
    /// Current cursor position in screen coordinates (x, y), or None if cursor is not over the window
    pub cursor_position: &'c mut Option<(f32, f32)>,
    /// Mutable reference to the scene
    pub scene: &'c mut Scene,
    /// Mutable reference to the selection manager
    pub selection: &'c mut SelectionManager,
}

impl<'c> EventContext<'c> {
    /// Returns a [`PositionedCamera`] for the active camera node.
    ///
    /// Combines the node's world transform with its [`CameraProjection`] payload and
    /// the current viewport aspect ratio. Panics if no active camera is set.
    pub fn camera(&self) -> PositionedCamera {
        let aspect = self.size.0 as f32 / self.size.1 as f32;
        self.scene
            .active_camera_positioned(aspect)
            .expect("no active camera in scene")
    }

    /// Writes a [`PositionedCamera`] back to the active camera node.
    ///
    /// Updates both the node transform (pose) and the Camera payload (projection
    /// intrinsics + focus distance).
    pub fn set_camera(&mut self, cam: PositionedCamera) {
        let id = self.scene.active_camera().expect("no active camera in scene");
        self.scene.set_node_transform(id, cam.to_node_transform());
        self.scene.set_node_payload(id, NodePayload::Camera(cam.projection()));
    }

    /// Clones the active camera, passes it to `f` for mutation, then writes it back.
    pub fn with_camera_mut(&mut self, f: impl FnOnce(&mut PositionedCamera)) {
        let mut cam = self.camera();
        f(&mut cam);
        self.set_camera(cam);
    }
}

/// Unique identifier for a registered callback.
pub type CallbackId = u32;

/// Type alias for event callback functions.
///
/// Callbacks receive a reference to the event and mutable access to the event context.
/// They return `true` to stop event propagation or `false` to continue processing
/// additional callbacks for the same event kind.
type EventCallback = Box<dyn for<'c> Fn(&Event, &mut EventContext<'c>) -> bool>;

/// enum representing event types.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum EventKind {
    #[cfg(test)]
    /// Event kind used for testing
    Test,
    /// Per-frame update tick for continuous operations (movement, animation, etc.)
    Update,
    /// Window was resized
    Resized,
    /// Keyboard input occurred
    KeyboardInput,
    /// Mouse was moved
    MouseMotion,
    /// Cursor position changed within the window
    CursorMoved,
    /// Mouse button was pressed or released
    MouseInput,
    /// Mouse wheel was scrolled
    MouseWheel,
    /// Mouse drag started (button held and moved beyond threshold)
    MouseDragStart,
    /// Mouse drag in progress (button held and moving)
    MouseDrag,
    /// Mouse drag ended (button released after dragging)
    MouseDragEnd,
    /// Mouse click (button pressed and released quickly without dragging)
    MouseClick,
    /// Touch input (finger down, move, up, or cancel)
    Touch,
}

/// Application events with associated data.
pub enum Event {
    #[cfg(test)]
    /// Event used for testing
    Test,
    /// Per-frame update tick for continuous operations.
    ///
    /// This event should be dispatched once per frame before rendering.
    /// Operators can use this for smooth continuous movement, animations, etc.
    ///
    /// The delta_time field contains the time elapsed since the last update
    /// in seconds, which can be used for frame-rate independent movement.
    Update {
        /// Time elapsed since last update in seconds
        delta_time: f32,
    },
    /// Window was resized to the given physical size (width, height)
    Resized((u32, u32)),
    /// Keyboard input occurred
    KeyboardInput {
        /// The keyboard event details
        event: KeyEvent,
        /// If `true`, the event was generated synthetically
        is_synthetic: bool,
    },
    /// Mouse was moved (relative motion)
    MouseMotion {
        /// Delta (dx, dy) in pixels
        delta: (f64, f64),
    },
    /// Cursor position changed (absolute position)
    CursorMoved {
        /// New cursor position in physical pixels (x, y)
        position: (f64, f64),
    },
    /// Mouse button was pressed or released
    MouseInput {
        /// Whether the button was pressed or released
        state: ElementState,
        /// Which mouse button
        button: MouseButton,
    },
    /// Mouse wheel was scrolled
    MouseWheel {
        /// Scroll delta (line, pixel, or page units)
        delta: MouseScrollDelta,
    },
    /// Mouse drag started (button held and moved beyond threshold)
    MouseDragStart {
        /// Which mouse button is being dragged
        button: MouseButton,
        /// Position where the button was initially pressed (in physical pixels)
        start_pos: (f32, f32),
        /// Current cursor position (in physical pixels)
        current_pos: (f32, f32),
    },
    /// Mouse drag in progress (button held and moving)
    MouseDrag {
        /// Which mouse button is being dragged
        button: MouseButton,
        /// Position where the button was initially pressed (in physical pixels)
        start_pos: (f32, f32),
        /// Current cursor position (in physical pixels)
        current_pos: (f32, f32),
        /// Delta from last cursor position (in physical pixels)
        delta: (f32, f32),
    },
    /// Mouse drag ended (button released after dragging)
    MouseDragEnd {
        /// Which mouse button was being dragged
        button: MouseButton,
        /// Position where the button was initially pressed (in physical pixels)
        start_pos: (f32, f32),
        /// Position where the button was released (in physical pixels)
        end_pos: (f32, f32),
    },
    /// Mouse click (button pressed and released quickly without dragging)
    MouseClick {
        /// Which mouse button was clicked
        button: MouseButton,
        /// Position where the click occurred (in physical pixels)
        position: (f32, f32),
        /// Duration of the button press in milliseconds
        duration_ms: u64,
    },
    /// Touch input (finger down, move, up, or cancel)
    Touch {
        /// Unique identifier for this touch point
        id: TouchId,
        /// Phase of the touch (started, moved, ended, cancelled)
        phase: TouchPhase,
        /// Position in physical pixels
        position: (f64, f64),
    },
}

impl Event {
    /// Returns the [`EventKind`] discriminant for this event.
    pub fn kind(&self) -> EventKind {
        match self {
            Self::Update { .. } => EventKind::Update,
            Self::Resized(_) => EventKind::Resized,
            Self::KeyboardInput { .. } => EventKind::KeyboardInput,
            Self::MouseMotion { .. } => EventKind::MouseMotion,
            Self::CursorMoved { .. } => EventKind::CursorMoved,
            Self::MouseInput { .. } => EventKind::MouseInput,
            Self::MouseWheel { .. } => EventKind::MouseWheel,
            Self::MouseDragStart { .. } => EventKind::MouseDragStart,
            Self::MouseDrag { .. } => EventKind::MouseDrag,
            Self::MouseDragEnd { .. } => EventKind::MouseDragEnd,
            Self::MouseClick { .. } => EventKind::MouseClick,
            Self::Touch { .. } => EventKind::Touch,
            #[cfg(test)]
            Self::Test => EventKind::Test,
        }
    }

    // NOTE: Winit conversion functions have been removed to make the core library
    // implementation-agnostic. If you're using winit, you can create a winit_support
    // module that provides conversion functions from winit types to our input types.
}

/// Current touch interaction mode for synthesis.
#[derive(Debug, Clone)]
enum TouchMode {
    /// No active touch interaction
    None,
    /// Single finger is active
    SingleFinger { id: TouchId },
    /// Two fingers are active
    TwoFinger { ids: [TouchId; 2] },
}

/// State for synthesizing mouse events from touch input.
#[derive(Debug, Clone)]
struct TouchSynthState {
    /// Active touch points: id → last known position
    active_touches: HashMap<TouchId, (f64, f64)>,
    /// Current interaction mode
    mode: TouchMode,
    /// Previous pinch distance for delta computation
    prev_pinch_distance: Option<f64>,
}

impl TouchSynthState {
    fn new() -> Self {
        Self {
            active_touches: HashMap::new(),
            mode: TouchMode::None,
            prev_pinch_distance: None,
        }
    }
}

/// State tracking for a mouse button that is currently pressed.
#[derive(Debug, Clone)]
struct ButtonState {
    /// Position where the button was initially pressed (in physical pixels)
    down_position: (f32, f32),
    /// Time when the button was pressed
    down_time: Instant,
    /// Whether this button press has transitioned into a drag
    is_dragging: bool,
    /// Total distance dragged since button down (in pixels)
    distance_dragged: f32,
}

/// Event dispatcher that manages callbacks for different event types.
///
/// Callbacks are registered by [`EventKind`] and invoked when matching events
/// are dispatched. Multiple callbacks can be registered for the same event kind,
/// and they are called until one returns `true`.
///
/// Each callback is assigned a unique [`CallbackId`] when registered.
///
/// The dispatcher also tracks mouse button state to synthesize high-level events
/// like [`EventKind::MouseDragStart`], [`EventKind::MouseDrag`], [`EventKind::MouseDragEnd`],
/// and [`EventKind::MouseClick`].
pub struct EventDispatcher {
    callback_map: HashMap<EventKind, Vec<(CallbackId, EventCallback)>>,
    next_id: u32,
    /// State for each currently-pressed mouse button
    button_states: HashMap<MouseButton, ButtonState>,
    /// Current cursor position in physical pixels (from CursorMoved events)
    current_cursor_position: Option<(f32, f32)>,
    /// State for synthesizing mouse events from touch input
    touch_state: TouchSynthState,
}

impl EventDispatcher {
    /// Creates a new empty event dispatcher.
    pub(crate) fn new() -> Self {
        Self {
            callback_map: HashMap::new(),
            next_id: 0,
            button_states: HashMap::new(),
            current_cursor_position: None,
            touch_state: TouchSynthState::new(),
        }
    }

    /// Registers a callback for a specific event kind.
    ///
    /// Multiple callbacks can be registered for the same event kind. They will
    /// be called in order when an event of that kind is dispatched.
    ///
    /// Returns a [`CallbackId`] that can be used to unregister or reorder this callback.
    pub fn register<F>(&mut self, kind: EventKind, callback: F) -> CallbackId
    where
        F: for<'c> Fn(&Event, &mut EventContext<'c>) -> bool + 'static
    {
        let id = self.next_id;
        self.next_id += 1;

        self.callback_map
            .entry(kind)
            .or_default()
            .push((id, Box::new(callback)));

        id
    }

    /// Unregisters a callback by its ID.
    ///
    /// Find and remove the callback with the given ID.
    /// Returns `true` if the callback was found and removed, `false` otherwise.
    pub fn unregister(&mut self, id: CallbackId) -> bool {
        for callbacks in self.callback_map.values_mut() {
            if let Some(pos) = callbacks.iter().position(|(cid, _)| *cid == id) {
                let _ = callbacks.remove(pos);
                return true;
            }
        }
        false
    }

    /// Reorders callbacks for a specific event kind.
    ///
    /// The callbacks will be reordered according to the provided slice of IDs.
    /// Any callbacks not mentioned in the IDs slice will remain at the end in their current order.
    /// IDs that don't exist are ignored.
    ///
    /// Returns `true` if the event kind exists, `false` otherwise.
    pub fn reorder_kind(&mut self, kind: EventKind, ids: &[CallbackId]) -> bool {
        if let Some(callbacks) = self.callback_map.get_mut(&kind) {
            let mut reordered = Vec::new();

            // Add callbacks in the order specified by ids
            for &id in ids {
                if let Some(pos) = callbacks.iter().position(|(cid, _)| *cid == id) {
                    reordered.push(callbacks.remove(pos));
                }
            }

            // Append any remaining callbacks that weren't in the ids list
            reordered.append(callbacks);

            *callbacks = reordered;
            true
        } else {
            false
        }
    }

    /// Reorders callbacks for all registered event kinds.
    ///
    /// This applies the same reordering to all event kinds that have callbacks registered.
    /// Any callbacks not mentioned in the IDs slice will remain at the end in their current order.
    pub fn reorder(&mut self, ids: &[CallbackId]) {
        // Collect event kinds first to avoid borrowing issues
        let kinds: Vec<EventKind> = self.callback_map.keys().copied().collect();
        for kind in kinds {
            self.reorder_kind(kind, ids);
        }
    }

    /// Dispatches an event to all registered callbacks for its kind.
    ///
    /// This method also handles stateful tracking of mouse interactions to synthesize
    /// high-level events like MouseDragStart, MouseDrag, MouseDragEnd, and MouseClick.
    /// These synthesized events are dispatched before the original event.
    ///
    /// Callbacks are invoked in registration order. If a callback returns `true`,
    /// no further callbacks are invoked (propagation is stopped).
    pub fn dispatch<'c>(&mut self, event: &Event, ctx: &mut EventContext<'c>) -> bool {
        // First, update state and synthesize high-level events based on the incoming event
        self.process_event(event, ctx);

        // Then dispatch the original event
        self.dispatch_to_callbacks(event, ctx)
    }

    /// Internal method to dispatch an event to callbacks without state processing.
    fn dispatch_to_callbacks<'c>(&self, event: &Event, ctx: &mut EventContext<'c>) -> bool {
        if let Some(callbacks) = self.callback_map.get(&event.kind()) {
            for (_id, callback) in callbacks {
                if callback(event, ctx) {
                    return true;
                }
            }
        }
        false
    }

    /// Processes an event to update internal state and synthesize high-level events.
    fn process_event<'c>(&mut self, event: &Event, ctx: &mut EventContext<'c>) {
        match event {
            Event::CursorMoved { position } => {
                self.process_cursor_moved(*position);
            }
            Event::MouseInput { state, button } => {
                self.process_mouse_input(*state, *button, ctx);
            }
            Event::MouseMotion { delta } => {
                self.process_mouse_motion(*delta, ctx);
            }
            Event::Touch { id, phase, position } => {
                self.process_touch(*id, *phase, *position, ctx);
            }
            _ => {
                // No state processing needed for other events
            }
        }
    }

    /// Processes CursorMoved events to update cursor position tracking.
    fn process_cursor_moved(&mut self, position: (f64, f64)) {
        self.current_cursor_position = Some((position.0 as f32, position.1 as f32));
    }

    /// Processes MouseInput events to track button press/release and synthesize click events.
    fn process_mouse_input<'c>(
        &mut self,
        state: ElementState,
        button: MouseButton,
        ctx: &mut EventContext<'c>,
    ) {
        match state {
            ElementState::Pressed => {
                // Button pressed - start tracking
                if let Some(pos) = self.current_cursor_position {
                    self.button_states.insert(
                        button,
                        ButtonState {
                            down_position: pos,
                            down_time: Instant::now(),
                            is_dragging: false,
                            distance_dragged: 0.0,
                        },
                    );
                }
            }
            ElementState::Released => {
                // Button released - check if it was a click or drag
                if let Some(button_state) = self.button_states.remove(&button) {
                    if button_state.is_dragging {
                        self.synthesize_drag_end(button, button_state, ctx);
                    } else {
                        self.synthesize_click_if_quick(button, button_state, ctx);
                    }
                }
            }
        }
    }

    /// Synthesizes a MouseDragEnd event when a dragged button is released.
    fn synthesize_drag_end<'c>(
        &self,
        button: MouseButton,
        button_state: ButtonState,
        ctx: &mut EventContext<'c>,
    ) {
        if let Some(end_pos) = self.current_cursor_position {
            let drag_end = Event::MouseDragEnd {
                button,
                start_pos: button_state.down_position,
                end_pos,
            };
            self.dispatch_to_callbacks(&drag_end, ctx);
        }
    }

    /// Synthesizes a MouseClick event if the button was released quickly.
    fn synthesize_click_if_quick<'c>(
        &self,
        button: MouseButton,
        button_state: ButtonState,
        ctx: &mut EventContext<'c>,
    ) {
        let duration = button_state.down_time.elapsed();
        if duration.as_millis() as u64 <= CLICK_TIME_THRESHOLD_MS {
            let click = Event::MouseClick {
                button,
                position: button_state.down_position,
                duration_ms: duration.as_millis() as u64,
            };
            self.dispatch_to_callbacks(&click, ctx);
        }
    }

    /// Processes a touch event and synthesizes corresponding mouse events.
    ///
    /// Touch gestures are mapped to mouse events so existing operators work unchanged:
    /// - Single-finger drag → left mouse drag (orbit)
    /// - Two-finger drag → right mouse drag (pan)
    /// - Two-finger pinch → mouse wheel (zoom)
    /// - Single tap → left mouse click (selection)
    fn process_touch<'c>(
        &mut self,
        id: TouchId,
        phase: TouchPhase,
        position: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        match phase {
            TouchPhase::Started => self.process_touch_started(id, position, ctx),
            TouchPhase::Moved => self.process_touch_moved(id, position, ctx),
            TouchPhase::Ended | TouchPhase::Cancelled => self.process_touch_ended(id, ctx),
        }
    }

    /// Handles a new finger touching the screen.
    fn process_touch_started<'c>(
        &mut self,
        id: TouchId,
        position: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        self.touch_state.active_touches.insert(id, position);

        match &self.touch_state.mode {
            TouchMode::None => {
                // First finger: start single-finger mode (maps to left mouse)
                self.touch_state.mode = TouchMode::SingleFinger { id };
                self.dispatch_synthetic_mouse(
                    &Event::CursorMoved { position },
                    ctx,
                );
                self.dispatch_synthetic_mouse(
                    &Event::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                    },
                    ctx,
                );
            }
            TouchMode::SingleFinger { id: first_id } => {
                // Second finger: transition to two-finger mode
                let first_id = *first_id;

                // Release the left button to cancel any single-finger drag
                self.dispatch_synthetic_mouse(
                    &Event::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Left,
                    },
                    ctx,
                );

                // Calculate initial pinch distance and center
                if let (Some(&first_pos), Some(&second_pos)) = (
                    self.touch_state.active_touches.get(&first_id),
                    self.touch_state.active_touches.get(&id),
                ) {
                    let center = touch_center(first_pos, second_pos);
                    let distance = touch_distance(first_pos, second_pos);
                    self.touch_state.prev_pinch_distance = Some(distance);
                    self.touch_state.mode = TouchMode::TwoFinger {
                        ids: [first_id, id],
                    };

                    // Start right-button press at center for pan
                    self.dispatch_synthetic_mouse(
                        &Event::CursorMoved { position: center },
                        ctx,
                    );
                    self.dispatch_synthetic_mouse(
                        &Event::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Right,
                        },
                        ctx,
                    );
                }
            }
            TouchMode::TwoFinger { .. } => {
                // Third+ finger: ignore
            }
        }
    }

    /// Handles a finger moving on the screen.
    fn process_touch_moved<'c>(
        &mut self,
        id: TouchId,
        position: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        let old_position = self.touch_state.active_touches.get(&id).copied();
        self.touch_state.active_touches.insert(id, position);

        match &self.touch_state.mode {
            TouchMode::None => {}
            TouchMode::SingleFinger { id: finger_id } => {
                if id != *finger_id {
                    return;
                }
                if let Some(old_pos) = old_position {
                    let delta = (position.0 - old_pos.0, position.1 - old_pos.1);
                    self.dispatch_synthetic_mouse(
                        &Event::CursorMoved { position },
                        ctx,
                    );
                    self.dispatch_synthetic_mouse(
                        &Event::MouseMotion { delta },
                        ctx,
                    );
                }
            }
            TouchMode::TwoFinger { ids } => {
                let [id0, id1] = *ids;
                if id != id0 && id != id1 {
                    return;
                }
                if let (Some(&pos0), Some(&pos1)) = (
                    self.touch_state.active_touches.get(&id0),
                    self.touch_state.active_touches.get(&id1),
                ) {
                    let center = touch_center(pos0, pos1);
                    let distance = touch_distance(pos0, pos1);

                    // Emit pan as right-drag (center movement)
                    if let Some(old_pos) = old_position {
                        // Compute how the center moved due to this finger's movement
                        let half_delta = (
                            (position.0 - old_pos.0) * 0.5,
                            (position.1 - old_pos.1) * 0.5,
                        );
                        self.dispatch_synthetic_mouse(
                            &Event::CursorMoved { position: center },
                            ctx,
                        );
                        self.dispatch_synthetic_mouse(
                            &Event::MouseMotion { delta: half_delta },
                            ctx,
                        );
                    }

                    // Emit zoom as mouse wheel (pinch distance change)
                    if let Some(prev_distance) = self.touch_state.prev_pinch_distance
                        && prev_distance > 0.0 {
                            let delta = distance - prev_distance;
                            // Scale: positive delta = fingers moving apart = zoom in
                            if delta.abs() > 0.5 {
                                self.dispatch_synthetic_mouse(
                                    &Event::MouseWheel {
                                        delta: MouseScrollDelta::PixelDelta(0.0, delta as f32),
                                    },
                                    ctx,
                                );
                            }
                        }
                    self.touch_state.prev_pinch_distance = Some(distance);
                }
            }
        }
    }

    /// Handles a finger leaving the screen.
    fn process_touch_ended<'c>(
        &mut self,
        id: TouchId,
        ctx: &mut EventContext<'c>,
    ) {
        self.touch_state.active_touches.remove(&id);

        match &self.touch_state.mode {
            TouchMode::None => {}
            TouchMode::SingleFinger { id: finger_id } => {
                if id != *finger_id {
                    return;
                }
                // Release left button
                self.dispatch_synthetic_mouse(
                    &Event::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Left,
                    },
                    ctx,
                );
                self.touch_state.mode = TouchMode::None;
            }
            TouchMode::TwoFinger { ids } => {
                let [id0, id1] = *ids;
                if id != id0 && id != id1 {
                    return;
                }

                // Release right button (end pan)
                self.dispatch_synthetic_mouse(
                    &Event::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Right,
                    },
                    ctx,
                );
                self.touch_state.prev_pinch_distance = None;

                // Find remaining finger and transition to single-finger mode
                let remaining_id = if id == id0 { id1 } else { id0 };
                if let Some(&remaining_pos) = self.touch_state.active_touches.get(&remaining_id) {
                    self.touch_state.mode = TouchMode::SingleFinger { id: remaining_id };
                    self.dispatch_synthetic_mouse(
                        &Event::CursorMoved {
                            position: remaining_pos,
                        },
                        ctx,
                    );
                    self.dispatch_synthetic_mouse(
                        &Event::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Left,
                        },
                        ctx,
                    );
                } else {
                    self.touch_state.mode = TouchMode::None;
                }
            }
        }
    }

    /// Dispatches a synthesized mouse event through the full processing pipeline.
    ///
    /// This calls `process_event` so the synthesized mouse event updates button/cursor
    /// state and triggers further synthesis (e.g., drag and click events).
    fn dispatch_synthetic_mouse<'c>(
        &mut self,
        event: &Event,
        ctx: &mut EventContext<'c>,
    ) {
        self.process_event(event, ctx);
        self.dispatch_to_callbacks(event, ctx);
    }

    /// Processes MouseMotion events to track drag distance and synthesize drag events.
    fn process_mouse_motion<'c>(
        &mut self,
        delta: (f64, f64),
        ctx: &mut EventContext<'c>,
    ) {
        let delta_magnitude = ((delta.0 * delta.0 + delta.1 * delta.1) as f32).sqrt();

        // Collect buttons that need drag events (to avoid borrow issues)
        let mut drag_events = Vec::new();

        for (button, button_state) in &mut self.button_states {
            button_state.distance_dragged += delta_magnitude;

            if !button_state.is_dragging && button_state.distance_dragged > DRAG_THRESHOLD_PIXELS {
                // Transition to dragging state
                button_state.is_dragging = true;
                if let Some(current_pos) = self.current_cursor_position {
                    drag_events.push(Event::MouseDragStart {
                        button: *button,
                        start_pos: button_state.down_position,
                        current_pos,
                    });
                }
            } else if button_state.is_dragging {
                // Already dragging - queue MouseDrag event
                if let Some(current_pos) = self.current_cursor_position {
                    drag_events.push(Event::MouseDrag {
                        button: *button,
                        start_pos: button_state.down_position,
                        current_pos,
                        delta: (delta.0 as f32, delta.1 as f32),
                    });
                }
            }
        }

        // Dispatch all queued drag events
        for drag_event in drag_events {
            self.dispatch_to_callbacks(&drag_event, ctx);
        }
    }
}

/// Compute the center point between two touch positions.
fn touch_center(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

/// Compute the Euclidean distance between two touch positions.
fn touch_distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{ElementState, MouseButton, MouseScrollDelta};
    use std::rc::Rc;
    use std::cell::Cell;

    // ===== Helper Functions =====

    /// Creates a mock EventContext for testing.
    ///
    /// Uses real stack-allocated values so the context fields are valid.
    /// Test callbacks that don't access context fields can safely ignore them.
    fn create_mock_context_parts() -> (Option<(f32, f32)>, Scene, SelectionManager) {
        use crate::scene::PositionedCamera;
        let camera = PositionedCamera {
            eye: (0.0, 0.0, 1.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        };
        let mut scene = Scene::new();
        let cam_id = scene.add_node(None, None, camera.to_node_transform()).unwrap();
        scene.set_node_payload(cam_id, NodePayload::Camera(camera.projection()));
        scene.set_active_camera(Some(cam_id));
        (None, scene, SelectionManager::new())
    }

    fn make_context(parts: &mut (Option<(f32, f32)>, Scene, SelectionManager)) -> EventContext<'_> {
        EventContext {
            size: (800, 600),
            cursor_position: &mut parts.0,
            scene: &mut parts.1,
            selection: &mut parts.2,
        }
    }

    // ===== EventDispatcher Tests =====

    #[test]
    fn test_dispatcher_new() {
        let dispatcher = EventDispatcher::new();
        assert_eq!(dispatcher.callback_map.len(), 0);
        assert_eq!(dispatcher.next_id, 0);
    }

    #[test]
    fn test_dispatcher_register() {
        let mut dispatcher = EventDispatcher::new();

        let id = dispatcher.register(EventKind::Test, |_event, _ctx| false);

        assert_eq!(id, 0);
        assert!(dispatcher.callback_map.contains_key(&EventKind::Test));
        assert_eq!(dispatcher.callback_map[&EventKind::Test].len(), 1);
    }

    #[test]
    fn test_dispatcher_register_multiple_same_kind() {
        let mut dispatcher = EventDispatcher::new();

        let id1 = dispatcher.register(EventKind::KeyboardInput, |_event, _ctx| false);
        let id2 = dispatcher.register(EventKind::KeyboardInput, |_event, _ctx| false);
        let id3 = dispatcher.register(EventKind::KeyboardInput, |_event, _ctx| false);

        // All IDs should be unique
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);

        // All callbacks should be registered for the same event kind
        assert_eq!(dispatcher.callback_map[&EventKind::KeyboardInput].len(), 3);
    }

    #[test]
    fn test_dispatcher_register_different_kinds() {
        let mut dispatcher = EventDispatcher::new();

        let id1 = dispatcher.register(EventKind::KeyboardInput, |_event, _ctx| false);
        let id2 = dispatcher.register(EventKind::MouseInput, |_event, _ctx| false);
        let id3 = dispatcher.register(EventKind::Test, |_event, _ctx| false);

        // All IDs should be unique
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);

        // Each event kind should have one callback
        assert_eq!(dispatcher.callback_map.len(), 3);
        assert_eq!(dispatcher.callback_map[&EventKind::KeyboardInput].len(), 1);
        assert_eq!(dispatcher.callback_map[&EventKind::MouseInput].len(), 1);
        assert_eq!(dispatcher.callback_map[&EventKind::Test].len(), 1);
    }

    #[test]
    fn test_dispatcher_unregister() {
        let mut dispatcher = EventDispatcher::new();

        let id = dispatcher.register(EventKind::MouseMotion, |_event, _ctx| false);

        // Callback should be registered
        assert_eq!(dispatcher.callback_map[&EventKind::MouseMotion].len(), 1);

        // Unregister should succeed
        let result = dispatcher.unregister(id);
        assert!(result);

        // Callback should be removed
        assert_eq!(dispatcher.callback_map[&EventKind::MouseMotion].len(), 0);
    }

    #[test]
    fn test_dispatcher_unregister_nonexistent() {
        let mut dispatcher = EventDispatcher::new();

        // Try to unregister an ID that was never registered
        let result = dispatcher.unregister(999);
        assert!(!result);
    }

    #[test]
    fn test_dispatcher_dispatch_no_callbacks() {
        let mut dispatcher = EventDispatcher::new();
        let event = Event::Test;

        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);

        // Should not panic and should return false
        let result = dispatcher.dispatch(&event, &mut ctx);
        assert!(!result);
    }

    #[test]
    fn test_dispatcher_dispatch_single_callback() {
        let mut dispatcher = EventDispatcher::new();
        let counter = Rc::new(Cell::new(0));
        let counter_clone = Rc::clone(&counter);

        dispatcher.register(EventKind::Test, move |_event, _ctx| {
            counter_clone.set(counter_clone.get() + 1);
            false
        });

        let event = Event::Test;
        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);

        dispatcher.dispatch(&event, &mut ctx);

        // Callback should have been invoked once
        assert_eq!(counter.get(), 1);
    }

    #[test]
    fn test_dispatcher_dispatch_multiple_callbacks() {
        let mut dispatcher = EventDispatcher::new();
        let counter = Rc::new(Cell::new(0));

        let c1 = Rc::clone(&counter);
        dispatcher.register(EventKind::MouseInput, move |_event, _ctx| {
            c1.set(c1.get() + 1);
            false
        });

        let c2 = Rc::clone(&counter);
        dispatcher.register(EventKind::MouseInput, move |_event, _ctx| {
            c2.set(c2.get() + 10);
            false
        });

        let c3 = Rc::clone(&counter);
        dispatcher.register(EventKind::MouseInput, move |_event, _ctx| {
            c3.set(c3.get() + 100);
            false
        });

        let event = Event::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
        };
        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);

        dispatcher.dispatch(&event, &mut ctx);

        // All three callbacks should have been invoked
        assert_eq!(counter.get(), 111);
    }

    #[test]
    fn test_dispatcher_dispatch_stop_propagation() {
        let mut dispatcher = EventDispatcher::new();
        let counter = Rc::new(Cell::new(0));

        let c1 = Rc::clone(&counter);
        dispatcher.register(EventKind::CursorMoved, move |_event, _ctx| {
            c1.set(c1.get() + 1);
            false // Continue
        });

        let c2 = Rc::clone(&counter);
        dispatcher.register(EventKind::CursorMoved, move |_event, _ctx| {
            c2.set(c2.get() + 10);
            true // Stop propagation
        });

        let c3 = Rc::clone(&counter);
        dispatcher.register(EventKind::CursorMoved, move |_event, _ctx| {
            c3.set(c3.get() + 100);
            false // This should never run
        });

        let event = Event::CursorMoved {
            position: (100.0, 200.0),
        };
        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);

        let result = dispatcher.dispatch(&event, &mut ctx);

        // First two callbacks ran, third did not
        assert_eq!(counter.get(), 11);
        assert!(result); // dispatch should return true when propagation stopped
    }

    #[test]
    fn test_dispatcher_dispatch_continue_propagation() {
        let mut dispatcher = EventDispatcher::new();
        let counter = Rc::new(Cell::new(0));

        let c1 = Rc::clone(&counter);
        dispatcher.register(EventKind::MouseWheel, move |_event, _ctx| {
            c1.set(c1.get() + 1);
            false // Continue
        });

        let c2 = Rc::clone(&counter);
        dispatcher.register(EventKind::MouseWheel, move |_event, _ctx| {
            c2.set(c2.get() + 10);
            false // Continue
        });

        let event = Event::MouseWheel {
            delta: MouseScrollDelta::LineDelta(0.0, 1.0),
        };
        let mut parts = create_mock_context_parts();
        let mut ctx = make_context(&mut parts);

        let result = dispatcher.dispatch(&event, &mut ctx);

        // Both callbacks should have run
        assert_eq!(counter.get(), 11);
        assert!(!result); // dispatch should return false when no callback stopped propagation
    }

    #[test]
    fn test_dispatcher_reorder_kind() {
        let mut dispatcher = EventDispatcher::new();

        let id1 = dispatcher.register(EventKind::Resized, |_event, _ctx| false);
        let id2 = dispatcher.register(EventKind::Resized, |_event, _ctx| false);
        let id3 = dispatcher.register(EventKind::Resized, |_event, _ctx| false);

        // Original order: id1, id2, id3
        let callbacks = &dispatcher.callback_map[&EventKind::Resized];
        assert_eq!(callbacks[0].0, id1);
        assert_eq!(callbacks[1].0, id2);
        assert_eq!(callbacks[2].0, id3);

        // Reorder to: id3, id1, id2
        let result = dispatcher.reorder_kind(EventKind::Resized, &[id3, id1, id2]);
        assert!(result);

        let callbacks = &dispatcher.callback_map[&EventKind::Resized];
        assert_eq!(callbacks[0].0, id3);
        assert_eq!(callbacks[1].0, id1);
        assert_eq!(callbacks[2].0, id2);
    }

    #[test]
    fn test_dispatcher_reorder_all() {
        let mut dispatcher = EventDispatcher::new();

        // Register callbacks for multiple event kinds
        let id1 = dispatcher.register(EventKind::MouseInput, |_event, _ctx| false);
        let id2 = dispatcher.register(EventKind::MouseInput, |_event, _ctx| false);
        let id3 = dispatcher.register(EventKind::KeyboardInput, |_event, _ctx| false);
        let id4 = dispatcher.register(EventKind::KeyboardInput, |_event, _ctx| false);

        // Reorder all to: id2, id4, id1, id3
        dispatcher.reorder(&[id2, id4, id1, id3]);

        // Check MouseInput order
        let mouse_callbacks = &dispatcher.callback_map[&EventKind::MouseInput];
        assert_eq!(mouse_callbacks[0].0, id2);
        assert_eq!(mouse_callbacks[1].0, id1);

        // Check KeyboardInput order
        let key_callbacks = &dispatcher.callback_map[&EventKind::KeyboardInput];
        assert_eq!(key_callbacks[0].0, id4);
        assert_eq!(key_callbacks[1].0, id3);
    }

    #[test]
    fn test_dispatcher_reorder_partial_ids() {
        let mut dispatcher = EventDispatcher::new();

        let id1 = dispatcher.register(EventKind::CursorMoved, |_event, _ctx| false);
        let id2 = dispatcher.register(EventKind::CursorMoved, |_event, _ctx| false);
        let id3 = dispatcher.register(EventKind::CursorMoved, |_event, _ctx| false);

        // Only reorder id2 - id1 and id3 should remain at end in original order
        let result = dispatcher.reorder_kind(EventKind::CursorMoved, &[id2]);
        assert!(result);

        let callbacks = &dispatcher.callback_map[&EventKind::CursorMoved];
        assert_eq!(callbacks[0].0, id2); // Specified first
        assert_eq!(callbacks[1].0, id1); // Remaining in original order
        assert_eq!(callbacks[2].0, id3); // Remaining in original order
    }

    #[test]
    fn test_dispatcher_reorder_nonexistent_ids() {
        let mut dispatcher = EventDispatcher::new();

        let id1 = dispatcher.register(EventKind::Test, |_event, _ctx| false);
        let id2 = dispatcher.register(EventKind::Test, |_event, _ctx| false);

        // Try to reorder with some nonexistent IDs
        let result = dispatcher.reorder_kind(EventKind::Test, &[999, id2, 888, id1, 777]);
        assert!(result);

        // Only the valid IDs should be reordered
        let callbacks = &dispatcher.callback_map[&EventKind::Test];
        assert_eq!(callbacks[0].0, id2);
        assert_eq!(callbacks[1].0, id1);
    }
}