use std::collections::HashMap;
use web_time::Instant;

use crate::annotation::AnnotationManager;
use crate::drawstate::Renderer;
use crate::scene::Scene;
use crate::input::{ElementState, MouseButton, MouseScrollDelta, KeyEvent};

/// Movement threshold in pixels before a mouse button hold becomes a drag.
const DRAG_THRESHOLD_PIXELS: f32 = 4.0;

/// Maximum time in milliseconds for a button press/release to be considered a click.
const CLICK_TIME_THRESHOLD_MS: u64 = 300;

/// Context passed to event callbacks, providing mutable access to application state.
///
/// This struct bundles all the mutable state that event callbacks need to access,
/// including the rendering state, scene, and annotation manager.
///
/// ## Lifetime Parameters
/// - `'w`: The surface lifetime - DrawState holds a reference to a rendering surface with this lifetime
/// - `'c`: The callback lifetime - represents the duration of a single event callback invocation
pub struct EventContext<'w, 'c> {
    /// Mutable reference to the rendering state
    pub(crate) state: &'c mut Renderer<'w>,
    /// Mutable reference to the scene
    pub scene: &'c mut Scene,
    /// Mutable reference to the annotation manager
    pub annotation_manager: &'c mut AnnotationManager,
}

/// Unique identifier for a registered callback.
pub type CallbackId = u32;

/// Type alias for event callback functions.
///
/// Callbacks receive a reference to the event and mutable access to the event context.
/// They return `true` to stop event propagation or `false` to continue processing
/// additional callbacks for the same event kind.
type EventCallback = Box<dyn for<'w, 'c> Fn(&Event, &mut EventContext<'w, 'c>) -> bool>;

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
            #[cfg(test)]
            Self::Test => EventKind::Test,
        }
    }

    // NOTE: Winit conversion functions have been removed to make the core library
    // implementation-agnostic. If you're using winit, you can create a winit_support
    // module that provides conversion functions from winit types to our input types.
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
}

impl EventDispatcher {
    /// Creates a new empty event dispatcher.
    pub(crate) fn new() -> Self {
        Self {
            callback_map: HashMap::new(),
            next_id: 0,
            button_states: HashMap::new(),
            current_cursor_position: None,
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
        F: for<'w, 'c> Fn(&Event, &mut EventContext<'w, 'c>) -> bool + 'static
    {
        let id = self.next_id;
        self.next_id += 1;

        self.callback_map
            .entry(kind)
            .or_insert_with(Vec::new)
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
    pub fn dispatch<'w, 'c>(&mut self, event: &Event, ctx: &mut EventContext<'w, 'c>) -> bool {
        // First, update state and synthesize high-level events based on the incoming event
        self.process_event(event, ctx);

        // Then dispatch the original event
        self.dispatch_to_callbacks(event, ctx)
    }

    /// Internal method to dispatch an event to callbacks without state processing.
    fn dispatch_to_callbacks<'w, 'c>(&self, event: &Event, ctx: &mut EventContext<'w, 'c>) -> bool {
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
    fn process_event<'w, 'c>(&mut self, event: &Event, ctx: &mut EventContext<'w, 'c>) {
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
    fn process_mouse_input<'w, 'c>(
        &mut self,
        state: ElementState,
        button: MouseButton,
        ctx: &mut EventContext<'w, 'c>,
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
    fn synthesize_drag_end<'w, 'c>(
        &self,
        button: MouseButton,
        button_state: ButtonState,
        ctx: &mut EventContext<'w, 'c>,
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
    fn synthesize_click_if_quick<'w, 'c>(
        &self,
        button: MouseButton,
        button_state: ButtonState,
        ctx: &mut EventContext<'w, 'c>,
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

    /// Processes MouseMotion events to track drag distance and synthesize drag events.
    fn process_mouse_motion<'w, 'c>(
        &mut self,
        delta: (f64, f64),
        ctx: &mut EventContext<'w, 'c>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{ElementState, MouseButton, MouseScrollDelta};
    use std::rc::Rc;
    use std::cell::Cell;

    // ===== Helper Functions =====

    /// Creates a mock EventContext for testing.
    ///
    /// SAFETY: This creates dangling references that should NEVER be dereferenced.
    /// It's only safe to use in test callbacks that don't actually access the context fields.
    /// We use non-null pointers to avoid triggering the zero-initialization check.
    #[allow(invalid_reference_casting)]
    unsafe fn create_mock_context<'w, 'c>() -> EventContext<'w, 'c> {
        let dangling = std::ptr::NonNull::dangling();
        EventContext {
            state: unsafe { &mut *(dangling.as_ptr() as *mut Renderer) },
            scene: unsafe { &mut *(dangling.as_ptr() as *mut Scene) },
            annotation_manager: unsafe { &mut *(dangling.as_ptr() as *mut AnnotationManager) },
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

        // SAFETY: We're creating a mock context that won't be used
        let mut ctx = unsafe { create_mock_context() };

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
        let mut ctx = unsafe { create_mock_context() };

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
        let mut ctx = unsafe { create_mock_context() };

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
        let mut ctx = unsafe { create_mock_context() };

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
        let mut ctx = unsafe { create_mock_context() };

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