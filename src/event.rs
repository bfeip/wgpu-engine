use std::collections::HashMap;

use crate::drawstate::DrawState;
use crate::scene::Scene;

/// Context passed to event callbacks, providing mutable access to application state.
///
/// This struct bundles all the mutable state that event callbacks need to access,
/// including the rendering state, scene, and event loop control flow.
///
/// ## Lifetime Parameters
/// - `'w`: The window lifetime - DrawState holds a reference to the Window with this lifetime
/// - `'c`: The callback lifetime - represents the duration of a single event callback invocation
pub struct EventContext<'w, 'c> {
    /// Mutable reference to the rendering state
    pub state: &'c mut DrawState<'w>,
    /// Mutable reference to the scene
    pub scene: &'c mut Scene,
    /// Reference to the event loop control flow (for exiting the application, etc.)
    pub control_flow: &'c winit::event_loop::EventLoopWindowTarget<()>,
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
    /// Window close was requested
    CloseRequested,
    /// Window was resized
    Resized,
    /// Window needs to be redrawn
    RedrawRequested,
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
}

/// Application events with associated data, converted from winit events.
pub enum Event {
    /// Window close was requested
    CloseRequested,
    /// Window was resized to the given physical size
    Resized(winit::dpi::PhysicalSize<u32>),
    /// Window needs to be redrawn
    RedrawRequested,
    /// Keyboard input occurred
    KeyboardInput {
        /// The keyboard event details
        event: winit::event::KeyEvent,
        /// If `true`, the event was generated synthetically by winit (see their docs)
        is_synthetic: bool,
    },
    /// Mouse was moved (relative motion)
    MouseMotion {
        /// Delta (dx, dy) in pixels
        delta: (f64, f64),
    },
    /// Cursor position changed (absolute position)
    CursorMoved {
        /// New cursor position in physical pixels
        position: winit::dpi::PhysicalPosition<f64>,
    },
    /// Mouse button was pressed or released
    MouseInput {
        /// Whether the button was pressed or released
        state: winit::event::ElementState,
        /// Which mouse button
        button: winit::event::MouseButton,
    },
    /// Mouse wheel was scrolled
    MouseWheel {
        /// Scroll delta (line, pixel, or page units)
        delta: winit::event::MouseScrollDelta,
    },
}

impl Event {
    /// Returns the [`EventKind`] discriminant for this event.
    pub fn kind(&self) -> EventKind {
        match self {
            Self::CloseRequested => EventKind::CloseRequested,
            Self::Resized(_) => EventKind::Resized,
            Self::RedrawRequested => EventKind::RedrawRequested,
            Self::KeyboardInput { .. } => EventKind::KeyboardInput,
            Self::MouseMotion { .. } => EventKind::MouseMotion,
            Self::CursorMoved { .. } => EventKind::CursorMoved,
            Self::MouseInput { .. } => EventKind::MouseInput,
            Self::MouseWheel { .. } => EventKind::MouseWheel,
        }
    }

    /// Converts a winit event to an application event.
    ///
    /// Returns `None` if the event is not supported or should be ignored.
    pub fn from_winit_event(wevent: winit::event::Event<()>) -> Option<Self> {
        use winit::event::Event as WEvent;

        match wevent {
            WEvent::WindowEvent { event, .. } => Self::from_winit_window_event(event),
            WEvent::DeviceEvent { event, .. } => Self::from_winit_device_event(event),
            _ => None,
        }
    }

    /// Converts a winit window event to an application event.
    ///
    /// Returns `None` if the event is not supported or should be ignored.
    pub fn from_winit_window_event(wevent: winit::event::WindowEvent) -> Option<Self> {
        use winit::event::WindowEvent as WEvent;

        match wevent {
            WEvent::CloseRequested => Some(Self::CloseRequested),
            WEvent::Resized(size) => Some(Self::Resized(size)),
            WEvent::RedrawRequested => Some(Self::RedrawRequested),
            WEvent::KeyboardInput { event, is_synthetic, .. } => Some(Self::KeyboardInput {
                event,
                is_synthetic,
            }),
            WEvent::CursorMoved { position, .. } => Some(Self::CursorMoved { position }),
            WEvent::MouseInput { state, button, .. } => Some(Self::MouseInput { state, button }),
            WEvent::MouseWheel { delta, .. } => Some(Self::MouseWheel { delta }),
            _ => None,
        }
    }

    /// Converts a winit device event to an application event.
    ///
    /// Returns `None` if the event is not supported or should be ignored.
    pub fn from_winit_device_event(wevent: winit::event::DeviceEvent) -> Option<Self> {
        use winit::event::DeviceEvent as DEvent;

        match wevent {
            DEvent::MouseMotion { delta } => Some(Self::MouseMotion { delta }),
            _ => None,
        }
    }
}

/// Event dispatcher that manages callbacks for different event types.
///
/// Callbacks are registered by [`EventKind`] and invoked when matching events
/// are dispatched. Multiple callbacks can be registered for the same event kind,
/// and they are called until one returns `true`.
///
/// Each callback is assigned a unique [`CallbackId`] when registered.
pub struct EventDispatcher {
    callback_map: HashMap<EventKind, Vec<(CallbackId, EventCallback)>>,
    next_id: u32,
}

impl EventDispatcher {
    /// Creates a new empty event dispatcher.
    pub fn new() -> Self {
        Self {
            callback_map: HashMap::new(),
            next_id: 0,
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
    /// Callbacks are invoked in registration order. If a callback returns `true`,
    /// no further callbacks are invoked (propagation is stopped).
    pub fn dispatch<'w, 'c>(&self, event: &Event, ctx: &mut EventContext<'w, 'c>) -> bool {
        if let Some(callbacks) = self.callback_map.get(&event.kind()) {
            for (_id, callback) in callbacks {
                if callback(event, ctx) {
                    return true;
                }
            }
        }
        false
    }
}