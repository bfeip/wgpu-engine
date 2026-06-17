use crate::input::{
    ElementState, KeyEvent, MouseButton, MouseScrollDelta, TouchId, TouchPhase,
};

/// Discriminant enum for [`DeviceEvent`] variants.
///
/// Can be used as a fast pre-filter inside [`Operator::dispatch`] implementations.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum DeviceEventKind {
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

/// Low-level device/input events with associated data.
#[derive(Clone)]
pub enum DeviceEvent {
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

impl DeviceEvent {
    /// Returns the [`DeviceEventKind`] discriminant for this event.
    pub fn kind(&self) -> DeviceEventKind {
        match self {
            Self::Update { .. } => DeviceEventKind::Update,
            Self::Resized(_) => DeviceEventKind::Resized,
            Self::KeyboardInput { .. } => DeviceEventKind::KeyboardInput,
            Self::MouseMotion { .. } => DeviceEventKind::MouseMotion,
            Self::CursorMoved { .. } => DeviceEventKind::CursorMoved,
            Self::MouseInput { .. } => DeviceEventKind::MouseInput,
            Self::MouseWheel { .. } => DeviceEventKind::MouseWheel,
            Self::MouseDragStart { .. } => DeviceEventKind::MouseDragStart,
            Self::MouseDrag { .. } => DeviceEventKind::MouseDrag,
            Self::MouseDragEnd { .. } => DeviceEventKind::MouseDragEnd,
            Self::MouseClick { .. } => DeviceEventKind::MouseClick,
            Self::Touch { .. } => DeviceEventKind::Touch,
            #[cfg(test)]
            Self::Test => DeviceEventKind::Test,
        }
    }
}
