/// Input types that are independent of any specific windowing library.
/// These types mirror common windowing system input abstractions.

/// Element state (pressed or released)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ElementState {
    Pressed,
    Released,
}

/// Mouse button identifier
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
    Other(u16),
}

/// Mouse scroll delta
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MouseScrollDelta {
    /// Scroll delta in lines
    LineDelta(f32, f32),
    /// Scroll delta in pixels
    PixelDelta(f32, f32),
}

/// Keyboard physical key code (scancode)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PhysicalKey {
    /// A known key code
    Code(u32),
    /// An unidentified key
    Unidentified,
}

/// Keyboard logical key (with consideration for layout)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    /// A named key
    Named(NamedKey),
    /// A character key
    Character(char),
    /// An unidentified key
    Unidentified,
}

/// Named keyboard keys
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NamedKey {
    Escape,
    Enter,
    Tab,
    Backspace,
    Delete,
    Space,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Home,
    End,
    PageUp,
    PageDown,
    Control,
    Alt,
    Shift,
    Super,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
}

/// Keyboard event
#[derive(Debug, Clone, PartialEq)]
pub struct KeyEvent {
    /// Physical key (scancode)
    pub physical_key: PhysicalKey,
    /// Logical key (with layout consideration)
    pub logical_key: Key,
    /// Whether the key is pressed or released
    pub state: ElementState,
    /// Whether this is a repeat event (key held down)
    pub repeat: bool,
}

/// Keyboard modifiers state
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub control: bool,
    pub alt: bool,
    pub super_key: bool,
}
