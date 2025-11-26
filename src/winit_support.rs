/// Winit integration module - provides conversion functions between winit types and our input types
///
/// This module is only compiled when the `winit-support` feature is enabled.

use crate::common::PhysicalSize;
use crate::input::{PhysicalPosition, ElementState, MouseButton, MouseScrollDelta, KeyEvent, PhysicalKey, Key, NamedKey};
use crate::event::Event;

/// Converts a winit PhysicalSize to our PhysicalSize
pub fn convert_physical_size(size: winit::dpi::PhysicalSize<u32>) -> PhysicalSize<u32> {
    PhysicalSize::new(size.width, size.height)
}

/// Converts a winit PhysicalPosition to our PhysicalPosition
pub fn convert_physical_position<T>(pos: winit::dpi::PhysicalPosition<T>) -> PhysicalPosition<T> {
    PhysicalPosition::new(pos.x, pos.y)
}

/// Converts a winit ElementState to our ElementState
pub fn convert_element_state(state: winit::event::ElementState) -> ElementState {
    match state {
        winit::event::ElementState::Pressed => ElementState::Pressed,
        winit::event::ElementState::Released => ElementState::Released,
    }
}

/// Converts a winit MouseButton to our MouseButton
pub fn convert_mouse_button(button: winit::event::MouseButton) -> MouseButton {
    match button {
        winit::event::MouseButton::Left => MouseButton::Left,
        winit::event::MouseButton::Right => MouseButton::Right,
        winit::event::MouseButton::Middle => MouseButton::Middle,
        winit::event::MouseButton::Back => MouseButton::Back,
        winit::event::MouseButton::Forward => MouseButton::Forward,
        winit::event::MouseButton::Other(id) => MouseButton::Other(id),
    }
}

/// Converts a winit MouseScrollDelta to our MouseScrollDelta
pub fn convert_mouse_scroll_delta(delta: winit::event::MouseScrollDelta) -> MouseScrollDelta {
    match delta {
        winit::event::MouseScrollDelta::LineDelta(x, y) => MouseScrollDelta::LineDelta(x, y),
        winit::event::MouseScrollDelta::PixelDelta(pos) => {
            MouseScrollDelta::PixelDelta(pos.x as f32, pos.y as f32)
        }
    }
}

/// Converts a winit KeyEvent to our KeyEvent (simplified version)
pub fn convert_key_event(event: &winit::event::KeyEvent) -> KeyEvent {
    let physical_key = match event.physical_key {
        winit::keyboard::PhysicalKey::Code(code) => PhysicalKey::Code(code as u32),
        winit::keyboard::PhysicalKey::Unidentified(_code) => PhysicalKey::Unidentified,
    };

    let logical_key = match &event.logical_key {
        winit::keyboard::Key::Named(named) => {
            // Convert named keys
            Key::Named(convert_named_key(*named))
        }
        winit::keyboard::Key::Character(s) => {
            if let Some(c) = s.chars().next() {
                Key::Character(c)
            } else {
                Key::Unidentified
            }
        }
        _ => Key::Unidentified,
    };

    KeyEvent {
        physical_key,
        logical_key,
        state: convert_element_state(event.state),
        repeat: event.repeat,
    }
}

/// Converts a winit NamedKey to our NamedKey (partial mapping)
fn convert_named_key(key: winit::keyboard::NamedKey) -> NamedKey {
    use winit::keyboard::NamedKey as WK;
    match key {
        WK::Escape => NamedKey::Escape,
        WK::Enter => NamedKey::Enter,
        WK::Tab => NamedKey::Tab,
        WK::Backspace => NamedKey::Backspace,
        WK::Delete => NamedKey::Delete,
        WK::Space => NamedKey::Space,
        WK::ArrowLeft => NamedKey::ArrowLeft,
        WK::ArrowRight => NamedKey::ArrowRight,
        WK::ArrowUp => NamedKey::ArrowUp,
        WK::ArrowDown => NamedKey::ArrowDown,
        WK::Home => NamedKey::Home,
        WK::End => NamedKey::End,
        WK::PageUp => NamedKey::PageUp,
        WK::PageDown => NamedKey::PageDown,
        WK::Control => NamedKey::Control,
        WK::Alt => NamedKey::Alt,
        WK::Shift => NamedKey::Shift,
        WK::Super => NamedKey::Super,
        WK::F1 => NamedKey::F1,
        WK::F2 => NamedKey::F2,
        WK::F3 => NamedKey::F3,
        WK::F4 => NamedKey::F4,
        WK::F5 => NamedKey::F5,
        WK::F6 => NamedKey::F6,
        WK::F7 => NamedKey::F7,
        WK::F8 => NamedKey::F8,
        WK::F9 => NamedKey::F9,
        WK::F10 => NamedKey::F10,
        WK::F11 => NamedKey::F11,
        WK::F12 => NamedKey::F12,
        _ => NamedKey::Escape, // Default fallback
    }
}

/// Converts a winit Event to our Event
pub fn convert_event(wevent: winit::event::Event<()>) -> Option<Event> {
    use winit::event::Event as WEvent;

    match wevent {
        WEvent::WindowEvent { event, .. } => convert_window_event(event),
        WEvent::DeviceEvent { event, .. } => convert_device_event(event),
        _ => None,
    }
}

/// Converts a winit WindowEvent to our Event
pub fn convert_window_event(wevent: winit::event::WindowEvent) -> Option<Event> {
    use winit::event::WindowEvent as WEvent;

    match wevent {
        WEvent::Resized(size) => Some(Event::Resized(convert_physical_size(size))),
        WEvent::RedrawRequested => Some(Event::RedrawRequested),
        WEvent::KeyboardInput { event, is_synthetic, .. } => Some(Event::KeyboardInput {
            event: convert_key_event(&event),
            is_synthetic,
        }),
        WEvent::CursorMoved { position, .. } => Some(Event::CursorMoved {
            position: convert_physical_position(position),
        }),
        WEvent::MouseInput { state, button, .. } => Some(Event::MouseInput {
            state: convert_element_state(state),
            button: convert_mouse_button(button),
        }),
        WEvent::MouseWheel { delta, .. } => Some(Event::MouseWheel {
            delta: convert_mouse_scroll_delta(delta),
        }),
        _ => None,
    }
}

/// Converts a winit DeviceEvent to our Event
pub fn convert_device_event(wevent: winit::event::DeviceEvent) -> Option<Event> {
    use winit::event::DeviceEvent as DEvent;

    match wevent {
        DEvent::MouseMotion { delta } => Some(Event::MouseMotion { delta }),
        _ => None,
    }
}
