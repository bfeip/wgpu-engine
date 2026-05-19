//! Input binding system for mapping raw input events to semantic actions.
//!
//! # Overview
//!
//! The binding system replaces inline `match key { ... }` checks inside operators
//! with a table-driven lookup. Each operator defines an action enum and holds an
//! [`InputMap<A>`] that maps [`InputBinding`] triggers to one or more actions.
//!
//! Bindings can be replaced at runtime via [`InputMap::rebind`], enabling full
//! keybind remapping without recompiling. The complete binding table is exposed
//! through [`InputMap::all_bindings`] for display in a controls reference UI.
//!
//! # Multiple actions per binding
//!
//! A single trigger may map to multiple actions (stored as a `Vec<A>`). This
//! allows a single keypress to fire several independent actions — for example,
//! one from a built-in operator and one from a custom overlay operator. Iterating
//! the returned slice gives all actions in insertion order.
//!
//! # Modifier matching
//!
//! All binding variants that involve a key or mouse button include a [`Modifiers`]
//! field that must match exactly (all four modifier flags). A binding with
//! `modifiers: Modifiers::default()` fires only when no modifier is held.
//!
//! # Key normalization
//!
//! [`Key::Character`] values are normalized to lowercase on both store and lookup,
//! so binding `Key::Character('g')` matches both `'g'` and `'G'` keystrokes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::input::{Key, Modifiers, MouseButton};

/// A raw input condition that can trigger one or more actions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputBinding {
    /// A key press. `Key::Character` values are matched case-insensitively.
    Key {
        key: Key,
        modifiers: Modifiers,
    },
    /// A mouse button click (press + release within the click time threshold).
    MouseClick {
        button: MouseButton,
        modifiers: Modifiers,
    },
    /// A mouse drag (button held and moved past the drag threshold).
    MouseDrag {
        button: MouseButton,
        modifiers: Modifiers,
    },
    /// The start of a mouse drag (first frame past the drag threshold).
    MouseDragStart {
        button: MouseButton,
        modifiers: Modifiers,
    },
    /// The end of a mouse drag (button released after dragging).
    MouseDragEnd {
        button: MouseButton,
        modifiers: Modifiers,
    },
    /// Mouse scroll wheel (any direction or amount).
    MouseScroll,
}

impl InputBinding {
    /// Returns a normalized copy of this binding.
    ///
    /// `Key::Character` is lowercased so that bindings are case-insensitive.
    pub fn normalized(self) -> Self {
        match self {
            Self::Key { key: Key::Character(c), modifiers } => Self::Key {
                key: Key::Character(c.to_lowercase().next().unwrap_or(c)),
                modifiers,
            },
            other => other,
        }
    }
}

/// Maps [`InputBinding`] triggers to semantic actions for a single operator.
///
/// Each trigger may be bound to multiple actions; all are returned as
/// a slice by the `actions_for_*` methods. This lets separate subsystems
/// each register actions for the same input without one stomping the other.
///
/// Inserting the same `(binding, action)` pair a second time is a no-op (deduplicated
/// by value equality). Actions within a single binding are stored in insertion order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMap<A> {
    map: HashMap<InputBinding, Vec<A>>,
}

impl<A: Clone + PartialEq> InputMap<A> {
    /// Creates an empty binding map.
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Builder method: binds `action` to `binding` and returns `self`.
    ///
    /// Normalizes `Key::Character` to lowercase. Duplicate `(binding, action)` pairs
    /// are silently deduplicated.
    pub fn bind(mut self, binding: InputBinding, action: A) -> Self {
        self.add(binding, action);
        self
    }

    /// Binds `action` to `binding` (non-builder variant).
    ///
    /// Normalizes `Key::Character` to lowercase. Duplicate pairs are deduplicated.
    pub fn add(&mut self, binding: InputBinding, action: A) {
        let entry = self.map.entry(binding.normalized()).or_default();
        if !entry.contains(&action) {
            entry.push(action);
        }
    }

    /// Moves all actions from `old` to `new`, replacing the trigger while keeping actions.
    ///
    /// Returns `true` if `old` was found and replaced.
    pub fn rebind(&mut self, old: &InputBinding, new: InputBinding) -> bool {
        let old_norm = old.clone().normalized();
        if let Some(actions) = self.map.remove(&old_norm) {
            let entry = self.map.entry(new.normalized()).or_default();
            for a in actions {
                if !entry.contains(&a) {
                    entry.push(a);
                }
            }
            true
        } else {
            false
        }
    }

    /// Removes all actions for the given binding.
    ///
    /// Returns `true` if the binding existed and was removed.
    pub fn unbind(&mut self, binding: &InputBinding) -> bool {
        self.map.remove(&binding.clone().normalized()).is_some()
    }

    /// Removes a single action from a binding, leaving other actions for that binding intact.
    ///
    /// Returns `true` if the action was found and removed.
    pub fn remove_action(&mut self, binding: &InputBinding, action: &A) -> bool {
        let norm = binding.clone().normalized();
        if let Some(actions) = self.map.get_mut(&norm) {
            if let Some(pos) = actions.iter().position(|a| a == action) {
                actions.remove(pos);
                if actions.is_empty() {
                    self.map.remove(&norm);
                }
                return true;
            }
        }
        false
    }

    /// Returns the actions bound to a key press, or an empty slice if unbound.
    pub fn actions_for_key(&self, key: &Key, modifiers: Modifiers) -> &[A] {
        // normalize key case\match key 
        let key = match key {
            Key::Character(c) => Key::Character(c.to_lowercase().next().unwrap_or(*c)),
            other => other.clone(),
        };
        let norm = InputBinding::Key { key, modifiers };
        self.map.get(&norm).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns the actions bound to a mouse click, or an empty slice if unbound.
    pub fn actions_for_click(&self, button: MouseButton, modifiers: Modifiers) -> &[A] {
        let binding = InputBinding::MouseClick { button, modifiers };
        self.map.get(&binding).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns the actions bound to a mouse drag, or an empty slice if unbound.
    pub fn actions_for_drag(&self, button: MouseButton, modifiers: Modifiers) -> &[A] {
        let binding = InputBinding::MouseDrag { button, modifiers };
        self.map.get(&binding).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns the actions bound to a mouse drag start, or an empty slice if unbound.
    pub fn actions_for_drag_start(&self, button: MouseButton, modifiers: Modifiers) -> &[A] {
        let binding = InputBinding::MouseDragStart { button, modifiers };
        self.map.get(&binding).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns the actions bound to a mouse drag end, or an empty slice if unbound.
    pub fn actions_for_drag_end(&self, button: MouseButton, modifiers: Modifiers) -> &[A] {
        let binding = InputBinding::MouseDragEnd { button, modifiers };
        self.map.get(&binding).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns the actions bound to the mouse scroll wheel, or an empty slice if unbound.
    pub fn actions_for_scroll(&self) -> &[A] {
        self.map.get(&InputBinding::MouseScroll).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Returns all current bindings as an iterator of `(trigger, actions)` pairs.
    ///
    /// Iteration order is unspecified (HashMap order).
    pub fn all_bindings(&self) -> impl Iterator<Item = (&InputBinding, &[A])> {
        self.map.iter().map(|(b, v)| (b, v.as_slice()))
    }
}

impl<A: Clone + PartialEq> Default for InputMap<A> {
    fn default() -> Self {
        Self::new()
    }
}
