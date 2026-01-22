//! Selection management for scene elements.
//!
//! This module provides types and utilities for managing selection state in the viewer.
//! Currently supports node-level selection, with the design allowing for future expansion
//! to more granular selection (faces, edges, points).

use std::collections::HashSet;

use crate::scene::NodeId;

/// Configuration for selection visual feedback.
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Color of the selection outline (RGBA, 0.0-1.0).
    pub outline_color: [f32; 4],
    /// Width of the outline in pixels (screen-space).
    pub outline_width: f32,
    /// Whether outline rendering is enabled.
    pub outline_enabled: bool,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            outline_color: [1.0, 0.6, 0.0, 1.0], // Orange
            outline_width: 3.0,                   // 3 pixels
            outline_enabled: true,
        }
    }
}

/// Represents a selectable element in the scene.
///
/// Designed for future expansion to support different selection granularities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionItem {
    /// A complete scene node
    Node(NodeId),
    // Future expansion:
    // Face { node_id: NodeId, face_index: u32 },
    // Edge { node_id: NodeId, edge_index: u32 },
    // Point { node_id: NodeId, point_index: u32 },
}

impl SelectionItem {
    /// Returns the NodeId associated with this selection item.
    pub fn node_id(&self) -> NodeId {
        match self {
            SelectionItem::Node(id) => *id,
        }
    }
}

/// Manages the current selection state.
///
/// Supports single and multiple selection of scene elements.
/// Maintains both a set for fast lookup and a vector for ordered iteration.
pub struct SelectionManager {
    /// Set of currently selected items (for fast lookup)
    selected: HashSet<SelectionItem>,
    /// Selection in order of addition (for ordered iteration)
    selection_order: Vec<SelectionItem>,
    /// The primary/active selection (last selected, or explicitly set)
    primary: Option<SelectionItem>,
    /// Configuration for selection visual feedback
    config: SelectionConfig,
}

impl Default for SelectionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectionManager {
    /// Creates a new empty selection manager with default configuration.
    pub fn new() -> Self {
        Self {
            selected: HashSet::new(),
            selection_order: Vec::new(),
            primary: None,
            config: SelectionConfig::default(),
        }
    }

    /// Returns the selection configuration.
    pub fn config(&self) -> &SelectionConfig {
        &self.config
    }

    /// Returns a mutable reference to the selection configuration.
    pub fn config_mut(&mut self) -> &mut SelectionConfig {
        &mut self.config
    }

    // ========== Query API ==========

    /// Returns true if there are no selections.
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Returns the number of selected items.
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    /// Returns true if the given item is selected.
    pub fn contains(&self, item: &SelectionItem) -> bool {
        self.selected.contains(item)
    }

    /// Returns true if the given node is selected (any SelectionItem referencing it).
    pub fn is_node_selected(&self, node_id: NodeId) -> bool {
        self.selected.iter().any(|item| item.node_id() == node_id)
    }

    /// Returns the primary/active selection, if any.
    pub fn primary(&self) -> Option<SelectionItem> {
        self.primary
    }

    /// Iterates over all selected items in order of selection.
    pub fn iter(&self) -> impl Iterator<Item = &SelectionItem> {
        self.selection_order.iter()
    }

    /// Returns all selected items as a slice.
    pub fn as_slice(&self) -> &[SelectionItem] {
        &self.selection_order
    }

    /// Returns all selected node IDs.
    pub fn selected_nodes(&self) -> Vec<NodeId> {
        self.selection_order
            .iter()
            .map(|item| item.node_id())
            .collect()
    }

    // ========== Mutation API ==========

    /// Clears all selections.
    pub fn clear(&mut self) {
        self.selected.clear();
        self.selection_order.clear();
        self.primary = None;
    }

    /// Sets the selection to exactly this item (clears previous selections).
    pub fn set(&mut self, item: SelectionItem) {
        self.clear();
        self.add(item);
    }

    /// Adds an item to the selection (does nothing if already selected).
    pub fn add(&mut self, item: SelectionItem) {
        if self.selected.insert(item) {
            self.selection_order.push(item);
        }
        self.primary = Some(item);
    }

    /// Removes an item from the selection.
    pub fn remove(&mut self, item: &SelectionItem) -> bool {
        if self.selected.remove(item) {
            self.selection_order.retain(|i| i != item);
            if self.primary == Some(*item) {
                self.primary = self.selection_order.last().copied();
            }
            true
        } else {
            false
        }
    }

    /// Toggles selection of an item.
    pub fn toggle(&mut self, item: SelectionItem) {
        if self.contains(&item) {
            self.remove(&item);
        } else {
            self.add(item);
        }
    }

    /// Extends the selection with multiple items.
    pub fn extend(&mut self, items: impl IntoIterator<Item = SelectionItem>) {
        for item in items {
            self.add(item);
        }
    }

    /// Removes all items referencing a specific node.
    pub fn remove_node(&mut self, node_id: NodeId) {
        let items_to_remove: Vec<_> = self
            .selected
            .iter()
            .filter(|item| item.node_id() == node_id)
            .copied()
            .collect();

        for item in items_to_remove {
            self.remove(&item);
        }
    }

    /// Sets the primary selection without changing what is selected.
    pub fn set_primary(&mut self, item: Option<SelectionItem>) {
        self.primary = item;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_manager_is_empty() {
        let manager = SelectionManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
        assert!(manager.primary().is_none());
    }

    #[test]
    fn test_add_and_contains() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(42);

        manager.add(item);

        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
        assert!(manager.contains(&item));
        assert!(manager.is_node_selected(42));
        assert_eq!(manager.primary(), Some(item));
    }

    #[test]
    fn test_add_duplicate() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(42);

        manager.add(item);
        manager.add(item);

        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_set_clears_previous() {
        let mut manager = SelectionManager::new();
        manager.add(SelectionItem::Node(1));
        manager.add(SelectionItem::Node(2));
        manager.add(SelectionItem::Node(3));

        manager.set(SelectionItem::Node(42));

        assert_eq!(manager.len(), 1);
        assert!(manager.is_node_selected(42));
        assert!(!manager.is_node_selected(1));
    }

    #[test]
    fn test_remove() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(42);

        manager.add(item);
        assert!(manager.remove(&item));
        assert!(!manager.remove(&item)); // Already removed

        assert!(manager.is_empty());
        assert!(!manager.contains(&item));
    }

    #[test]
    fn test_toggle() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(42);

        manager.toggle(item);
        assert!(manager.contains(&item));

        manager.toggle(item);
        assert!(!manager.contains(&item));
    }

    #[test]
    fn test_clear() {
        let mut manager = SelectionManager::new();
        manager.add(SelectionItem::Node(1));
        manager.add(SelectionItem::Node(2));

        manager.clear();

        assert!(manager.is_empty());
        assert!(manager.primary().is_none());
    }

    #[test]
    fn test_selection_order() {
        let mut manager = SelectionManager::new();
        manager.add(SelectionItem::Node(1));
        manager.add(SelectionItem::Node(2));
        manager.add(SelectionItem::Node(3));

        let nodes: Vec<NodeId> = manager.selected_nodes();
        assert_eq!(nodes, vec![1, 2, 3]);
    }

    #[test]
    fn test_primary_updates_on_remove() {
        let mut manager = SelectionManager::new();
        manager.add(SelectionItem::Node(1));
        manager.add(SelectionItem::Node(2));

        assert_eq!(manager.primary(), Some(SelectionItem::Node(2)));

        manager.remove(&SelectionItem::Node(2));
        assert_eq!(manager.primary(), Some(SelectionItem::Node(1)));

        manager.remove(&SelectionItem::Node(1));
        assert_eq!(manager.primary(), None);
    }

    #[test]
    fn test_selection_item_node_id() {
        let item = SelectionItem::Node(42);
        assert_eq!(item.node_id(), 42);
    }
}
