//! Selection management for scene elements.
//!
//! Supports node-level selection as well as sub-geometry selection (faces, edges,
//! points) for meshes that carry [`Topology`](crate::scene::Topology) metadata.

use std::collections::HashSet;

use crate::scene::NodeId;
use crate::renderer::{HighlightQuery, HighlightConfig};

/// Configuration for selection visual feedback.
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Color of the primary selection outline (RGBA, 0.0-1.0).
    pub outline_color: [f32; 4],
    /// Color of secondary (non-primary) selection outlines (RGBA, 0.0-1.0).
    pub secondary_outline_color: [f32; 4],
    /// Width of the outline in pixels (screen-space).
    pub outline_width: f32,
    /// Alpha (0.0-1.0) of the transparent face-highlight overlay.
    pub face_highlight_alpha: f32,
    /// Whether outline rendering is enabled.
    pub outline_enabled: bool,
    /// Whether debug annotations (pick rays) are drawn on selection.
    pub debug_annotations: bool,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            outline_color: [1.0, 0.6, 0.0, 1.0],           // Orange
            secondary_outline_color: [0.7, 0.35, 0.0, 1.0], // Darker orange
            outline_width: 3.0,                              // 3 pixels
            face_highlight_alpha: 0.4,
            outline_enabled: true,
            debug_annotations: false,
        }
    }
}

/// Represents a selectable element in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectionItem {
    /// A complete scene node.
    Node(NodeId),
    /// A single face within a node's mesh topology.
    Face { node_id: NodeId, face_index: u32 },
    /// A single edge within a node's mesh topology.
    Edge { node_id: NodeId, edge_index: u32 },
    /// A single point within a node's mesh topology.
    Pointset { node_id: NodeId, pointset_index: u32 },
}

impl SelectionItem {
    /// Returns the [`NodeId`] associated with this selection item.
    pub fn node_id(&self) -> NodeId {
        match self {
            SelectionItem::Node(id) => *id,
            SelectionItem::Face { node_id, .. } => *node_id,
            SelectionItem::Edge { node_id, .. } => *node_id,
            SelectionItem::Pointset { node_id, .. } => *node_id,
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

    /// Returns true if the given face is selected.
    pub fn is_face_selected(&self, node_id: NodeId, face_index: u32) -> bool {
        self.selected.contains(&SelectionItem::Face { node_id, face_index })
    }

    /// Returns true if the given edge is selected.
    pub fn is_edge_selected(&self, node_id: NodeId, edge_index: u32) -> bool {
        self.selected.contains(&SelectionItem::Edge { node_id, edge_index })
    }

    /// Returns true if the given pointset is selected.
    pub fn is_pointset_selected(&self, node_id: NodeId, pointset_index: u32) -> bool {
        self.selected.contains(&SelectionItem::Pointset { node_id, pointset_index })
    }

    /// Returns an iterator over face indices currently selected on `node_id`.
    pub fn selected_faces_for_node(&self, node_id: NodeId) -> impl Iterator<Item = u32> + '_ {
        self.selection_order.iter().filter_map(move |item| {
            if let SelectionItem::Face { node_id: nid, face_index } = item {
                if *nid == node_id { Some(*face_index) } else { None }
            } else {
                None
            }
        })
    }

    /// Returns an iterator over edge indices currently selected on `node_id`.
    pub fn selected_edges_for_node(&self, node_id: NodeId) -> impl Iterator<Item = u32> + '_ {
        self.selection_order.iter().filter_map(move |item| {
            if let SelectionItem::Edge { node_id: nid, edge_index } = item {
                if *nid == node_id { Some(*edge_index) } else { None }
            } else {
                None
            }
        })
    }

    /// Returns an iterator over pointset indices currently selected on `node_id`.
    pub fn selected_pointsets_for_node(&self, node_id: NodeId) -> impl Iterator<Item = u32> + '_ {
        self.selection_order.iter().filter_map(move |item| {
            if let SelectionItem::Pointset { node_id: nid, pointset_index } = item {
                if *nid == node_id { Some(*pointset_index) } else { None }
            } else {
                None
            }
        })
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
    /// The first item added becomes the primary and remains so until cleared.
    pub fn add(&mut self, item: SelectionItem) {
        if self.selected.insert(item) {
            self.selection_order.push(item);
        }
        if self.primary.is_none() {
            self.primary = Some(item);
        }
    }

    /// Removes an item from the selection.
    pub fn remove(&mut self, item: &SelectionItem) -> bool {
        if self.selected.remove(item) {
            self.selection_order.retain(|i| i != item);
            if self.primary == Some(*item) {
                self.primary = self.selection_order.first().copied();
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

impl HighlightQuery for SelectionManager {
    fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    fn outlined_nodes(&self) -> Vec<NodeId> {
        self.selection_order
            .iter()
            .filter_map(|item| match item {
                SelectionItem::Node(node_id) => Some(*node_id),
                _ => None,
            })
            .collect()
    }

    fn primary_outlined_node(&self) -> Option<NodeId> {
        match self.primary {
            Some(SelectionItem::Node(node_id)) => Some(node_id),
            _ => None,
        }
    }

    fn highlighted_faces_for_node(&self, node_id: NodeId) -> Vec<u32> {
        self.selected_faces_for_node(node_id).collect()
    }

    fn highlighted_edges_for_node(&self, node_id: NodeId) -> Vec<u32> {
        self.selected_edges_for_node(node_id).collect()
    }

    fn highlighted_pointsets_for_node(&self, node_id: NodeId) -> Vec<u32> {
        self.selected_pointsets_for_node(node_id).collect()
    }

    fn nodes_with_highlighted_faces(&self) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .selected
            .iter()
            .filter_map(|item| {
                if let SelectionItem::Face { node_id, .. } = item { Some(*node_id) } else { None }
            })
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    fn nodes_with_highlighted_edges(&self) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .selected
            .iter()
            .filter_map(|item| {
                if let SelectionItem::Edge { node_id, .. } = item { Some(*node_id) } else { None }
            })
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    fn nodes_with_highlighted_pointsets(&self) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self
            .selected
            .iter()
            .filter_map(|item| {
                if let SelectionItem::Pointset { node_id, .. } = item { Some(*node_id) } else { None }
            })
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    fn primary_face(&self) -> Option<(NodeId, u32)> {
        match self.primary {
            Some(SelectionItem::Face { node_id, face_index }) => Some((node_id, face_index)),
            _ => None,
        }
    }

    fn primary_edge(&self) -> Option<(NodeId, u32)> {
        match self.primary {
            Some(SelectionItem::Edge { node_id, edge_index }) => Some((node_id, edge_index)),
            _ => None,
        }
    }

    fn primary_pointset(&self) -> Option<(NodeId, u32)> {
        match self.primary {
            Some(SelectionItem::Pointset { node_id, pointset_index: point_index }) => Some((node_id, point_index)),
            _ => None,
        }
    }

    fn highlight_config(&self) -> HighlightConfig {
        HighlightConfig {
            color: self.config.outline_color,
            width_pixels: self.config.outline_width,
            face_alpha: self.config.face_highlight_alpha,
        }
    }

    fn secondary_highlight_config(&self) -> Option<HighlightConfig> {
        let has_secondary = self.selected.iter().any(|item| Some(*item) != self.primary);
        has_secondary.then(|| HighlightConfig {
            color: self.config.secondary_outline_color,
            width_pixels: self.config.outline_width,
            face_alpha: self.config.face_highlight_alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nid() -> NodeId { NodeId::new() }

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
        let id = nid();
        let item = SelectionItem::Node(id);

        manager.add(item);

        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
        assert!(manager.contains(&item));
        assert!(manager.is_node_selected(id));
        assert_eq!(manager.primary(), Some(item));
    }

    #[test]
    fn test_add_duplicate() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(nid());

        manager.add(item);
        manager.add(item);

        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_set_clears_previous() {
        let mut manager = SelectionManager::new();
        let id_a = nid();
        let id_b = nid();
        let id_c = nid();
        let id_new = nid();

        manager.add(SelectionItem::Node(id_a));
        manager.add(SelectionItem::Node(id_b));
        manager.add(SelectionItem::Node(id_c));

        manager.set(SelectionItem::Node(id_new));

        assert_eq!(manager.len(), 1);
        assert!(manager.is_node_selected(id_new));
        assert!(!manager.is_node_selected(id_a));
    }

    #[test]
    fn test_remove() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(nid());

        manager.add(item);
        assert!(manager.remove(&item));
        assert!(!manager.remove(&item)); // Already removed

        assert!(manager.is_empty());
        assert!(!manager.contains(&item));
    }

    #[test]
    fn test_toggle() {
        let mut manager = SelectionManager::new();
        let item = SelectionItem::Node(nid());

        manager.toggle(item);
        assert!(manager.contains(&item));

        manager.toggle(item);
        assert!(!manager.contains(&item));
    }

    #[test]
    fn test_clear() {
        let mut manager = SelectionManager::new();
        manager.add(SelectionItem::Node(nid()));
        manager.add(SelectionItem::Node(nid()));

        manager.clear();

        assert!(manager.is_empty());
        assert!(manager.primary().is_none());
    }

    #[test]
    fn test_selection_order() {
        let mut manager = SelectionManager::new();
        let id_a = nid();
        let id_b = nid();
        let id_c = nid();
        manager.add(SelectionItem::Node(id_a));
        manager.add(SelectionItem::Node(id_b));
        manager.add(SelectionItem::Node(id_c));

        let nodes: Vec<NodeId> = manager.selected_nodes();
        assert_eq!(nodes, vec![id_a, id_b, id_c]);
    }

    #[test]
    fn test_primary_is_first_selected() {
        let mut manager = SelectionManager::new();
        let id_a = nid();
        let id_b = nid();

        manager.add(SelectionItem::Node(id_a));
        manager.add(SelectionItem::Node(id_b));

        // First selection remains primary regardless of subsequent adds.
        assert_eq!(manager.primary(), Some(SelectionItem::Node(id_a)));

        // Removing the primary falls back to the next oldest item.
        manager.remove(&SelectionItem::Node(id_a));
        assert_eq!(manager.primary(), Some(SelectionItem::Node(id_b)));

        manager.remove(&SelectionItem::Node(id_b));
        assert_eq!(manager.primary(), None);
    }

    #[test]
    fn test_selection_item_node_id() {
        let id = nid();
        let item = SelectionItem::Node(id);
        assert_eq!(item.node_id(), id);
    }
}
