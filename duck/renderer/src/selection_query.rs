//! Trait for providing selection information to the renderer.
//!
//! This trait decouples the renderer from any specific selection implementation,
//! allowing it to query which nodes are selected for outline rendering.

use crate::scene::NodeId;

/// Configuration for selection outline rendering.
#[derive(Debug, Clone)]
pub struct OutlineConfig {
    /// Color of the selection outline (RGBA, 0.0-1.0).
    pub color: [f32; 4],
    /// Width of the outline in pixels (screen-space).
    pub width_pixels: f32,
}

impl Default for OutlineConfig {
    fn default() -> Self {
        Self {
            color: [1.0, 0.6, 0.0, 1.0], // Orange
            width_pixels: 3.0,
        }
    }
}

/// Trait for querying selection state during rendering.
///
/// The renderer uses this trait to determine which nodes should have
/// selection outlines, without depending on any specific selection
/// management implementation.
pub trait SelectionQuery {
    /// Returns true if there are no selections.
    fn is_empty(&self) -> bool;

    /// Returns true if the given node is selected.
    fn is_node_selected(&self, node_id: NodeId) -> bool;

    /// Returns the face indices selected on `node_id`, if any.
    fn selected_faces_for_node(&self, node_id: NodeId) -> Vec<u32>;

    /// Returns the edge indices selected on `node_id`, if any.
    fn selected_edges_for_node(&self, node_id: NodeId) -> Vec<u32>;

    /// Returns all nodes that have at least one face selected.
    fn nodes_with_face_selection(&self) -> Vec<NodeId>;

    /// Returns all nodes that have at least one edge selected.
    fn nodes_with_edge_selection(&self) -> Vec<NodeId>;

    /// Returns the outline rendering configuration.
    fn outline_config(&self) -> OutlineConfig;
}
