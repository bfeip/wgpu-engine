//! Trait for providing highlight information to the renderer.
//!
//! This trait decouples the renderer from any specific highlight implementation,
//! allowing it to query which nodes are highlighted for outline rendering.

use crate::scene::NodeId;

/// Configuration for highlight outline rendering.
#[derive(Debug, Clone)]
pub struct OutlineConfig {
    /// Color of the highlight outline (RGBA, 0.0-1.0).
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

/// Trait for querying highlight state during rendering.
///
/// The renderer uses this trait to determine which nodes should have
/// highlight outlines, without depending on any specific highlight
/// management implementation.
pub trait HighlightQuery {
    /// Returns true if there are no highlights.
    fn is_empty(&self) -> bool;

    /// Returns true if the given node is highlighted.
    fn is_node_highlighted(&self, node_id: NodeId) -> bool;

    /// Returns the face indices highlighted on `node_id`, if any.
    fn highlighted_faces_for_node(&self, node_id: NodeId) -> Vec<u32>;

    /// Returns the edge indices highlighted on `node_id`, if any.
    fn highlighted_edges_for_node(&self, node_id: NodeId) -> Vec<u32>;

    /// Returns all nodes that have at least one face highlighted.
    fn nodes_with_highlighted_faces(&self) -> Vec<NodeId>;

    /// Returns all nodes that have at least one edge highlighted.
    fn nodes_with_highlighted_edges(&self) -> Vec<NodeId>;

    /// Returns the outline rendering configuration.
    fn outline_config(&self) -> OutlineConfig;
}
