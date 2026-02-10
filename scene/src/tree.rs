use super::{NodeId, Node, Scene};

/// Trait for implementing tree traversal operations.
///
/// Implementors of this trait can be passed to tree walking functions
/// to perform arbitrary operations on each node during traversal.
///
/// The visitor receives callbacks when entering and exiting nodes.
pub trait TreeVisitor {
    /// Called when entering a node (before processing its children).
    ///
    /// Returns true to continue traversing children, false to skip the subtree.
    fn enter_node(&mut self, node: &Node) -> bool;

    /// Called when exiting a node (after processing its children).
    fn exit_node(&mut self, node: &Node);
}

/// Walks the scene tree starting from a given node.
pub fn walk_tree<V: TreeVisitor>(
    scene: &Scene,
    node_id: NodeId,
    visitor: &mut V,
) {
    // Get the node (return early if not found)
    let node = match scene.get_node(node_id) {
        Some(n) => n,
        None => return,
    };

    // Enter this node
    let should_visit_children = visitor.enter_node(node);

    // Recurse for all children if enter_node returned true
    if should_visit_children {
        for &child_id in node.children() {
            walk_tree(scene, child_id, visitor);
        }
    }

    // Exit this node
    visitor.exit_node(node);
}
