use crate::common;

use super::{InstanceId, Node, NodeId, Scene};
use cgmath::{Matrix3, Matrix4, SquareMatrix};

/// Trait for implementing tree traversal operations.
///
/// Implementors of this trait can be passed to tree walking functions
/// to perform arbitrary operations on each node during traversal.
///
/// The visitor receives callbacks when entering and exiting nodes.
pub trait TreeVisitor {
    /// Called when entering a node (before processing its children).
    fn enter_node(&mut self, node: &Node);

    /// Called when exiting a node (after processing its children).
    fn exit_node(&mut self, node: &Node);
}

/// Represents an instance with its computed world transform.
pub struct InstanceTransform {
    pub instance_id: InstanceId,
    pub world_transform: Matrix4<f32>,
    pub normal_matrix: Matrix3<f32>,
}

impl InstanceTransform {
    /// Creates a new InstanceTransform with the given world transform.
    /// The normal matrix is computed from the world transform.
    pub fn new(instance_id: InstanceId, world_transform: Matrix4<f32>) -> Self {
        let normal_matrix = common::compute_normal_matrix(&world_transform);
        Self {
            instance_id,
            world_transform,
            normal_matrix,
        }
    }
}

/// Recursively walks the scene tree starting from a given node.
pub fn walk_tree_recursive<V: TreeVisitor>(
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
    visitor.enter_node(node);

    // Recurse for all children
    for &child_id in node.children() {
        walk_tree_recursive(scene, child_id, visitor);
    }

    // Exit this node
    visitor.exit_node(node);
}

/// Visitor implementation that collects instance transforms during tree traversal.
pub struct InstanceTransformCollector {
    /// Stack of world transforms (one per tree depth level)
    transform_stack: Vec<Matrix4<f32>>,
    /// Stack tracking whether recomputation is needed at each level
    needs_recompute_stack: Vec<bool>,
    /// Collected instance transforms
    results: Vec<InstanceTransform>,
}

impl InstanceTransformCollector {
    /// Creates a new collector with an empty results vector.
    pub fn new() -> Self {
        Self {
            transform_stack: Vec::new(),
            needs_recompute_stack: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Consumes the collector and returns the collected instance transforms.
    pub fn into_results(self) -> Vec<InstanceTransform> {
        self.results
    }

    /// Gets the current parent transform from the top of the stack.
    fn current_parent_transform(&self) -> Matrix4<f32> {
        *self.transform_stack.last().unwrap_or(&Matrix4::identity())
    }

    /// Gets whether any parent forced a recomputation.
    fn parent_changed(&self) -> bool {
        *self.needs_recompute_stack.last().unwrap_or(&false)
    }
}

impl TreeVisitor for InstanceTransformCollector {
    fn enter_node(&mut self, node: &Node) {
        let parent_transform = self.current_parent_transform();
        let parent_changed = self.parent_changed();

        // Determine if we need to recompute this node's world transform
        let node_dirty = node.is_dirty();
        let needs_recompute = parent_changed || node_dirty;

        let world_transform = if needs_recompute {
            // Compute world transform: parent * local
            let local_transform = node.compute_local_transform();
            let new_world_transform = parent_transform * local_transform;

            // Cache the result
            node.set_cached_world_transform(new_world_transform);

            new_world_transform
        } else {
            // Use cached transform
            node.cached_world_transform()
        };

        // Push state onto stacks for children
        self.transform_stack.push(world_transform);
        self.needs_recompute_stack.push(needs_recompute);

        // If this node has an instance, collect it
        if let Some(instance_id) = node.instance() {
            self.results.push(InstanceTransform::new(instance_id, world_transform));
        }
    }

    fn exit_node(&mut self, _node: &Node) {
        // Pop the stacks when leaving the node
        self.transform_stack.pop();
        self.needs_recompute_stack.pop();
    }
}

/// Walks the entire scene tree and collects all instances with their world transforms.
pub fn collect_instance_transforms(scene: &Scene) -> Vec<InstanceTransform> {
    let mut visitor = InstanceTransformCollector::new();

    // Process each root node
    for &root_id in scene.root_nodes() {
        walk_tree_recursive(scene, root_id, &mut visitor);
    }

    visitor.into_results()
}
