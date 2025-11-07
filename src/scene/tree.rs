use super::{InstanceId, Node, NodeId, Scene};
use cgmath::{Matrix3, Matrix4, SquareMatrix};

/// Trait for implementing custom tree traversal operations.
///
/// Implementors of this trait can be passed to tree walking functions
/// to perform arbitrary operations on each node during traversal.
///
/// The visitor receives callbacks when entering and exiting nodes,
/// allowing it to maintain state (like a transform stack) during traversal.
pub trait TreeVisitor {
    /// Called when entering a node (before processing its children).
    ///
    /// # Arguments
    /// * `node` - Reference to the current node
    fn enter_node(&mut self, node: &Node);

    /// Called when exiting a node (after processing its children).
    ///
    /// # Arguments
    /// * `node` - Reference to the current node
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
        let normal_matrix = compute_normal_matrix(&world_transform);
        Self {
            instance_id,
            world_transform,
            normal_matrix,
        }
    }
}

/// Computes the normal matrix from a world transform matrix.
///
/// The normal matrix is the inverse-transpose of the upper-left 3x3 portion
/// of the transform matrix. This is necessary for correct normal transformation
/// when non-uniform scaling is present.
///
/// If the matrix is not invertible, returns an identity matrix.
pub fn compute_normal_matrix(world_transform: &Matrix4<f32>) -> Matrix3<f32> {
    // Extract the upper-left 3x3 matrix
    let mat3 = Matrix3::new(
        world_transform[0][0],
        world_transform[0][1],
        world_transform[0][2],
        world_transform[1][0],
        world_transform[1][1],
        world_transform[1][2],
        world_transform[2][0],
        world_transform[2][1],
        world_transform[2][2],
    );

    // Compute inverse-transpose
    match mat3.invert() {
        Some(inv) => {
            // Transpose by accessing columns as rows
            Matrix3::new(
                inv[0][0], inv[1][0], inv[2][0], // First row = first column of inv
                inv[0][1], inv[1][1], inv[2][1], // Second row = second column of inv
                inv[0][2], inv[1][2], inv[2][2], // Third row = third column of inv
            )
        }
        None => {
            // If not invertible, use identity (shouldn't happen in practice)
            Matrix3::identity()
        }
    }
}

/// Recursively walks the scene tree starting from a given node.
///
/// This is a generic tree traversal function that calls visitor methods
/// at appropriate times. All domain-specific logic (transforms, etc.)
/// is delegated to the visitor implementation.
///
/// # Arguments
/// * `scene` - The scene containing all nodes
/// * `node_id` - The ID of the current node to process
/// * `visitor` - The visitor implementation to call for each node
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
///
/// This visitor maintains a stack of world transforms as it walks the tree,
/// computing transforms incrementally and caching them in nodes. It optimizes
/// by using cached transforms when neither the node nor any parent is dirty.
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
            transform_stack: vec![Matrix4::identity()], // Start with identity at root
            needs_recompute_stack: vec![false], // Root level doesn't force recompute
            results: Vec::new(),
        }
    }

    /// Consumes the collector and returns the collected instance transforms.
    pub fn into_results(self) -> Vec<InstanceTransform> {
        self.results
    }

    /// Gets the current parent transform from the top of the stack.
    fn current_parent_transform(&self) -> Matrix4<f32> {
        *self.transform_stack.last().unwrap()
    }

    /// Gets whether any parent forced a recomputation.
    fn parent_changed(&self) -> bool {
        *self.needs_recompute_stack.last().unwrap()
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
///
/// This starts from all root nodes (nodes with no parent) and recursively
/// processes the entire tree, computing world transforms along the way.
pub fn collect_instance_transforms(scene: &Scene) -> Vec<InstanceTransform> {
    let mut visitor = InstanceTransformCollector::new();

    // Process each root node
    for &root_id in scene.root_nodes() {
        walk_tree_recursive(scene, root_id, &mut visitor);
    }

    visitor.into_results()
}
