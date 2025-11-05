use super::{InstanceId, Node, NodeId, Scene};
use cgmath::{Matrix3, Matrix4, SquareMatrix};

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

/// Recursively walks the scene tree starting from a given node,
/// computing world transforms and collecting instances.
///
/// # Arguments
/// * `scene` - The scene containing all nodes and instances
/// * `node_id` - The ID of the current node to process
/// * `parent_transform` - The accumulated world transform from the parent
/// * `parent_changed` - Whether the parent's transform changed (forces recomputation)
/// * `results` - Vector to collect instance transforms into
pub fn walk_tree_recursive(
    scene: &Scene,
    node_id: NodeId,
    parent_transform: Matrix4<f32>,
    parent_changed: bool,
    results: &mut Vec<InstanceTransform>,
) {
    // Get the node (return early if not found)
    let node = match scene.get_node(node_id) {
        Some(n) => n,
        None => return,
    };

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

    // If this node has an instance, collect it
    if let Some(instance_id) = node.instance() {
        results.push(InstanceTransform::new(instance_id, world_transform));
    }

    // Recurse for all children (they need recomputation if this node changed)
    for &child_id in node.children() {
        walk_tree_recursive(scene, child_id, world_transform, needs_recompute, results);
    }
}

/// Walks the entire scene tree and collects all instances with their world transforms.
///
/// This starts from all root nodes (nodes with no parent) and recursively
/// processes the entire tree.
pub fn collect_instance_transforms(scene: &Scene) -> Vec<InstanceTransform> {
    let mut results = Vec::new();
    let identity = Matrix4::identity();

    // Process each root node
    for &root_id in scene.root_nodes() {
        // Check if the root node is dirty to avoid unnecessary recomputation
        let parent_changed = scene.get_node(root_id)
            .map(|node| node.is_dirty())
            .unwrap_or(true); // If node not found, force recompute

        walk_tree_recursive(scene, root_id, identity, parent_changed, &mut results);
    }

    results
}
