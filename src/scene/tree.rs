use crate::{common, scene::Node};
use super::{InstanceId, NodeId, Scene};

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
#[derive(Clone)]
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
    visitor.enter_node(node);

    // Recurse for all children
    for &child_id in node.children() {
        walk_tree(scene, child_id, visitor);
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
        let node_dirty = node.transform_dirty();
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
            node.cached_world_transform().unwrap()
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
        walk_tree(scene, root_id, &mut visitor);
    }

    visitor.into_results()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, Vector3, SquareMatrix, Deg, Rotation3, Quaternion};
    use crate::common::EPSILON;

    // ========================================================================
    // InstanceTransform Tests
    // ========================================================================

    #[test]
    fn test_instance_transform_creation() {
        let transform = Matrix4::from_scale(2.0);
        let instance_transform = InstanceTransform::new(42, transform);

        assert_eq!(instance_transform.instance_id, 42);
        assert_eq!(instance_transform.world_transform, transform);
    }

    #[test]
    fn test_instance_transform_identity() {
        let identity = Matrix4::identity();
        let instance_transform = InstanceTransform::new(1, identity);

        // Verify identity transform
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(instance_transform.world_transform[i][j], 1.0);
                } else {
                    assert_eq!(instance_transform.world_transform[i][j], 0.0);
                }
            }
        }

        // Verify identity normal matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(instance_transform.normal_matrix[i][j], 1.0);
                } else {
                    assert_eq!(instance_transform.normal_matrix[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_instance_transform_normal_matrix_computed() {
        let transform = Matrix4::from_scale(2.0);
        let instance_transform = InstanceTransform::new(1, transform);

        // Normal matrix should be inverse-transpose
        // For uniform scale of 2.0, normal matrix should be 0.5
        assert!((instance_transform.normal_matrix[0][0] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[1][1] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[2][2] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_translation() {
        let transform = Matrix4::from_translation(Vector3::new(5.0, 10.0, 15.0));
        let instance_transform = InstanceTransform::new(1, transform);

        // Translation should be in the transform
        assert_eq!(instance_transform.world_transform[3][0], 5.0);
        assert_eq!(instance_transform.world_transform[3][1], 10.0);
        assert_eq!(instance_transform.world_transform[3][2], 15.0);

        // Normal matrix should remain identity (translation doesn't affect normals)
        assert_eq!(instance_transform.normal_matrix[0][0], 1.0);
        assert_eq!(instance_transform.normal_matrix[1][1], 1.0);
        assert_eq!(instance_transform.normal_matrix[2][2], 1.0);
    }

    #[test]
    fn test_instance_transform_rotation() {
        let rotation = Quaternion::from_angle_z(Deg(90.0));
        let transform = Matrix4::from(rotation);
        let instance_transform = InstanceTransform::new(1, transform);

        // Normal matrix should match rotation (orthogonal matrices)
        // For 90 degree Z rotation: (1,0,0) -> (0,1,0)
        let normal = instance_transform.normal_matrix;

        // Check that applying normal matrix to (1,0,0) gives approximately (0,1,0)
        let x = normal[0][0] * 1.0 + normal[1][0] * 0.0 + normal[2][0] * 0.0;
        let y = normal[0][1] * 1.0 + normal[1][1] * 0.0 + normal[2][1] * 0.0;

        assert!(x.abs() < EPSILON);
        assert!((y - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_non_uniform_scale() {
        let transform = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let instance_transform = InstanceTransform::new(1, transform);

        // Normal matrix should handle non-uniform scale correctly
        // Inverse of diagonal matrix (2,3,4) is (0.5, 0.333..., 0.25)
        assert!((instance_transform.normal_matrix[0][0] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[1][1] - 1.0/3.0).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[2][2] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_different_instances_different_ids() {
        let transform = Matrix4::identity();
        let instance1 = InstanceTransform::new(1, transform);
        let instance2 = InstanceTransform::new(2, transform);
        let instance3 = InstanceTransform::new(3, transform);

        assert_ne!(instance1.instance_id, instance2.instance_id);
        assert_ne!(instance1.instance_id, instance3.instance_id);
        assert_ne!(instance2.instance_id, instance3.instance_id);
    }

    #[test]
    fn test_instance_transform_with_complex_transform() {
        // Combine translation, rotation, and scale
        let translation = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0));
        let rotation = Matrix4::from(Quaternion::from_angle_y(Deg(45.0)));
        let scale = Matrix4::from_scale(2.0);
        let transform = translation * rotation * scale;

        let instance_transform = InstanceTransform::new(42, transform);

        assert_eq!(instance_transform.instance_id, 42);
        assert_eq!(instance_transform.world_transform, transform);

        // Verify normal matrix was computed (not identity)
        let expected_normal = common::compute_normal_matrix(&transform);

        for i in 0..3 {
            for j in 0..3 {
                assert!((instance_transform.normal_matrix[i][j] - expected_normal[i][j]).abs() < EPSILON);
            }
        }
    }
}
