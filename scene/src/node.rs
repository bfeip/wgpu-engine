use super::InstanceId;
use crate::common::{
    Aabb, apply_scale, compose_rotation, local_axes, local_axis_x, local_axis_y, local_axis_z,
    rotate_position_about_pivot, scale_position_about_pivot_local, scale_position_about_pivot_world,
};
use cgmath::{EuclideanSpace, Matrix4, Point3, Quaternion, Vector3};
use std::cell::Cell;

/// Unique identifier for a Node in the scene tree.
pub type NodeId = u32;

/// Explicit visibility state set by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Node is explicitly set to visible
    Visible,
    /// Node is explicitly set to invisible
    Invisible,
}

impl Default for Visibility {
    fn default() -> Self {
        Self::Visible
    }
}

/// Effective visibility state computed during tree traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveVisibility {
    /// Node and all descendants are visible
    Visible,
    /// Node is explicitly invisible
    Invisible,
    /// Node is visible but has some invisible descendants
    Mixed,
}

/// A node in the scene tree hierarchy.
#[derive(Clone)]
pub struct Node {
    pub id: NodeId,
    pub name: Option<String>,

    // Local transform components
    position: Point3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,

    // Hierarchy
    parent: Option<NodeId>,
    children: Vec<NodeId>,

    // Content: This node can reference an instance to be rendered
    instance: Option<InstanceId>,

    // Visibility
    visibility: Visibility,
    cached_effective_visibility: Cell<Option<EffectiveVisibility>>,

    // Cached computed values (for optimization)
    cached_world_transform: Cell<Option<Matrix4<f32>>>,
    cached_bounds: Cell<Option<Aabb>>,
}

impl Node {
    /// Creates a new node with the given transform components.
    pub fn new(
        id: NodeId,
        name: Option<String>,
        position: Point3<f32>,
        rotation: Quaternion<f32>,
        scale: Vector3<f32>,
    ) -> Self {
        Self {
            id,
            name,
            position,
            rotation,
            scale,
            parent: None,
            children: Vec::new(),
            instance: None,
            visibility: Visibility::default(),
            cached_effective_visibility: Cell::new(None),
            cached_world_transform: Cell::new(None),
            cached_bounds: Cell::new(None),
        }
    }

    /// Creates a new node with default transform (identity).
    pub fn new_default(id: NodeId) -> Self {
        Self::new(
            id,
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0), // Identity quaternion
            Vector3::new(1.0, 1.0, 1.0),
        )
    }

    /// Computes the local transform matrix from position, rotation, and scale.
    ///
    /// The order of operations is: Translation * Rotation * Scale (TRS)
    pub fn compute_local_transform(&self) -> Matrix4<f32> {
        let translation = Matrix4::from_translation(self.position.to_vec());
        let rotation = Matrix4::from(self.rotation);
        let scale = Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);

        translation * rotation * scale
    }

    // Getters and setters for transform components

    pub fn position(&self) -> Point3<f32> {
        self.position
    }

    pub fn set_position(&mut self, position: Point3<f32>) {
        self.position = position;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    pub fn rotation(&self) -> Quaternion<f32> {
        self.rotation
    }

    pub fn set_rotation(&mut self, rotation: Quaternion<f32>) {
        self.rotation = rotation;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    pub fn scale(&self) -> Vector3<f32> {
        self.scale
    }

    pub fn set_scale(&mut self, scale: Vector3<f32>) {
        self.scale = scale;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    // Hierarchy management

    /// Gets the parent node ID.
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Sets the parent node ID (internal use only - use Scene methods to maintain consistency).
    pub(super) fn set_parent(&mut self, parent: Option<NodeId>) {
        self.parent = parent;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    /// Gets the list of child node IDs.
    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    /// Adds a child node ID to this node's children list (internal use only - use Scene methods to maintain consistency).
    pub(super) fn add_child(&mut self, child: NodeId) {
        if !self.children.contains(&child) {
            self.children.push(child);
            self.mark_bounds_dirty();
        }
    }

    /// Removes a child node ID from this node's children list (internal use only - use Scene methods to maintain consistency).
    pub(super) fn remove_child(&mut self, child: NodeId) {
        self.children.retain(|&id| id != child);
        self.mark_bounds_dirty();
    }

    // Instance reference

    pub fn instance(&self) -> Option<InstanceId> {
        self.instance
    }

    pub fn set_instance(&mut self, instance: Option<InstanceId>) {
        self.instance = instance;
        // Only invalidate bounds, not transform (instance doesn't affect transform)
        self.cached_bounds.set(None);
    }

    /// Marks this node's world transform as dirty (needs recomputation).
    /// Note: This only marks this node, not descendants. The Scene is responsible
    /// for propagating dirty flags to children.
    pub(super) fn mark_transform_dirty(&self) {
        self.cached_world_transform.set(None);
    }

    /// Marks this node's bounds as dirty (needs recomputation).
    /// Note: This only marks this node, not descendants.
    pub(super) fn mark_bounds_dirty(&self) {
        self.cached_bounds.set(None);
    }

    pub fn transform_dirty(&self) -> bool {
        self.cached_world_transform.get().is_none()
    }

    pub fn bounds_dirty(&self) -> bool {
        self.cached_bounds.get().is_none()
    }

    /// Gets the cached world transform if valid
    /// You probably want [crate::Scene::nodes_transform]
    pub fn cached_world_transform(&self) -> Option<Matrix4<f32>> {
        self.cached_world_transform.get()
    }

    /// Sets the cached world transform
    pub fn set_cached_world_transform(&self, transform: Matrix4<f32>) {
        self.cached_world_transform.set(Some(transform));
    }

    /// Gets the cached bounding box if valid
    /// You probably want [crate::Scene::nodes_bounding]
    pub(super) fn cached_bounds(&self) -> Option<Aabb> {
        self.cached_bounds.get()
    }

    /// Sets the cached bounding box
    pub(super) fn set_cached_bounds(&self, bounds: Option<Aabb>) {
        self.cached_bounds.set(bounds);
    }

    // ========== Visibility Management ==========

    /// Gets the explicit visibility state of this node.
    pub fn visibility(&self) -> Visibility {
        self.visibility
    }

    /// Sets the explicit visibility state of this node.
    pub fn set_visibility(&mut self, visibility: Visibility) {
        self.visibility = visibility;
        self.cached_effective_visibility.set(None);
    }

    /// Gets the cached effective visibility if valid.
    pub(super) fn cached_effective_visibility(&self) -> Option<EffectiveVisibility> {
        self.cached_effective_visibility.get()
    }

    /// Sets the cached effective visibility.
    pub(super) fn set_cached_effective_visibility(&self, visibility: EffectiveVisibility) {
        self.cached_effective_visibility.set(Some(visibility));
    }

    /// Returns true if effective visibility needs recomputation.
    pub fn effective_visibility_dirty(&self) -> bool {
        self.cached_effective_visibility.get().is_none()
    }

    /// Marks visibility cache as dirty.
    pub fn mark_visibility_dirty(&self) {
        self.cached_effective_visibility.set(None);
    }

    // ========== Transform Manipulation Methods ==========

    /// Rotates the node's position and orientation around a world-space pivot point.
    ///
    /// # Arguments
    /// * `pivot` - The world-space pivot point
    /// * `rotation` - The rotation to apply
    pub fn rotate_about_pivot(&mut self, pivot: Point3<f32>, rotation: Quaternion<f32>) {
        let new_position = rotate_position_about_pivot(self.position, pivot, rotation);
        let new_rotation = compose_rotation(self.rotation, rotation);
        self.set_position(new_position);
        self.set_rotation(new_rotation);
    }

    /// Scales the node's position and scale relative to a world-space pivot point.
    ///
    /// # Arguments
    /// * `pivot` - The world-space pivot point
    /// * `scale_factor` - The scale factors (x, y, z)
    pub fn scale_about_pivot(&mut self, pivot: Point3<f32>, scale_factor: Vector3<f32>) {
        let new_position = scale_position_about_pivot_world(self.position, pivot, scale_factor);
        let new_scale = apply_scale(self.scale, scale_factor);
        self.set_position(new_position);
        self.set_scale(new_scale);
    }

    /// Scales the node's position and scale relative to a pivot point in local space.
    ///
    /// The local space is defined by the given orientation.
    ///
    /// # Arguments
    /// * `pivot` - The world-space pivot point
    /// * `scale_factor` - The scale factors in local space (x, y, z)
    /// * `local_orientation` - The orientation defining local space
    pub fn scale_about_pivot_local(
        &mut self,
        pivot: Point3<f32>,
        scale_factor: Vector3<f32>,
        local_orientation: Quaternion<f32>,
    ) {
        let new_position =
            scale_position_about_pivot_local(self.position, pivot, scale_factor, local_orientation);
        let new_scale = apply_scale(self.scale, scale_factor);
        self.set_position(new_position);
        self.set_scale(new_scale);
    }

    /// Translates the node by the given offset.
    ///
    /// # Arguments
    /// * `offset` - The translation offset in world space
    pub fn translate(&mut self, offset: Vector3<f32>) {
        self.set_position(self.position + offset);
    }

    /// Returns the local X axis (right) in world space.
    pub fn local_x_axis(&self) -> Vector3<f32> {
        local_axis_x(self.rotation)
    }

    /// Returns the local Y axis (up) in world space.
    pub fn local_y_axis(&self) -> Vector3<f32> {
        local_axis_y(self.rotation)
    }

    /// Returns the local Z axis (forward) in world space.
    pub fn local_z_axis(&self) -> Vector3<f32> {
        local_axis_z(self.rotation)
    }

    /// Returns all local axes (right, up, forward) in world space.
    pub fn local_axes(&self) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        local_axes(self.rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::EPSILON;
    use cgmath::{Deg, Quaternion, Rotation3, Vector3, EuclideanSpace};

    // ========================================================================
    // Node Creation Tests
    // ========================================================================

    #[test]
    fn test_node_new() {
        let position = Point3::new(1.0, 2.0, 3.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(2.0, 2.0, 2.0);

        let node = Node::new(42, None, position, rotation, scale);

        assert_eq!(node.id, 42);
        assert_eq!(node.position(), position);
        assert_eq!(node.rotation(), rotation);
        assert_eq!(node.scale(), scale);
    }

    #[test]
    fn test_node_default_values() {
        let node = Node::new_default(7);

        assert_eq!(node.id, 7);
        assert_eq!(node.name, None);
        assert_eq!(node.parent(), None);
        assert_eq!(node.children().len(), 0);
        assert_eq!(node.instance(), None);
    }

    #[test]
    fn test_node_local_transform_identity() {
        let node = Node::new_default(0);
        let transform = node.compute_local_transform();

        // Identity transform should be close to Matrix4::from_scale(1.0)
        let identity = Matrix4::from_scale(1.0);

        // Check each element is close to identity
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (transform[i][j] - identity[i][j]).abs() < EPSILON,
                    "Transform element [{i}][{j}] = {}, expected {}",
                    transform[i][j],
                    identity[i][j]
                );
            }
        }
    }

    // ========================================================================
    // Node Transform Tests
    // ========================================================================

    #[test]
    fn test_compute_local_transform_translation_only() {
        let position = Point3::new(5.0, 10.0, 15.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity
        let scale = Vector3::new(1.0, 1.0, 1.0); // Unity scale

        let node = Node::new(0, None, position, rotation, scale);
        let transform = node.compute_local_transform();

        // Check translation components (last column)
        assert!((transform[3][0] - 5.0).abs() < EPSILON);
        assert!((transform[3][1] - 10.0).abs() < EPSILON);
        assert!((transform[3][2] - 15.0).abs() < EPSILON);
        assert!((transform[3][3] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_compute_local_transform_rotation_only() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::from_angle_z(Deg(90.0)); // 90 degrees around Z
        let scale = Vector3::new(1.0, 1.0, 1.0);

        let node = Node::new(0, None, position, rotation, scale);
        let transform = node.compute_local_transform();

        // Apply transform to point (1, 0, 0) - should become roughly (0, 1, 0)
        let point = Vector3::new(1.0, 0.0, 0.0);
        let rotated_x = transform[0][0] * point.x + transform[1][0] * point.y + transform[2][0] * point.z;
        let rotated_y = transform[0][1] * point.x + transform[1][1] * point.y + transform[2][1] * point.z;
        let rotated_z = transform[0][2] * point.x + transform[1][2] * point.y + transform[2][2] * point.z;

        assert!(rotated_x.abs() < EPSILON, "Expected x ≈ 0, got {}", rotated_x);
        assert!((rotated_y - 1.0).abs() < EPSILON, "Expected y ≈ 1, got {}", rotated_y);
        assert!(rotated_z.abs() < EPSILON, "Expected z ≈ 0, got {}", rotated_z);
    }

    #[test]
    fn test_compute_local_transform_scale_only() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(2.0, 3.0, 4.0);

        let node = Node::new(0, None, position, rotation, scale);
        let transform = node.compute_local_transform();

        // Check diagonal elements (scale factors)
        assert!((transform[0][0] - 2.0).abs() < EPSILON);
        assert!((transform[1][1] - 3.0).abs() < EPSILON);
        assert!((transform[2][2] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_compute_local_transform_trs_composition() {
        // Translation × Rotation × Scale
        let position = Point3::new(10.0, 20.0, 30.0);
        let rotation = Quaternion::from_angle_y(Deg(45.0));
        let scale = Vector3::new(2.0, 2.0, 2.0);

        let node = Node::new(0, None, position, rotation, scale);
        let transform = node.compute_local_transform();

        // Manually compute expected transform
        let translation_matrix = Matrix4::from_translation(position.to_vec());
        let rotation_matrix = Matrix4::from(rotation);
        let scale_matrix = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);
        let expected = translation_matrix * rotation_matrix * scale_matrix;

        // Compare all elements
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (transform[i][j] - expected[i][j]).abs() < EPSILON,
                    "Transform element [{i}][{j}] = {}, expected {}",
                    transform[i][j],
                    expected[i][j]
                );
            }
        }
    }

    // ========================================================================
    // Node Hierarchy Tests
    // ========================================================================

    #[test]
    fn test_set_parent() {
        let mut node = Node::new_default(1);
        assert_eq!(node.parent(), None);

        node.set_parent(Some(10));
        assert_eq!(node.parent(), Some(10));

        node.set_parent(None);
        assert_eq!(node.parent(), None);
    }

    #[test]
    fn test_add_child() {
        let mut parent = Node::new_default(1);
        let mut child1 = Node::new_default(5);
        let mut child2 = Node::new_default(7);

        assert_eq!(parent.children().len(), 0);
        assert_eq!(child1.parent(), None);
        assert_eq!(child2.parent(), None);

        // Add first child (bidirectional setup)
        parent.add_child(5);
        child1.set_parent(Some(1));

        assert_eq!(parent.children().len(), 1);
        assert_eq!(parent.children()[0], 5);
        assert_eq!(child1.parent(), Some(1));

        // Add second child (bidirectional setup)
        parent.add_child(7);
        child2.set_parent(Some(1));

        assert_eq!(parent.children().len(), 2);
        assert!(parent.children().contains(&5));
        assert!(parent.children().contains(&7));
        assert_eq!(child1.parent(), Some(1));
        assert_eq!(child2.parent(), Some(1));
    }

    #[test]
    fn test_add_child_duplicate_ignored() {
        let mut node = Node::new_default(1);

        node.add_child(5);
        node.add_child(5); // Duplicate
        node.add_child(5); // Duplicate

        // Should only have one child
        assert_eq!(node.children().len(), 1);
        assert_eq!(node.children()[0], 5);
    }

    #[test]
    fn test_remove_child() {
        let mut node = Node::new_default(1);

        node.add_child(5);
        node.add_child(10);
        node.add_child(15);

        assert_eq!(node.children().len(), 3);

        node.remove_child(10);
        assert_eq!(node.children().len(), 2);
        assert!(node.children().contains(&5));
        assert!(!node.children().contains(&10));
        assert!(node.children().contains(&15));
    }

    #[test]
    fn test_remove_child_nonexistent() {
        let mut node = Node::new_default(1);
        node.add_child(5);

        // Removing non-existent child should not panic
        node.remove_child(999);
        assert_eq!(node.children().len(), 1);
        assert_eq!(node.children()[0], 5);
    }

    // ========================================================================
    // Node Cache Tests
    // ========================================================================

    #[test]
    fn test_cached_world_transform_initially_none() {
        let node = Node::new_default(0);
        assert!(node.cached_world_transform().is_none());
    }

    #[test]
    fn test_set_cached_world_transform() {
        let node = Node::new_default(0);
        let transform = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));

        node.set_cached_world_transform(transform);

        let cached = node.cached_world_transform();
        assert!(cached.is_some());

        let cached_transform = cached.unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(cached_transform[i][j], transform[i][j]);
            }
        }
    }

    #[test]
    fn test_cached_world_transform_retrieval() {
        let node = Node::new_default(0);

        // Initially None
        assert!(node.cached_world_transform().is_none());

        // Set and retrieve
        let transform1 = Matrix4::from_scale(2.0);
        node.set_cached_world_transform(transform1);
        assert!(node.cached_world_transform().is_some());

        // Set different value
        let transform2 = Matrix4::from_scale(3.0);
        node.set_cached_world_transform(transform2);
        let cached = node.cached_world_transform().unwrap();
        assert!((cached[0][0] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_cached_bounds_initially_none() {
        let node = Node::new_default(0);
        assert!(node.cached_bounds().is_none());
    }

    #[test]
    fn test_set_cached_bounds() {
        let node = Node::new_default(0);
        let bounds = Aabb {
            min: Point3::new(-1.0, -1.0, -1.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };

        node.set_cached_bounds(Some(bounds));

        let cached = node.cached_bounds();
        assert!(cached.is_some());

        let cached_bounds = cached.unwrap();
        assert_eq!(cached_bounds.min, bounds.min);
        assert_eq!(cached_bounds.max, bounds.max);
    }

    #[test]
    fn test_cached_bounds_retrieval() {
        let node = Node::new_default(0);

        // Initially None
        assert!(node.cached_bounds().is_none());

        // Set and retrieve
        let bounds = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(10.0, 10.0, 10.0),
        };
        node.set_cached_bounds(Some(bounds));
        assert!(node.cached_bounds().is_some());

        // Can set to None
        node.set_cached_bounds(None);
        assert!(node.cached_bounds().is_none());
    }

    #[test]
    fn test_bounds_dirty_flag() {
        let node = Node::new_default(0);

        // Initially dirty (no cache)
        assert!(node.bounds_dirty());

        // Set cache - no longer dirty
        let bounds = Aabb {
            min: Point3::new(-1.0, -1.0, -1.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };
        node.set_cached_bounds(Some(bounds));
        assert!(!node.bounds_dirty());

        // Mark bounds dirty
        node.mark_bounds_dirty();
        assert!(node.bounds_dirty());
    }

    #[test]
    fn test_transform_dirty_flag() {
        let node = Node::new_default(0);

        // Initially dirty (no cache)
        assert!(node.transform_dirty());

        // Set cache - no longer dirty
        let transform = Matrix4::from_scale(1.0);
        node.set_cached_world_transform(transform);
        assert!(!node.transform_dirty());

        // Mark transform dirty
        node.mark_transform_dirty();
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_dirty_flags_are_independent() {
        let node = Node::new_default(0);

        // Set both caches
        let transform = Matrix4::from_scale(1.0);
        node.set_cached_world_transform(transform);

        let bounds = Aabb {
            min: Point3::new(-1.0, -1.0, -1.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };
        node.set_cached_bounds(Some(bounds));

        assert!(!node.transform_dirty());
        assert!(!node.bounds_dirty());

        // Mark only transform dirty
        node.mark_transform_dirty();
        assert!(node.transform_dirty());
        assert!(!node.bounds_dirty());

        // Reset transform cache
        node.set_cached_world_transform(transform);
        assert!(!node.transform_dirty());

        // Mark only bounds dirty
        node.mark_bounds_dirty();
        assert!(!node.transform_dirty());
        assert!(node.bounds_dirty());
    }

    #[test]
    fn test_set_position_marks_dirty() {
        let mut node = Node::new_default(0);

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change position should mark dirty
        node.set_position(Point3::new(5.0, 5.0, 5.0));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_set_rotation_marks_dirty() {
        let mut node = Node::new_default(0);

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change rotation should mark dirty
        node.set_rotation(Quaternion::from_angle_z(Deg(45.0)));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_set_scale_marks_dirty() {
        let mut node = Node::new_default(0);

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change scale should mark dirty
        node.set_scale(Vector3::new(2.0, 2.0, 2.0));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_set_parent_marks_dirty() {
        let mut node = Node::new_default(0);

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change parent should mark dirty
        node.set_parent(Some(10));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_add_child_marks_bounds_dirty() {
        let mut node = Node::new_default(0);

        // Set both caches
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        node.set_cached_bounds(Some(Aabb::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.)
        )));
        assert!(!node.transform_dirty());
        assert!(!node.bounds_dirty());

        // Add child should only mark bounds dirty, not transform
        node.add_child(2);
        assert!(!node.transform_dirty());
        assert!(node.bounds_dirty());
    }

    // ========================================================================
    // Node Instance Tests
    // ========================================================================

    #[test]
    fn test_instance_none_by_default() {
        let node = Node::new_default(0);
        assert_eq!(node.instance(), None);
    }

    #[test]
    fn test_set_instance() {
        let mut node = Node::new_default(0);

        node.set_instance(Some(42));
        assert_eq!(node.instance(), Some(42));

        node.set_instance(Some(99));
        assert_eq!(node.instance(), Some(99));
    }

    #[test]
    fn test_instance_retrieval() {
        let mut node = Node::new_default(0);

        // Initially None
        assert_eq!(node.instance(), None);

        // Set to Some value
        node.set_instance(Some(123));
        assert_eq!(node.instance(), Some(123));

        // Set back to None
        node.set_instance(None);
        assert_eq!(node.instance(), None);
    }

    #[test]
    fn test_set_instance_marks_dirty() {
        let mut node = Node::new_default(0);

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        node.set_cached_bounds(Some(Aabb::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.)
        )));
        assert!(!node.transform_dirty());

        node.set_instance(Some(42));
        assert!(!node.transform_dirty()); // instance doesn't affect transform
        assert!(node.bounds_dirty()); // instance affects bounds
    }

    // ========================================================================
    // Visibility Tests
    // ========================================================================

    #[test]
    fn test_visibility_default() {
        let node = Node::new_default(0);
        assert_eq!(node.visibility(), Visibility::Visible);
    }

    #[test]
    fn test_set_visibility() {
        let mut node = Node::new_default(0);

        // Default is visible
        assert_eq!(node.visibility(), Visibility::Visible);

        // Set to invisible
        node.set_visibility(Visibility::Invisible);
        assert_eq!(node.visibility(), Visibility::Invisible);

        // Set back to visible
        node.set_visibility(Visibility::Visible);
        assert_eq!(node.visibility(), Visibility::Visible);
    }

    #[test]
    fn test_visibility_cache_invalidation() {
        let mut node = Node::new_default(0);

        // Set some cached effective visibility
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert!(!node.effective_visibility_dirty());

        // Setting visibility should invalidate cache
        node.set_visibility(Visibility::Invisible);
        assert!(node.effective_visibility_dirty());
    }

    #[test]
    fn test_effective_visibility_dirty() {
        let node = Node::new_default(0);

        // Initially dirty (no cache)
        assert!(node.effective_visibility_dirty());

        // Set cache - no longer dirty
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert!(!node.effective_visibility_dirty());

        // Mark dirty
        node.mark_visibility_dirty();
        assert!(node.effective_visibility_dirty());
    }

    #[test]
    fn test_cached_effective_visibility() {
        let node = Node::new_default(0);

        // Initially None
        assert_eq!(node.cached_effective_visibility(), None);

        // Set to Visible
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Visible));

        // Set to Invisible
        node.set_cached_effective_visibility(EffectiveVisibility::Invisible);
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Invisible));

        // Set to Mixed
        node.set_cached_effective_visibility(EffectiveVisibility::Mixed);
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Mixed));
    }

    #[test]
    fn test_visibility_independent_from_transform_cache() {
        let mut node = Node::new_default(0);

        // Set transform cache
        let transform = Matrix4::from_scale(2.0);
        node.set_cached_world_transform(transform);
        assert!(!node.transform_dirty());

        // Setting visibility should NOT invalidate transform cache
        node.set_visibility(Visibility::Invisible);
        assert!(!node.transform_dirty());
        assert!(node.effective_visibility_dirty()); // but should invalidate visibility cache
    }

    #[test]
    fn test_mark_dirty_does_not_affect_visibility_cache() {
        let node = Node::new_default(0);

        // Set visibility cache
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert!(!node.effective_visibility_dirty());

        // Mark transform/bounds dirty
        node.mark_bounds_dirty();

        // Visibility cache should still be valid
        assert!(!node.effective_visibility_dirty());
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Visible));
    }

    // ========================================================================
    // Additional Edge Case Tests
    // ========================================================================

    #[test]
    fn test_node_with_name() {
        let mut node = Node::new_default(0);
        assert_eq!(node.name, None);

        node.name = Some("TestNode".to_string());
        assert_eq!(node.name, Some("TestNode".to_string()));
    }

    #[test]
    fn test_zero_scale() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(0.0, 0.0, 0.0);

        let node = Node::new(0, None, position, rotation, scale);
        let transform = node.compute_local_transform();

        // Should produce a zero-scale transform (degenerate)
        assert_eq!(transform[0][0], 0.0);
        assert_eq!(transform[1][1], 0.0);
        assert_eq!(transform[2][2], 0.0);
    }

    #[test]
    fn test_negative_scale() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(-1.0, 1.0, 1.0); // Flip X

        let node = Node::new(0, None, position, rotation, scale);
        let transform = node.compute_local_transform();

        // X axis should be flipped
        assert!((transform[0][0] - (-1.0)).abs() < EPSILON);
        assert!((transform[1][1] - 1.0).abs() < EPSILON);
        assert!((transform[2][2] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_large_hierarchy() {
        let mut parent = Node::new_default(0);

        // Add 1000 children
        for i in 1..=1000 {
            parent.add_child(i);
        }

        assert_eq!(parent.children().len(), 1000);

        // Check all children are present
        for i in 1..=1000 {
            assert!(parent.children().contains(&i));
        }
    }
}
