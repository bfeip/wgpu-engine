use super::InstanceId;
use cgmath::{EuclideanSpace, Matrix4, Point3, Quaternion, Vector3};
use std::cell::Cell;

pub type NodeId = u32;

/// A node in the scene tree hierarchy.
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

    // Cached world transform (for optimization)
    world_transform: Cell<Matrix4<f32>>,
    world_transform_dirty: Cell<bool>,
}

impl Node {
    /// Creates a new node with the given transform components.
    pub fn new(
        id: NodeId,
        position: Point3<f32>,
        rotation: Quaternion<f32>,
        scale: Vector3<f32>,
    ) -> Self {
        Self {
            id,
            name: None,
            position,
            rotation,
            scale,
            parent: None,
            children: Vec::new(),
            instance: None,
            world_transform: Cell::new(Matrix4::from_scale(1.0)),
            world_transform_dirty: Cell::new(true),
        }
    }

    /// Creates a new node with default transform (identity).
    pub fn new_default(id: NodeId) -> Self {
        Self::new(
            id,
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
        self.mark_dirty();
    }

    pub fn rotation(&self) -> Quaternion<f32> {
        self.rotation
    }

    pub fn set_rotation(&mut self, rotation: Quaternion<f32>) {
        self.rotation = rotation;
        self.mark_dirty();
    }

    pub fn scale(&self) -> Vector3<f32> {
        self.scale
    }

    pub fn set_scale(&mut self, scale: Vector3<f32>) {
        self.scale = scale;
        self.mark_dirty();
    }

    // Hierarchy management

    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    pub fn set_parent(&mut self, parent: Option<NodeId>) {
        self.parent = parent;
        self.mark_dirty();
    }

    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    pub fn add_child(&mut self, child: NodeId) {
        if !self.children.contains(&child) {
            self.children.push(child);
        }
    }

    pub fn remove_child(&mut self, child: NodeId) {
        self.children.retain(|&id| id != child);
    }

    // Instance reference

    pub fn instance(&self) -> Option<InstanceId> {
        self.instance
    }

    pub fn set_instance(&mut self, instance: Option<InstanceId>) {
        self.instance = instance;
    }

    // World transform caching

    /// Marks this node's world transform as dirty (needs recomputation).
    /// Note: This only marks this node, not descendants. The Scene is responsible
    /// for propagating dirty flags to children.
    pub fn mark_dirty(&self) {
        self.world_transform_dirty.set(true);
    }

    pub fn is_dirty(&self) -> bool {
        self.world_transform_dirty.get()
    }

    pub fn cached_world_transform(&self) -> Matrix4<f32> {
        self.world_transform.get()
    }

    pub fn set_cached_world_transform(&self, transform: Matrix4<f32>) {
        self.world_transform.set(transform);
        self.world_transform_dirty.set(false);
    }
}
