mod mesh;
mod instance;
mod node;
mod tree;
mod batch;

use cgmath::{Matrix4, SquareMatrix};
pub use mesh::{Mesh, MeshDescriptor, MeshId, Vertex};
pub use instance::{Instance, InstanceId, InstanceRaw};
pub use node::{Node, NodeId};
pub use tree::{collect_instance_transforms};
pub use batch::DrawBatch;

use crate::{
    common::{Aabb},
    light::Light,
    material::MaterialId,
};
use std::collections::HashMap;

pub struct Scene {
    pub meshes: HashMap<MeshId, Mesh>,
    pub instances: HashMap<InstanceId, Instance>,
    pub lights: Vec<Light>,

    // Scene tree
    pub nodes: HashMap<NodeId, Node>,
    pub root_nodes: Vec<NodeId>,

    next_mesh_id: MeshId,
    next_instance_id: InstanceId,
    next_node_id: NodeId,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            instances: HashMap::new(),
            lights: Vec::new(),

            nodes: HashMap::new(),
            root_nodes: Vec::new(),

            next_mesh_id: 0,
            next_instance_id: 0,
            next_node_id: 0,
        }
    }

    /// Gets a reference to a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Gets a mutable reference to a node by ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Returns a slice of root node IDs.
    pub fn root_nodes(&self) -> &[NodeId] {
        &self.root_nodes
    }

    /// Collects all instances grouped into batches by mesh and material.
    ///
    /// This walks the scene tree, computes world transforms, and groups
    /// instances that share the same mesh and material into batches.
    /// Batches are sorted to minimize state changes during rendering:
    /// 1. By material ID (to minimize bind group changes)
    /// 2. By mesh ID (for GPU cache locality)
    pub fn collect_draw_batches(&self) -> Vec<DrawBatch> {
        use std::collections::HashMap;

        // Walk the tree to get all instance transforms
        let instance_transforms = collect_instance_transforms(self);

        // Group instances by (mesh_id, material_id)
        let mut batch_map: HashMap<(MeshId, MaterialId), DrawBatch> = HashMap::new();

        for inst_transform in instance_transforms {
            // Look up the instance to get its mesh and material
            if let Some(instance) = self.instances.get(&inst_transform.instance_id) {
                let key = (instance.mesh, instance.material);

                // Get or create batch for this (mesh, material) combination
                let batch = batch_map.entry(key).or_insert_with(|| {
                    DrawBatch::new(instance.mesh, instance.material)
                });

                // Add this instance to the batch
                batch.add_instance(inst_transform);
            }
        }

        // Convert to Vec and sort for optimal rendering
        let mut batches: Vec<DrawBatch> = batch_map.into_values().collect();

        // Sort by material ID first (minimize bind group changes),
        // then by mesh ID (GPU cache locality)
        batches.sort_by_key(|b| (b.material_id, b.mesh_id));

        batches
    }

    pub fn add_mesh<P: AsRef<std::path::Path>>(
        &mut self,
        device: &wgpu::Device,
        desc: MeshDescriptor<P>,
        label: Option<&str>
    ) -> anyhow::Result<MeshId> {
        let id = self.next_mesh_id;
        self.next_mesh_id += 1;

        let mesh = Mesh::new(id, device, desc, label)?;
        self.meshes.insert(id, mesh);
        Ok(id)
    }

    pub fn add_instance(
        &mut self,
        mesh: MeshId,
        material: MaterialId,
    ) -> InstanceId {
        let id = self.next_instance_id;
        self.next_instance_id += 1;

        let instance = Instance::new(id, mesh, material);
        self.instances.insert(id, instance);
        id
    }

    /// Adds a new node to the scene tree.
    ///
    /// Returns the ID of the newly created node.
    pub fn add_node(
        &mut self,
        parent: Option<NodeId>,
        position: cgmath::Point3<f32>,
        rotation: cgmath::Quaternion<f32>,
        scale: cgmath::Vector3<f32>,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;

        let mut node = Node::new(id, position, rotation, scale);

        // Set up parent-child relationship
        if let Some(parent_id) = parent {
            node.set_parent(Some(parent_id));
            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                parent_node.add_child(id);
            }
        } else {
            // No parent, so this is a root node
            self.root_nodes.push(id);
        }

        self.nodes.insert(id, node);
        id
    }

    /// Adds a new node with an instance attached.
    ///
    /// This is a convenience method that creates both an instance and a node
    /// in one call. Returns the node ID.
    pub fn add_instance_node(
        &mut self,
        parent: Option<NodeId>,
        mesh: MeshId,
        material: MaterialId,
        position: cgmath::Point3<f32>,
        rotation: cgmath::Quaternion<f32>,
        scale: cgmath::Vector3<f32>,
    ) -> NodeId {
        // Create the instance
        let instance_id = self.add_instance(mesh, material);

        // Create the node
        let node_id = self.add_node(parent, position, rotation, scale);

        // Attach instance to node
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.set_instance(Some(instance_id));
        }

        node_id
    }

    /// Adds a node with default transform (identity).
    pub fn add_default_node(&mut self, parent: Option<NodeId>) -> NodeId {
        use cgmath::{Point3, Quaternion, Vector3};

        self.add_node(
            parent,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0), // Identity quaternion
            Vector3::new(1.0, 1.0, 1.0),
        )
    }

    /// Gets the world transform of a node.
    ///
    /// This returns the cached transform if valid, otherwise computes it by
    /// walking from the root to the node, computing and caching transforms
    /// along the way.
    pub fn nodes_transform(&self, node_id: NodeId) -> Matrix4<f32> {
        let node = self.get_node(node_id).expect("Node not found");

        // If cached and valid, return it
        if let Some(cached) = node.cached_world_transform() {
            return cached;
        }

        // Need to compute: build path from root to node
        let mut path = Vec::new();
        let mut current_id = node_id;

        // Walk up to root
        loop {
            path.push(current_id);
            let current = self.get_node(current_id).unwrap();
            if let Some(parent_id) = current.parent() {
                current_id = parent_id;
            } else {
                // Reached root
                break;
            }
        }

        // Reverse to get root-to-node path
        path.reverse();

        // Walk down the path, computing transforms
        let mut world_transform = Matrix4::identity();

        for &id in &path {
            let node = self.get_node(id).expect("Node not found");

            // Check if this node has cached transform
            if let Some(cached) = node.cached_world_transform() {
                world_transform = cached;
            } else {
                // Compute: world = parent_world * local
                let local_transform = node.compute_local_transform();
                world_transform = world_transform * local_transform;

                // Cache it
                node.set_cached_world_transform(world_transform);
            }
        }

        world_transform
    }

    /// Gets the world-space bounding box of a node and its subtree.
    ///
    /// This returns the cached bounds if valid, otherwise recursively computes
    /// them bottom-up for the entire subtree rooted at this node.
    ///
    /// The bounds include both the node's instance (if any) and all descendants.
    pub fn nodes_bounding(&self, node_id: NodeId) -> Option<Aabb> {
        let node = self.get_node(node_id).expect("Node not found");

        // If cached and valid, return it
        if !node.bounds_dirty() {
            return node.cached_bounds();
        }

        // Need to compute: first ensure transform is valid
        let world_transform = self.nodes_transform(node_id);

        // Recursively compute bounds for children
        let mut merged_bounds: Option<Aabb> = None;

        for &child_id in node.children() {
            if let Some(child_bounds) = self.nodes_bounding(child_id) {
                merged_bounds = match merged_bounds {
                    Some(existing) => Some(existing.merge(&child_bounds)),
                    None => Some(child_bounds),
                };
            }
        }

        // If this node has an instance, get its mesh bounds and transform to world space
        let node_bounds = if let Some(instance_id) = node.instance() {
            let instance = self.instances.get(&instance_id)
                .expect("Instance referenced by node not found in scene");
            let mesh = self.meshes.get(&instance.mesh)
                .expect("Mesh referenced by instance not found in scene");

            let local_bounds = mesh.bounding();
            let world_bounds = local_bounds.map(|bounds| {
                bounds.transform(&world_transform)
            });

            // Merge with child bounds
            match (world_bounds, merged_bounds) {
                (Some(wb), Some(cb)) => Some(wb.merge(&cb)),
                (Some(wb), None) => Some(wb),
                (None, cb) => cb,
            }
        } else {
            // Branch node - just use merged child bounds
            merged_bounds
        };

        // Cache it
        node.set_cached_bounds(node_bounds);

        node_bounds
    }
}