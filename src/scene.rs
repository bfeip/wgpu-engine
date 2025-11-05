mod mesh;
mod instance;
mod node;
mod tree;
mod batch;

pub use mesh::{Mesh, MeshDescriptor, MeshId, Vertex};
pub use instance::{Instance, InstanceId, InstanceRaw};
pub use node::{Node, NodeId};
pub use tree::{collect_instance_transforms};
pub use batch::DrawBatch;

use crate::{
    common::RgbaColor,
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

    pub fn demo(device: &wgpu::Device, material_id: MaterialId) -> Self {
        use cgmath::{Point3, Quaternion, Rotation3, Vector3};

        let mut scene = Self::new();
        scene.lights = vec![
            Light::new(
                cgmath::Vector3 { x: 3., y: 3., z: 3. },
                RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
            )
        ];

        // Load a sample OBJ
        let monkey_bytes = include_bytes!("monkey.obj");
        let monkey_mesh_desc: MeshDescriptor<&str> = MeshDescriptor::Obj(mesh::ObjMesh::Bytes(monkey_bytes));
        let monkey_mesh = scene.add_mesh(device, monkey_mesh_desc, Some("monkey_mesh")).unwrap();

        // Create first instance at origin
        scene.add_instance_node(
            None,
            monkey_mesh,
            0,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::from_axis_angle(Vector3::unit_z(), cgmath::Rad(0.0)),
            Vector3::new(1.0, 1.0, 1.0),
        );

        // Create second instance with rotation
        scene.add_instance_node(
            None,
            monkey_mesh,
            material_id,
            Point3::new(2.0, 0.0, 0.0),
            Quaternion::from_angle_z(cgmath::Rad(3.14_f32)),
            Vector3::new(1.0, 1.0, 1.0),
        );

        scene
    }
}