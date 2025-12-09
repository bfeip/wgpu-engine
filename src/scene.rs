mod batch;
mod instance;
mod mesh;
mod node;
mod tree;

use cgmath::{Matrix4, SquareMatrix};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;

pub use instance::{Instance, InstanceId};
pub use mesh::{Mesh, MeshDescriptor, MeshId, MeshPrimitive, ObjMesh, PrimitiveType, Vertex};
pub use node::{Node, NodeId};
pub use tree::TreeVisitor;
pub(crate) use batch::DrawBatch;
pub(crate) use instance::InstanceRaw;
use tree::collect_instance_transforms;

use crate::{
    common::{Aabb, RgbaColor},
    light::Light,
    material::{Material, MaterialId},
    texture::{Texture, TextureId},
};

/// The scene container holding all meshes, materials, textures, instances, nodes, and lights.
///
/// Scene provides device-free APIs for creating and managing scene objects.
///
/// # Examples
///
/// ```ignore
/// let mut scene = Scene::new();
///
/// // Add a mesh (no device needed)
/// let mesh = Mesh::from_raw(vertices, primitives);
/// let mesh_id = scene.add_mesh(mesh);
///
/// // Add a material (no device needed)
/// let material = Material::new().with_face_color(RgbaColor::RED);
/// let mat_id = scene.add_material(material);
///
/// // Create an instance node
/// let node_id = scene.add_instance_node(None, mesh_id, mat_id, position, rotation, scale);
///
/// // GPU resources are created automatically when DrawState::render() is called
/// ```
pub struct Scene {
    pub meshes: HashMap<MeshId, Mesh>,
    pub instances: HashMap<InstanceId, Instance>,
    pub lights: Vec<Light>,

    // Scene tree
    pub nodes: HashMap<NodeId, Node>,
    pub root_nodes: Vec<NodeId>,

    pub materials: HashMap<MaterialId, Material>,
    pub textures: HashMap<TextureId, Texture>,

    next_mesh_id: MeshId,
    next_instance_id: InstanceId,
    next_node_id: NodeId,
    next_material_id: MaterialId,
    next_texture_id: TextureId,
}

impl Scene {
    /// Creates a new empty scene with a default material.
    ///
    /// The scene is initialized with one default material (ID 0) that has:
    /// - Face color: Magenta (for debugging unassigned faces)
    /// - Line color: Black
    /// - Point color: Black
    pub fn new() -> Self {
        let mut scene = Self {
            meshes: HashMap::new(),
            instances: HashMap::new(),
            lights: Vec::new(),

            nodes: HashMap::new(),
            root_nodes: Vec::new(),

            materials: HashMap::new(),
            textures: HashMap::new(),

            next_mesh_id: 0,
            next_instance_id: 0,
            next_node_id: 0,
            next_material_id: 0,
            next_texture_id: 0,
        };

        // Create default material (ID 0)
        scene.add_material(
            Material::new()
                .with_face_color(RgbaColor { r: 1.0, g: 0.3, b: 1.0, a: 1.0 }) // Magenta
                .with_line_color(RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 })
                .with_point_color(RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
        );

        scene
    }

    // ========== Mesh API ==========

    /// Adds a mesh to the scene.
    ///
    /// No GPU resources are allocated. They will be created lazily during rendering.
    ///
    /// # Arguments
    /// * `mesh` - The mesh to add
    ///
    /// # Returns
    /// The unique ID assigned to this mesh
    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        let id = self.next_mesh_id;
        self.next_mesh_id += 1;

        let mut mesh = mesh;
        mesh.id = id;
        self.meshes.insert(id, mesh);
        id
    }

    /// Creates and adds a mesh from a descriptor.
    ///
    /// # Arguments
    /// * `descriptor` - Source data for the mesh
    ///
    /// # Returns
    /// The unique ID assigned to this mesh, or an error if loading fails
    pub fn add_mesh_from_descriptor(&mut self, descriptor: MeshDescriptor) -> anyhow::Result<MeshId> {
        let mesh = Mesh::from_descriptor(descriptor)?;
        Ok(self.add_mesh(mesh))
    }

    /// Gets a reference to a mesh by ID.
    pub fn get_mesh(&self, id: MeshId) -> Option<&Mesh> {
        self.meshes.get(&id)
    }

    /// Gets a mutable reference to a mesh by ID.
    pub fn get_mesh_mut(&mut self, id: MeshId) -> Option<&mut Mesh> {
        self.meshes.get_mut(&id)
    }

    // ========== Material API (device-free) ==========

    /// Adds a material to the scene.
    ///
    /// No GPU resources are allocated. They will be created lazily during rendering.
    ///
    /// # Arguments
    /// * `material` - The material to add
    ///
    /// # Returns
    /// The unique ID assigned to this material
    pub fn add_material(&mut self, material: Material) -> MaterialId {
        let id = self.next_material_id;
        self.next_material_id += 1;

        let mut material = material;
        material.id = id;
        self.materials.insert(id, material);
        id
    }

    /// Gets a reference to a material by ID.
    pub fn get_material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(&id)
    }

    /// Gets a mutable reference to a material by ID.
    pub fn get_material_mut(&mut self, id: MaterialId) -> Option<&mut Material> {
        self.materials.get_mut(&id)
    }

    // ========== Texture API (device-free) ==========

    /// Adds a texture to the scene.
    ///
    /// No GPU resources are allocated. They will be created lazily during rendering.
    ///
    /// # Arguments
    /// * `texture` - The texture to add
    ///
    /// # Returns
    /// The unique ID assigned to this texture
    pub fn add_texture(&mut self, texture: Texture) -> TextureId {
        let id = self.next_texture_id;
        self.next_texture_id += 1;

        let mut texture = texture;
        texture.id = id;
        self.textures.insert(id, texture);
        id
    }

    /// Creates and adds a texture from an image.
    ///
    /// # Arguments
    /// * `image` - The image data
    ///
    /// # Returns
    /// The unique ID assigned to this texture
    pub fn add_texture_from_image(&mut self, image: DynamicImage) -> TextureId {
        self.add_texture(Texture::from_image(image))
    }

    /// Creates and adds a texture from a file path.
    ///
    /// The image is not loaded immediately - it will be loaded lazily when first needed.
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    ///
    /// # Returns
    /// The unique ID assigned to this texture
    pub fn add_texture_from_path(&mut self, path: impl AsRef<Path>) -> TextureId {
        self.add_texture(Texture::from_path(path.as_ref()))
    }

    /// Gets a reference to a texture by ID.
    pub fn get_texture(&self, id: TextureId) -> Option<&Texture> {
        self.textures.get(&id)
    }

    /// Gets a mutable reference to a texture by ID.
    pub fn get_texture_mut(&mut self, id: TextureId) -> Option<&mut Texture> {
        self.textures.get_mut(&id)
    }

    // ========== Instance API ==========

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

    /// Collects all instances grouped into batches by mesh, material, and primitive type.
    ///
    /// This walks the scene tree, computes world transforms, and groups
    /// instances that share the same mesh, material, and primitive type into batches.
    /// Each mesh can have multiple primitive types (triangles, lines, points), so
    /// a single instance may generate multiple batches.
    /// Batches are sorted to minimize state changes during rendering:
    /// 1. By material ID (to minimize bind group changes)
    /// 2. By primitive type (to minimize pipeline changes)
    /// 3. By mesh ID (for GPU cache locality)
    pub(crate) fn collect_draw_batches(&self) -> Vec<DrawBatch> {
        use std::collections::HashMap;

        let instance_transforms = collect_instance_transforms(self);
        let mut batch_map: HashMap<(MeshId, MaterialId, PrimitiveType), DrawBatch> = HashMap::new();

        for inst_transform in instance_transforms {
            let Some(instance) = self.instances.get(&inst_transform.instance_id) else { continue };
            let Some(mesh) = self.meshes.get(&instance.mesh) else { continue };

            // Create a separate batch for each primitive type the mesh supports
            for primitive_type in [PrimitiveType::TriangleList, PrimitiveType::LineList, PrimitiveType::PointList] {
                if !mesh.has_primitive_type(primitive_type) {
                    continue;
                }

                let key = (instance.mesh, instance.material, primitive_type);
                batch_map.entry(key)
                    .or_insert_with(|| DrawBatch::new(instance.mesh, instance.material, primitive_type))
                    .add_instance(inst_transform.clone());
            }
        }

        // Convert to Vec and sort for optimal rendering
        let mut batches: Vec<DrawBatch> = batch_map.into_values().collect();
        batches.sort_by_key(|b| (b.material_id, b.primitive_type as u8, b.mesh_id));
        batches
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

    /// Removes a node and all its children from the scene tree.
    ///
    /// This recursively removes all descendant nodes and cleans up
    /// parent-child relationships. Cached bounds in all ancestor nodes
    /// are invalidated since the removed subtree affects their bounds.
    pub fn remove_node(&mut self, node_id: NodeId) {
        // Store the parent before removal so we can invalidate ancestors
        let parent = self.nodes.get(&node_id).and_then(|node| node.parent());

        // Perform the recursive removal
        self.remove_node_recursive(node_id);

        // Invalidate cached bounds for all ancestors
        if let Some(parent_id) = parent {
            self.invalidate_ancestor_bounds(parent_id);
        }
    }

    /// Recursive helper for removing a node and all its children.
    ///
    /// This does NOT invalidate ancestor bounds. The caller is responsible
    /// for invalidating bounds after the entire removal is complete.
    fn remove_node_recursive(&mut self, node_id: NodeId) {
        // Get the node to find its parent and children
        let Some(node) = self.nodes.get(&node_id) else {
            return; // Node doesn't exist
        };

        let parent = node.parent();
        let children: Vec<NodeId> = node.children().to_vec();

        // Recursively remove all children first
        for child_id in children {
            self.remove_node_recursive(child_id);
        }

        // Remove this node from its parent's children list
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                parent_node.remove_child(node_id);
            }
        } else {
            // This is a root node, remove from root_nodes list
            self.root_nodes.retain(|&id| id != node_id);
        }

        // Finally, remove the node itself
        self.nodes.remove(&node_id);
    }

    /// Invalidates cached bounds for a node and all its ancestors.
    ///
    /// Walks up the parent chain from the given node to the root,
    /// clearing cached bounds on each node. This should be called
    /// when a subtree's bounds change (e.g., after removing nodes).
    fn invalidate_ancestor_bounds(&self, node_id: NodeId) {
        let mut current_id = Some(node_id);

        while let Some(id) = current_id {
            let Some(node) = self.get_node(id) else {
                break;
            };

            // Mark node as dirty to clear cached values
            node.mark_dirty();

            // Move to parent
            current_id = node.parent();
        }
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

    /// Gets the world-space bounding box of the entire scene.
    ///
    /// Computes bounds by merging the bounds of all root nodes and their subtrees.
    /// Returns None if the scene has no geometry.
    pub fn bounding(&self) -> Option<Aabb> {
        let mut merged_bounds: Option<Aabb> = None;

        for &root_id in &self.root_nodes {
            if let Some(root_bounds) = self.nodes_bounding(root_id) {
                merged_bounds = match merged_bounds {
                    Some(existing) => Some(existing.merge(&root_bounds)),
                    None => Some(root_bounds),
                };
            }
        }

        merged_bounds
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

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Point3, Quaternion, Vector3, Matrix4, SquareMatrix};
    use crate::common::EPSILON;

    // ========================================================================
    // Scene Creation and Basic Operations
    // ========================================================================

    #[test]
    fn test_scene_new() {
        let scene = Scene::new();

        assert_eq!(scene.meshes.len(), 0);
        assert_eq!(scene.instances.len(), 0);
        assert_eq!(scene.nodes.len(), 0);
        assert_eq!(scene.root_nodes.len(), 0);
        assert_eq!(scene.lights.len(), 0);
    }

    #[test]
    fn test_add_instance() {
        let mut scene = Scene::new();

        let instance_id = scene.add_instance(10, 5);
        assert_eq!(instance_id, 0);
        assert_eq!(scene.instances.len(), 1);

        let instance = scene.instances.get(&instance_id).unwrap();
        assert_eq!(instance.mesh, 10);
        assert_eq!(instance.material, 5);
    }

    #[test]
    fn test_add_multiple_instances() {
        let mut scene = Scene::new();

        let id1 = scene.add_instance(1, 1);
        let id2 = scene.add_instance(2, 2);
        let id3 = scene.add_instance(3, 3);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
        assert_eq!(scene.instances.len(), 3);
    }

    // ========================================================================
    // Scene Tree Construction
    // ========================================================================

    #[test]
    fn test_add_root_node() {
        let mut scene = Scene::new();

        let node_id = scene.add_default_node(None);

        assert_eq!(node_id, 0);
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.root_nodes.len(), 1);
        assert_eq!(scene.root_nodes[0], node_id);

        let node = scene.get_node(node_id).unwrap();
        assert_eq!(node.parent(), None);
        assert_eq!(node.children().len(), 0);
    }

    #[test]
    fn test_add_child_node() {
        let mut scene = Scene::new();

        let root = scene.add_default_node(None);
        let child = scene.add_default_node(Some(root));

        assert_eq!(scene.nodes.len(), 2);
        assert_eq!(scene.root_nodes.len(), 1);

        // Verify parent-child relationship is bidirectional
        let root_node = scene.get_node(root).unwrap();
        assert_eq!(root_node.children().len(), 1);
        assert_eq!(root_node.children()[0], child);

        let child_node = scene.get_node(child).unwrap();
        assert_eq!(child_node.parent(), Some(root));
    }

    #[test]
    fn test_add_multiple_children() {
        let mut scene = Scene::new();

        let root = scene.add_default_node(None);
        let child1 = scene.add_default_node(Some(root));
        let child2 = scene.add_default_node(Some(root));
        let child3 = scene.add_default_node(Some(root));

        let root_node = scene.get_node(root).unwrap();
        assert_eq!(root_node.children().len(), 3);
        assert!(root_node.children().contains(&child1));
        assert!(root_node.children().contains(&child2));
        assert!(root_node.children().contains(&child3));

        // Verify all children point back to root
        assert_eq!(scene.get_node(child1).unwrap().parent(), Some(root));
        assert_eq!(scene.get_node(child2).unwrap().parent(), Some(root));
        assert_eq!(scene.get_node(child3).unwrap().parent(), Some(root));
    }

    #[test]
    fn test_deep_hierarchy() {
        let mut scene = Scene::new();

        // Create a chain: root -> child1 -> child2 -> child3
        let root = scene.add_default_node(None);
        let child1 = scene.add_default_node(Some(root));
        let child2 = scene.add_default_node(Some(child1));
        let child3 = scene.add_default_node(Some(child2));

        // Verify the chain is correct
        assert_eq!(scene.get_node(root).unwrap().parent(), None);
        assert_eq!(scene.get_node(child1).unwrap().parent(), Some(root));
        assert_eq!(scene.get_node(child2).unwrap().parent(), Some(child1));
        assert_eq!(scene.get_node(child3).unwrap().parent(), Some(child2));

        // Verify each node has the correct children
        assert_eq!(scene.get_node(root).unwrap().children().len(), 1);
        assert_eq!(scene.get_node(child1).unwrap().children().len(), 1);
        assert_eq!(scene.get_node(child2).unwrap().children().len(), 1);
        assert_eq!(scene.get_node(child3).unwrap().children().len(), 0);
    }

    #[test]
    fn test_multiple_root_nodes() {
        let mut scene = Scene::new();

        let root1 = scene.add_default_node(None);
        let root2 = scene.add_default_node(None);
        let root3 = scene.add_default_node(None);

        assert_eq!(scene.root_nodes.len(), 3);
        assert!(scene.root_nodes.contains(&root1));
        assert!(scene.root_nodes.contains(&root2));
        assert!(scene.root_nodes.contains(&root3));

        // Verify all are truly roots
        assert_eq!(scene.get_node(root1).unwrap().parent(), None);
        assert_eq!(scene.get_node(root2).unwrap().parent(), None);
        assert_eq!(scene.get_node(root3).unwrap().parent(), None);
    }

    #[test]
    fn test_complex_tree_structure() {
        let mut scene = Scene::new();

        // Create a tree:
        //       root
        //      /    \
        //    c1      c2
        //   /  \      \
        //  gc1  gc2   gc3

        let root = scene.add_default_node(None);
        let c1 = scene.add_default_node(Some(root));
        let c2 = scene.add_default_node(Some(root));
        let gc1 = scene.add_default_node(Some(c1));
        let gc2 = scene.add_default_node(Some(c1));
        let gc3 = scene.add_default_node(Some(c2));

        // Verify structure
        assert_eq!(scene.root_nodes.len(), 1);
        assert_eq!(scene.nodes.len(), 6);

        let root_node = scene.get_node(root).unwrap();
        assert_eq!(root_node.children().len(), 2);

        let c1_node = scene.get_node(c1).unwrap();
        assert_eq!(c1_node.children().len(), 2);

        let c2_node = scene.get_node(c2).unwrap();
        assert_eq!(c2_node.children().len(), 1);

        // Verify leaf nodes have no children
        assert_eq!(scene.get_node(gc1).unwrap().children().len(), 0);
        assert_eq!(scene.get_node(gc2).unwrap().children().len(), 0);
        assert_eq!(scene.get_node(gc3).unwrap().children().len(), 0);
    }

    // ========================================================================
    // Node with Instance
    // ========================================================================

    #[test]
    fn test_add_instance_node() {
        let mut scene = Scene::new();

        let node_id = scene.add_instance_node(
            None,
            10,
            5,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.instances.len(), 1);

        let node = scene.get_node(node_id).unwrap();
        assert!(node.instance().is_some());

        let instance_id = node.instance().unwrap();
        let instance = scene.instances.get(&instance_id).unwrap();
        assert_eq!(instance.mesh, 10);
        assert_eq!(instance.material, 5);
    }

    // ========================================================================
    // Transform Computation and Caching
    // ========================================================================

    #[test]
    fn test_root_node_identity_transform() {
        let mut scene = Scene::new();
        let root = scene.add_default_node(None);

        let transform = scene.nodes_transform(root);
        let identity = Matrix4::identity();

        // Convert to arrays for comparison
        let t: [[f32; 4]; 4] = transform.into();
        let i: [[f32; 4]; 4] = identity.into();

        for row in 0..4 {
            for col in 0..4 {
                assert!((t[row][col] - i[row][col]).abs() < EPSILON,
                    "Transform mismatch at [{row}][{col}]");
            }
        }
    }

    #[test]
    fn test_child_transform_accumulation() {
        let mut scene = Scene::new();

        // Root at (10, 0, 0)
        let root = scene.add_node(
            None,
            Point3::new(10.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        // Child at (5, 0, 0) relative to parent
        let child = scene.add_node(
            Some(root),
            Point3::new(5.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        let child_transform = scene.nodes_transform(child);

        // Child should be at (15, 0, 0) in world space
        assert!((child_transform[3][0] - 15.0).abs() < EPSILON);
        assert!((child_transform[3][1] - 0.0).abs() < EPSILON);
        assert!((child_transform[3][2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_transform_with_scale() {
        let mut scene = Scene::new();

        // Parent with scale 2.0
        let parent = scene.add_node(
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(2.0, 2.0, 2.0),
        );

        // Child at (1, 0, 0)
        let child = scene.add_node(
            Some(parent),
            Point3::new(1.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        let child_transform = scene.nodes_transform(child);

        // Child should be at (2, 0, 0) due to parent's scale
        assert!((child_transform[3][0] - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_transform_caching() {
        let mut scene = Scene::new();
        let root = scene.add_default_node(None);

        // First computation
        let transform1 = scene.nodes_transform(root);

        // Second computation should use cache
        let transform2 = scene.nodes_transform(root);

        // Should be identical
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(transform1[i][j], transform2[i][j]);
            }
        }

        // Verify cache is being used
        let node = scene.get_node(root).unwrap();
        assert!(!node.transform_dirty());
    }

    #[test]
    fn test_transform_invalidation_on_change() {
        let mut scene = Scene::new();
        let node_id = scene.add_default_node(None);

        // Compute transform to populate cache
        let _transform = scene.nodes_transform(node_id);

        // Modify the node
        let node = scene.get_node_mut(node_id).unwrap();
        node.set_position(Point3::new(5.0, 5.0, 5.0));

        // Cache should be dirty
        assert!(node.transform_dirty());
    }

    // ========================================================================
    // Scene Tree Consistency
    // ========================================================================

    #[test]
    fn test_tree_consistency_after_multiple_operations() {
        let mut scene = Scene::new();

        // Build a tree
        let root = scene.add_default_node(None);
        let child1 = scene.add_default_node(Some(root));
        let child2 = scene.add_default_node(Some(root));
        let grandchild = scene.add_default_node(Some(child1));

        // Verify every node can be accessed
        assert!(scene.get_node(root).is_some());
        assert!(scene.get_node(child1).is_some());
        assert!(scene.get_node(child2).is_some());
        assert!(scene.get_node(grandchild).is_some());

        // Verify every parent reference is valid
        for (_, node) in &scene.nodes {
            if let Some(parent_id) = node.parent() {
                assert!(scene.get_node(parent_id).is_some(),
                    "Node references non-existent parent");
            }
        }

        // Verify every child reference is valid
        for (_, node) in &scene.nodes {
            for &child_id in node.children() {
                assert!(scene.get_node(child_id).is_some(),
                    "Node references non-existent child");
            }
        }

        // Verify all root nodes are actually roots
        for &root_id in &scene.root_nodes {
            let node = scene.get_node(root_id).unwrap();
            assert!(node.parent().is_none(),
                "Root node has a parent");
        }
    }

    #[test]
    fn test_bidirectional_consistency() {
        let mut scene = Scene::new();

        let parent = scene.add_default_node(None);
        let child = scene.add_default_node(Some(parent));

        // Parent should list child
        let parent_node = scene.get_node(parent).unwrap();
        assert!(parent_node.children().contains(&child));

        // Child should reference parent
        let child_node = scene.get_node(child).unwrap();
        assert_eq!(child_node.parent(), Some(parent));
    }

    #[test]
    fn test_no_orphaned_nodes() {
        let mut scene = Scene::new();

        let root1 = scene.add_default_node(None);
        let root2 = scene.add_default_node(None);
        let _child1 = scene.add_default_node(Some(root1));
        let _child2 = scene.add_default_node(Some(root2));

        // All nodes should be reachable from roots
        let mut reachable = std::collections::HashSet::new();

        fn visit_tree(
            scene: &Scene,
            node_id: NodeId,
            reachable: &mut std::collections::HashSet<NodeId>
        ) {
            reachable.insert(node_id);
            let node = scene.get_node(node_id).unwrap();
            for &child_id in node.children() {
                visit_tree(scene, child_id, reachable);
            }
        }

        for &root_id in &scene.root_nodes {
            visit_tree(&scene, root_id, &mut reachable);
        }

        // All nodes should be reachable
        assert_eq!(reachable.len(), scene.nodes.len());
    }

    #[test]
    fn test_large_tree_consistency() {
        let mut scene = Scene::new();

        let root = scene.add_default_node(None);

        // Create 100 children
        let mut children = Vec::new();
        for _ in 0..100 {
            let child = scene.add_default_node(Some(root));
            children.push(child);
        }

        // Verify all children are registered
        let root_node = scene.get_node(root).unwrap();
        assert_eq!(root_node.children().len(), 100);

        // Verify all children reference the parent
        for &child_id in &children {
            let child_node = scene.get_node(child_id).unwrap();
            assert_eq!(child_node.parent(), Some(root));
        }
    }

    // ========================================================================
    // Batch Collection
    // ========================================================================

    #[test]
    fn test_collect_draw_batches_empty_scene() {
        let scene = Scene::new();
        let batches = scene.collect_draw_batches();

        assert_eq!(batches.len(), 0);
    }

    #[ignore = "Reactivate when and re-write batches are properly sorted and merged"]
    #[test]
    fn test_collect_draw_batches_sort() {
        let mut scene = Scene::new();

        // Create instances with different mesh/material combinations
        let _root = scene.add_instance_node(
            None, 2, 2,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        scene.add_instance_node(
            None, 1, 2,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        scene.add_instance_node(
            None, 1, 1,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        let batches = scene.collect_draw_batches();

        // Should be sorted by material ID first, then primitive type, then mesh ID
        assert!(batches.len() > 0);

        for i in 1..batches.len() {
            let prev = &batches[i - 1];
            let curr = &batches[i];

            // Material ID should be non-decreasing
            assert!(prev.material_id <= curr.material_id);

            // If material IDs are equal, primitive type should be non-decreasing
            if prev.material_id == curr.material_id {
                assert!((prev.primitive_type as u8) <= (curr.primitive_type as u8));

                // If material IDs and primitive types are equal, mesh ID should be non-decreasing
                if prev.primitive_type as u8 == curr.primitive_type as u8 {
                    assert!(prev.mesh_id <= curr.mesh_id);
                }
            }
        }
    }

    #[test]
    fn test_get_node_mut_allows_modification() {
        let mut scene = Scene::new();
        let node_id = scene.add_default_node(None);

        {
            let node = scene.get_node_mut(node_id).unwrap();
            node.set_position(Point3::new(10.0, 20.0, 30.0));
        }

        let node = scene.get_node(node_id).unwrap();
        assert_eq!(node.position(), Point3::new(10.0, 20.0, 30.0));
    }
}