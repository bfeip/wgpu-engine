mod batch;
mod instance;
mod light;
mod material;
mod mesh;
mod node;
mod texture;
mod tree;

use cgmath::{Matrix4, SquareMatrix};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;

use crate::ibl::{EnvironmentMap, EnvironmentMapId};

// Public API exports
pub use instance::{Instance, InstanceId};
pub use light::{Light, LightType, MAX_LIGHTS};
pub use material::{Material, MaterialId, DEFAULT_MATERIAL_ID};
pub use mesh::{Mesh, MeshDescriptor, MeshId, MeshPrimitive, ObjMesh, PrimitiveType, Vertex};
pub use node::{Node, NodeId};
pub use texture::{Texture, TextureId};
pub use tree::TreeVisitor;

// Crate-internal exports
pub(crate) use batch::DrawBatch;
pub(crate) use instance::InstanceRaw;
pub(crate) use light::LightsArrayUniform;
pub(crate) use material::{MaterialGpuResources, MaterialProperties};
pub(crate) use texture::GpuTexture;

use tree::collect_instance_transforms;

use crate::common::{Aabb, RgbaColor};

/// The scene container holding all meshes, materials, textures, instances, nodes, and lights.
///
/// Scene provides device-free APIs for creating and managing scene objects.
///
/// # Examples
///
/// ```
/// use wgpu_engine::scene::{Scene, Mesh, MeshPrimitive, Vertex, Material, PrimitiveType};
/// use wgpu_engine::common::RgbaColor;
/// use cgmath::{Point3, Quaternion, Vector3};
///
/// let mut scene = Scene::new();
///
/// // Add a mesh (no device needed)
/// let vertices = vec![
///     Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [0.5, 1.0, 0.0], tex_coords: [0.5, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
/// ];
/// let primitives = vec![MeshPrimitive {
///     primitive_type: PrimitiveType::TriangleList,
///     indices: vec![0, 1, 2],
/// }];
/// let mesh = Mesh::from_raw(vertices, primitives);
/// let mesh_id = scene.add_mesh(mesh);
///
/// // Add a material (no device needed)
/// let material = Material::new().with_base_color_factor(RgbaColor::RED);
/// let mat_id = scene.add_material(material);
///
/// // Create an instance node
/// let position = Point3::new(0.0, 0.0, 0.0);
/// let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// let scale = Vector3::new(1.0, 1.0, 1.0);
/// let node_id = scene.add_instance_node(None, mesh_id, mat_id, None, position, rotation, scale);
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

    // Environment maps for IBL
    pub environment_maps: HashMap<EnvironmentMapId, EnvironmentMap>,
    /// The currently active environment map for IBL lighting.
    pub active_environment_map: Option<EnvironmentMapId>,

    /// Root node for annotations, created lazily when first requested
    annotation_root_node: Option<NodeId>,

    next_mesh_id: MeshId,
    next_instance_id: InstanceId,
    next_node_id: NodeId,
    next_material_id: MaterialId,
    next_texture_id: TextureId,
    next_environment_map_id: EnvironmentMapId,
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

            environment_maps: HashMap::new(),
            active_environment_map: None,

            annotation_root_node: None,

            next_mesh_id: 0,
            next_instance_id: 0,
            next_node_id: 0,
            next_material_id: 0,
            next_texture_id: 0,
            next_environment_map_id: 0,
        };

        // Create default material (ID 0)
        scene.add_material(
            Material::new()
                .with_base_color_factor(RgbaColor::MAGENTA) // For debugging unassigned faces
                .with_line_color(RgbaColor::BLACK)
                .with_point_color(RgbaColor::BLACK),
        );

        scene
    }

    // ========== Mesh API ==========

    /// Adds a mesh to the scene.
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

    // ========== Environment Map API (IBL) ==========

    /// Adds an environment map to the scene.
    ///
    /// # Arguments
    /// * `environment_map` - The environment map to add
    ///
    /// # Returns
    /// The unique ID assigned to this environment map
    pub fn add_environment_map(&mut self, environment_map: EnvironmentMap) -> EnvironmentMapId {
        let id = self.next_environment_map_id;
        self.next_environment_map_id += 1;

        let mut environment_map = environment_map;
        environment_map.id = id;
        self.environment_maps.insert(id, environment_map);
        id
    }

    /// Gets a reference to an environment map by ID.
    pub fn get_environment_map(&self, id: EnvironmentMapId) -> Option<&EnvironmentMap> {
        self.environment_maps.get(&id)
    }

    /// Gets a mutable reference to an environment map by ID.
    pub fn get_environment_map_mut(&mut self, id: EnvironmentMapId) -> Option<&mut EnvironmentMap> {
        self.environment_maps.get_mut(&id)
    }

    /// Sets the active environment map for IBL lighting.
    ///
    /// Pass `None` to disable IBL lighting.
    pub fn set_active_environment_map(&mut self, id: Option<EnvironmentMapId>) {
        self.active_environment_map = id;
    }

    /// Gets the currently active environment map ID, if any.
    pub fn active_environment_map(&self) -> Option<EnvironmentMapId> {
        self.active_environment_map
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
    /// # Arguments
    /// * `parent` - Optional parent node ID. If `Some`, the parent must exist in the scene.
    /// * `position` - Local position of the node.
    /// * `rotation` - Local rotation of the node.
    /// * `scale` - Local scale of the node.
    ///
    /// # Returns
    /// The ID of the newly created node, or an error if the parent doesn't exist.
    ///
    /// # Errors
    /// Returns an error if `parent` is `Some` but the specified node doesn't exist.
    pub fn add_node(
        &mut self,
        parent: Option<NodeId>,
        name: Option<String>,
        position: cgmath::Point3<f32>,
        rotation: cgmath::Quaternion<f32>,
        scale: cgmath::Vector3<f32>,
    ) -> anyhow::Result<NodeId> {
        // Validate parent exists if specified
        if let Some(parent_id) = parent {
            if !self.nodes.contains_key(&parent_id) {
                anyhow::bail!("Parent node with ID {} not found in scene", parent_id);
            }
        }

        let id = self.next_node_id;
        self.next_node_id += 1;

        let mut node = Node::new(id, name, position, rotation, scale);

        // Set up parent-child relationship
        if let Some(parent_id) = parent {
            node.set_parent(Some(parent_id));
            // Safe to unwrap since we validated parent exists above
            self.nodes.get_mut(&parent_id).unwrap().add_child(id);
        } else {
            // No parent, so this is a root node
            self.root_nodes.push(id);
        }

        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Adds a new node with an instance attached.
    ///
    /// This is a convenience method that creates both an instance and a node
    /// in one call.
    ///
    /// # Arguments
    /// * `parent` - Optional parent node ID. If `Some`, the parent must exist in the scene.
    /// * `mesh` - The mesh ID for this instance.
    /// * `material` - The material ID for this instance.
    /// * `position` - Local position of the node.
    /// * `rotation` - Local rotation of the node.
    /// * `scale` - Local scale of the node.
    ///
    /// # Returns
    /// The ID of the newly created node, or an error if the parent doesn't exist.
    ///
    /// # Errors
    /// Returns an error if `parent` is `Some` but the specified node doesn't exist.
    pub fn add_instance_node(
        &mut self,
        parent: Option<NodeId>,
        mesh: MeshId,
        material: MaterialId,
        name: Option<String>,
        position: cgmath::Point3<f32>,
        rotation: cgmath::Quaternion<f32>,
        scale: cgmath::Vector3<f32>,
    ) -> anyhow::Result<NodeId> {
        // Create the instance
        let instance_id = self.add_instance(mesh, material);

        // Create the node (validates parent exists)
        let node_id = self.add_node(parent, name, position, rotation, scale)?;

        // Attach instance to node
        // Safe to unwrap since we just created the node above
        self.nodes.get_mut(&node_id).unwrap().set_instance(Some(instance_id));

        Ok(node_id)
    }

    /// Adds a node with default transform (identity).
    ///
    /// # Arguments
    /// * `parent` - Optional parent node ID. If `Some`, the parent must exist in the scene.
    /// * `name` - Optional name for the node.
    ///
    /// # Returns
    /// The ID of the newly created node, or an error if the parent doesn't exist.
    ///
    /// # Errors
    /// Returns an error if `parent` is `Some` but the specified node doesn't exist.
    pub fn add_default_node(&mut self, parent: Option<NodeId>, name: Option<String>) -> anyhow::Result<NodeId> {
        use cgmath::{Point3, Quaternion, Vector3};

        self.add_node(
            parent,
            name,
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

    /// Clears all nodes from the scene.
    ///
    /// This removes all nodes, instances, meshes, materials (except the default),
    /// textures, lights, and environment maps from the scene, resetting it to an empty state.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_nodes.clear();
        self.instances.clear();
        self.meshes.clear();
        self.lights.clear();
        self.textures.clear();
        self.environment_maps.clear();
        self.active_environment_map = None;
        self.annotation_root_node = None;

        // Keep only the default material (ID 0), remove all others
        self.materials.retain(|&id, _| id == DEFAULT_MATERIAL_ID);

        // Reset ID counters (but keep material counter since default material exists)
        self.next_node_id = 0;
        self.next_instance_id = 0;
        self.next_mesh_id = 0;
        self.next_texture_id = 0;
        self.next_environment_map_id = 0;
        // Don't reset next_material_id since we keep the default material
    }

    /// Sets up default lighting for the scene if no lights are present.
    ///
    /// Adds a single white point light at position (3, 3, 3) with intensity 1.0.
    /// This is useful when loading scenes that don't define their own lights.
    pub fn set_default_lights(&mut self) {
        if self.lights.is_empty() {
            self.lights.push(Light::point(
                cgmath::Vector3::new(3.0, 3.0, 3.0),
                crate::common::RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                1.0,
            ));
        }
    }

    /// Returns the root node for annotations, creating it if necessary.
    ///
    /// The annotation root node is a dedicated root node used by the
    /// `AnnotationManager` to organize all annotation geometry. This node
    /// is automatically recreated if the scene is cleared.
    ///
    /// # Panics
    /// Panics in debug builds if the annotation root node was set but no longer
    /// exists in the scene. This indicates the node was removed directly instead
    /// of using `clear()`.
    pub fn annotation_root_node(&mut self) -> NodeId {
        if let Some(node_id) = self.annotation_root_node {
            debug_assert!(
                self.nodes.contains_key(&node_id),
                "Annotation root node {} was removed without clearing the scene",
                node_id
            );
            return node_id;
        }

        // Create a new annotation root node
        let node_id = self.add_default_node(None, None)
            .expect("Failed to create annotation root node");
        self.annotation_root_node = Some(node_id);
        node_id
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

        let node_id = scene.add_default_node(None, None).unwrap();

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

        let root = scene.add_default_node(None, None).unwrap();
        let child = scene.add_default_node(Some(root), None).unwrap();

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

        let root = scene.add_default_node(None, None).unwrap();
        let child1 = scene.add_default_node(Some(root), None).unwrap();
        let child2 = scene.add_default_node(Some(root), None).unwrap();
        let child3 = scene.add_default_node(Some(root), None).unwrap();

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
        let root = scene.add_default_node(None, None).unwrap();
        let child1 = scene.add_default_node(Some(root), None).unwrap();
        let child2 = scene.add_default_node(Some(child1), None).unwrap();
        let child3 = scene.add_default_node(Some(child2), None).unwrap();

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

        let root1 = scene.add_default_node(None, None).unwrap();
        let root2 = scene.add_default_node(None, None).unwrap();
        let root3 = scene.add_default_node(None, None).unwrap();

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

        let root = scene.add_default_node(None, None).unwrap();
        let c1 = scene.add_default_node(Some(root), None).unwrap();
        let c2 = scene.add_default_node(Some(root), None).unwrap();
        let gc1 = scene.add_default_node(Some(c1), None).unwrap();
        let gc2 = scene.add_default_node(Some(c1), None).unwrap();
        let gc3 = scene.add_default_node(Some(c2), None).unwrap();

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
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

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
        let root = scene.add_default_node(None, None).unwrap();

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
            None,
            Point3::new(10.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

        // Child at (5, 0, 0) relative to parent
        let child = scene.add_node(
            Some(root),
            None,
            Point3::new(5.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

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
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(2.0, 2.0, 2.0),
        ).unwrap();

        // Child at (1, 0, 0)
        let child = scene.add_node(
            Some(parent),
            None,
            Point3::new(1.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

        let child_transform = scene.nodes_transform(child);

        // Child should be at (2, 0, 0) due to parent's scale
        assert!((child_transform[3][0] - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_transform_caching() {
        let mut scene = Scene::new();
        let root = scene.add_default_node(None, None).unwrap();

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
        let node_id = scene.add_default_node(None, None).unwrap();

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
        let root = scene.add_default_node(None, None).unwrap();
        let child1 = scene.add_default_node(Some(root), None).unwrap();
        let child2 = scene.add_default_node(Some(root), None).unwrap();
        let grandchild = scene.add_default_node(Some(child1), None).unwrap();

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

        let parent = scene.add_default_node(None, None).unwrap();
        let child = scene.add_default_node(Some(parent), None).unwrap();

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

        let root1 = scene.add_default_node(None, None).unwrap();
        let root2 = scene.add_default_node(None, None).unwrap();
        let _child1 = scene.add_default_node(Some(root1), None).unwrap();
        let _child2 = scene.add_default_node(Some(root2), None).unwrap();

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

        let root = scene.add_default_node(None, None).unwrap();

        // Create 100 children
        let mut children = Vec::new();
        for _ in 0..100 {
            let child = scene.add_default_node(Some(root), None).unwrap();
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
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

        scene.add_instance_node(
            None, 1, 2,
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

        scene.add_instance_node(
            None, 1, 1,
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        ).unwrap();

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
        let node_id = scene.add_default_node(None, None).unwrap();

        {
            let node = scene.get_node_mut(node_id).unwrap();
            node.set_position(Point3::new(10.0, 20.0, 30.0));
        }

        let node = scene.get_node(node_id).unwrap();
        assert_eq!(node.position(), Point3::new(10.0, 20.0, 30.0));
    }

    // ========================================================================
    // Parent Validation Tests
    // ========================================================================

    #[test]
    fn test_add_node_with_invalid_parent_fails() {
        let mut scene = Scene::new();

        // Try to add a node with a non-existent parent
        let result = scene.add_node(
            Some(999), // Non-existent parent ID
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        assert!(result.is_err());
        assert_eq!(scene.nodes.len(), 0);
    }

    #[test]
    fn test_add_default_node_with_invalid_parent_fails() {
        let mut scene = Scene::new();

        let result = scene.add_default_node(Some(999), None);

        assert!(result.is_err());
        assert_eq!(scene.nodes.len(), 0);
    }

    #[test]
    fn test_add_instance_node_with_invalid_parent_fails() {
        let mut scene = Scene::new();

        let result = scene.add_instance_node(
            Some(999),
            0,
            0,
            None,
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        assert!(result.is_err());
        // No node should be created
        assert_eq!(scene.nodes.len(), 0);
        // But instance is created before the node validation - this is a side effect
        // that could be improved in the future
    }
}