
pub use duck_engine_common as common;

// Scene submodules
mod id;
mod camera;
pub mod annotation;
mod environment;
pub mod geom_query;
pub mod gizmo;
mod instance;
mod coordinate_space;
mod light;
mod material;
mod mesh;
mod node;
mod texture;
mod view;

use cgmath::{InnerSpace, Matrix4, Point3, Quaternion, SquareMatrix, Vector3};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;

// ID types
pub use id::Id;
pub use annotation::AnnotationId;
pub use environment::EnvironmentMapId;
pub use instance::InstanceId;
pub use material::MaterialId;
pub use mesh::MeshId;
pub use node::NodeId;
pub use texture::TextureId;

pub use camera::Camera;
pub use coordinate_space::CoordinateSpace;
pub use view::{View, ViewId};
pub use instance::Instance;
pub use light::{Light, LightType, MAX_LIGHTS};
pub use material::{
    AlphaMode, Material, MaterialFlags, MaterialProperties,
    DEFAULT_METALLIC, DEFAULT_ROUGHNESS,
};
pub use mesh::{Mesh, MeshDescriptor, MeshIndex, MeshPrimitive, ObjMesh, PrimitiveType, SubMeshRange, Topology, Vertex};
pub use node::{CustomNodePayload, EffectiveVisibility, Node, NodePayload, Visibility};
pub use texture::{Texture, TextureFormat};
pub use environment::{
    CubemapFaceData, CubemapMipData, EnvironmentMap, EnvironmentSource, PreprocessedCubemap,
    PreprocessedIbl, CUBEMAP_FACES,
};

use annotation::{Annotation, AnnotationManager};
use crate::common::{Aabb};

/// Default generation counter value for newly created resources.
/// Starts at 1 so initial change detection triggers on first use.
pub(crate) fn initial_generation() -> u64 {
    1
}

/// Scene-level properties that affect shader generation.
///
/// Unlike MaterialProperties which describe individual materials,
/// SceneProperties describes scene-wide rendering features.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct SceneProperties {
    /// Whether image-based lighting is active (environment map present)
    pub has_ibl: bool,
}

/// The scene container holding all meshes, materials, textures, instances, nodes, and lights.
///
/// Scene provides device-free APIs for creating and managing scene objects.
///
/// # Examples
///
/// ```
/// use duck_engine_scene::{Scene, Mesh, MeshPrimitive, Vertex, Material, PrimitiveType, common};
/// use duck_engine_scene::common::RgbaColor;
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
/// let node_id = scene.add_instance_node(None, mesh_id, mat_id, None, common::Transform::IDENTITY);
/// ```
#[derive(Clone)]
pub struct Scene {
    meshes: HashMap<MeshId, Mesh>,
    instances: HashMap<InstanceId, Instance>,

    // Scene tree
    nodes: HashMap<NodeId, Node>,
    root_nodes: Vec<NodeId>,

    materials: HashMap<MaterialId, Material>,
    textures: HashMap<TextureId, Texture>,

    // Environment maps for IBL
    environment_maps: HashMap<EnvironmentMapId, EnvironmentMap>,
    /// The currently active environment map for IBL lighting.
    active_environment_map: Option<EnvironmentMapId>,

    /// The node (with a Camera payload) that drives rendering when no explicit camera is passed.
    active_camera: Option<NodeId>,

    /// Annotation manager
    pub annotations: AnnotationManager,

    /// Generation counter that increments on any node add, remove, or mutation.
    /// Used by the renderer to detect when scene data need re-collection.
    node_generation: u64,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            instances: HashMap::new(),

            nodes: HashMap::new(),
            root_nodes: Vec::new(),

            materials: HashMap::new(),
            textures: HashMap::new(),

            environment_maps: HashMap::new(),
            active_environment_map: None,

            active_camera: None,

            annotations: AnnotationManager::new(),

            node_generation: initial_generation(),
        }
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
        let id = mesh.id;
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

    /// Returns an iterator over all meshes.
    pub fn meshes(&self) -> impl Iterator<Item = &Mesh> {
        self.meshes.values()
    }

    /// Returns a mutable iterator over all meshes.
    pub fn meshes_mut(&mut self) -> impl Iterator<Item = &mut Mesh> {
        self.meshes.values_mut()
    }

    /// Returns the number of meshes in the scene.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Removes a mesh from the scene by ID.
    pub fn remove_mesh(&mut self, id: MeshId) {
        self.meshes.remove(&id);
    }

    // ========== Material API ==========

    /// Adds a material to the scene.
    ///
    /// # Arguments
    /// * `material` - The material to add
    ///
    /// # Returns
    /// The unique ID assigned to this material
    pub fn add_material(&mut self, material: Material) -> MaterialId {
        let id = material.id;
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

    /// Returns an iterator over all materials.
    pub fn materials(&self) -> impl Iterator<Item = &Material> {
        self.materials.values()
    }

    /// Returns a mutable iterator over all materials.
    pub fn materials_mut(&mut self) -> impl Iterator<Item = &mut Material> {
        self.materials.values_mut()
    }

    /// Returns the number of materials in the scene.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Removes a material from the scene by ID.
    pub fn remove_material(&mut self, id: MaterialId) {
        self.materials.remove(&id);
    }

    // ========== Texture API ==========

    /// Adds a texture to the scene.
    ///
    /// # Arguments
    /// * `texture` - The texture to add
    ///
    /// # Returns
    /// The unique ID assigned to this texture
    pub fn add_texture(&mut self, texture: Texture) -> TextureId {
        let id = texture.id;
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

    /// Returns an iterator over all textures.
    pub fn textures(&self) -> impl Iterator<Item = &Texture> {
        self.textures.values()
    }

    /// Returns a mutable iterator over all textures.
    pub fn textures_mut(&mut self) -> impl Iterator<Item = &mut Texture> {
        self.textures.values_mut()
    }

    /// Returns the number of textures in the scene.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    // ========== Environment Map API (IBL) ==========

    /// Creates and adds an environment map from an equirectangular HDR file path.
    ///
    /// The HDR file will be loaded and processed into IBL maps when first rendered.
    ///
    /// # Arguments
    /// * `path` - Path to an HDR file (.hdr format)
    ///
    /// # Returns
    /// The unique ID assigned to this environment map
    pub fn add_environment_map_from_hdr_path(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> EnvironmentMapId {
        let env_map = EnvironmentMap::from_hdr_path(path);
        let id = env_map.id;
        self.environment_maps.insert(id, env_map);
        id
    }

    /// Creates and adds an environment map from in-memory HDR data.
    ///
    /// The HDR data will be processed into IBL maps when first rendered.
    ///
    /// # Arguments
    /// * `data` - Raw .hdr file bytes
    ///
    /// # Returns
    /// The unique ID assigned to this environment map
    pub fn add_environment_map_from_hdr_data(
        &mut self,
        data: Vec<u8>,
    ) -> EnvironmentMapId {
        let env_map = EnvironmentMap::from_hdr_data(data);
        let id = env_map.id;
        self.environment_maps.insert(id, env_map);
        id
    }

    /// Adds an environment map to the scene using its existing ID.
    pub fn add_environment_map(&mut self, env_map: EnvironmentMap) -> EnvironmentMapId {
        let id = env_map.id;
        self.environment_maps.insert(id, env_map);
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

    /// Returns an iterator over all environment maps.
    pub fn environment_maps(&self) -> impl Iterator<Item = &EnvironmentMap> {
        self.environment_maps.values()
    }

    /// Returns a mutable iterator over all environment maps.
    pub fn environment_maps_mut(&mut self) -> impl Iterator<Item = &mut EnvironmentMap> {
        self.environment_maps.values_mut()
    }

    /// Returns the number of environment maps in the scene.
    pub fn environment_map_count(&self) -> usize {
        self.environment_maps.len()
    }

    /// Returns true if the scene has any environment maps.
    pub fn has_environment_maps(&self) -> bool {
        !self.environment_maps.is_empty()
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

    /// Adds an instance to the scene, binding a mesh to a material.
    ///
    /// # Arguments
    /// * `mesh` - The mesh ID for this instance
    /// * `material` - The material ID for this instance
    ///
    /// # Returns
    /// The unique ID assigned to this instance
    pub fn add_instance(&mut self, instance: Instance) -> InstanceId {
        let id = instance.id;
        self.instances.insert(id, instance);
        id
    }

    /// Gets a reference to an instance by ID.
    pub fn get_instance(&self, id: InstanceId) -> Option<&Instance> {
        self.instances.get(&id)
    }

    /// Gets a mutable reference to an instance by ID.
    pub fn get_instance_mut(&mut self, id: InstanceId) -> Option<&mut Instance> {
        self.instances.get_mut(&id)
    }

    /// Returns an iterator over all instances.
    pub fn instances(&self) -> impl Iterator<Item = &Instance> {
        self.instances.values()
    }

    /// Returns a mutable iterator over all instances.
    pub fn instances_mut(&mut self) -> impl Iterator<Item = &mut Instance> {
        self.instances.values_mut()
    }

    /// Returns the number of instances in the scene.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Removes an instance from the scene by ID.
    pub fn remove_instance(&mut self, id: InstanceId) {
        self.instances.remove(&id);
    }


    // ========== Active Camera API ==========

    /// Returns the node ID of the active camera, if any.
    pub fn active_camera(&self) -> Option<NodeId> {
        self.active_camera
    }

    /// Returns the [`Camera`] from the active camera node, or `None` if there is no active camera
    /// or the node does not carry a `NodePayload::Camera` payload.
    pub fn active_camera_data(&self) -> Option<&Camera> {
        let node = self.get_node(self.active_camera?)?;
        match node.payload() {
            NodePayload::Camera(cam) => Some(cam),
            _ => None,
        }
    }

    /// Sets the active camera node. Pass `None` to clear.
    pub fn set_active_camera(&mut self, node_id: Option<NodeId>) {
        self.active_camera = node_id;
    }

    // ========== Node Generation ==========

    /// Returns the current node generation counter.
    ///
    /// Increments on every node add, remove, or payload mutation. The renderer
    /// uses this to detect when lights (and other node data) need re-collection.
    pub fn node_generation(&self) -> u64 {
        self.node_generation
    }

    // ========== Light Node Helpers ==========

    /// Returns true if any node in the scene carries a `Light` payload.
    pub fn has_light_nodes(&self) -> bool {
        self.nodes.values().any(|n| matches!(n.payload(), NodePayload::Light(_)))
    }

    /// Returns the number of nodes with a [`NodePayload::Light`] payload.
    pub fn light_count(&self) -> usize {
        self.nodes.values().filter(|n| matches!(n.payload(), NodePayload::Light(_))).count()
    }

    /// Adds a default key + fill directional light pair as children of `camera_node_id`.
    ///
    /// Because the lights are children of the camera node, callers must keep that
    /// node's world Transform in sync with the viewer camera so the lights remain
    /// camera-relative.
    ///
    /// # TODO
    /// When Camera is refactored to intrinsics-only (dropping eye/target/up), the
    /// camera node's Transform will be the canonical pose and this sync will be
    /// automatic. At that point the `camera_node_id` relationship stays but the
    /// manual Transform sync in the viewer can be removed.
    pub fn set_default_light_nodes(&mut self, camera_node_id: NodeId) {
        let white = crate::common::RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
        let key_id = self
            .add_node(Some(camera_node_id), Some("DefaultKeyLight".to_string()), common::Transform::IDENTITY)
            .expect("Failed to create key light node");
        self.nodes.get_mut(&key_id).unwrap().set_payload(NodePayload::Light(Light::directional(white, 1.0)));
        self.node_generation += 1;
        let fill_id = self
            .add_node(Some(camera_node_id), Some("DefaultFillLight".to_string()), common::Transform::IDENTITY)
            .expect("Failed to create fill light node");
        self.nodes.get_mut(&fill_id).unwrap().set_payload(NodePayload::Light(Light::directional(white, 0.3)));
        self.node_generation += 1;
    }

    // ========== Node API ==========

    /// Gets a reference to a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Returns an iterator over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Returns the number of nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if a node with the given ID exists in the scene.
    pub fn has_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    // TODO: Remove this after format.rs refactor
    /// Sets the order of root nodes (used during deserialization).
    pub fn set_root_node_order(&mut self, order: Vec<NodeId>) {
        self.root_nodes = order;
    }

    /// Returns a slice of root node IDs.
    pub fn root_nodes(&self) -> &[NodeId] {
        &self.root_nodes
    }

    /// Adds a new node to the scene tree.
    ///
    /// # Arguments
    /// * `parent` - Optional parent node ID. If `Some`, the parent must exist in the scene.
    /// * `position` - Local position of the node.
    /// * `transform` - Local transform of the node.
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
        transform: common::Transform,
    ) -> anyhow::Result<NodeId> {
        // Validate parent exists if specified
        if let Some(parent_id) = parent
            && !self.nodes.contains_key(&parent_id) {
                anyhow::bail!("Parent node with ID {} not found in scene", parent_id);
            }

        let mut node = Node::new(name, transform);
        let id = node.id;

        // Set up parent-child relationship
        if let Some(parent_id) = parent {
            node.set_parent_unchecked(Some(parent_id));
            // Safe to unwrap since we validated parent exists above
            self.nodes.get_mut(&parent_id).unwrap().add_child_unchecked(id);
        } else {
            // No parent, so this is a root node
            self.root_nodes.push(id);
        }

        self.nodes.insert(id, node);
        self.node_generation += 1;
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
    /// * `name` - Optional name for the node.
    /// * `transform` - Local transform of the node.
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
        transform: common::Transform,
    ) -> anyhow::Result<NodeId> {
        // Create the instance
        let instance_id = self.add_instance(Instance::new(mesh, material));

        // Create the node (validates parent exists)
        let node_id = self.add_node(parent, name, transform)?;

        // Attach instance to node
        // Safe to unwrap since we just created the node above
        self.nodes.get_mut(&node_id).unwrap().set_payload(NodePayload::Instance(instance_id));

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
        self.add_node(parent, name, common::Transform::IDENTITY)
    }

    /// Inserts a pre-built node using its existing ID.
    ///
    /// Does **not** update `root_nodes` or parent/child links — the caller
    /// is responsible for tree consistency (e.g. via `set_root_node_order`).
    pub fn insert_node(&mut self, node: Node) {
        self.nodes.insert(node.id, node);
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
                parent_node.remove_child_unchecked(node_id);
            }
        } else {
            // This is a root node, remove from root_nodes list
            self.root_nodes.retain(|&id| id != node_id);
        }

        // Finally, remove the node itself
        self.nodes.remove(&node_id);
        self.node_generation += 1;
    }

    /// Clears all nodes from the scene, resetting it to an empty state.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_nodes.clear();
        self.instances.clear();
        self.meshes.clear();
        self.materials.clear();
        self.textures.clear();
        self.environment_maps.clear();
        self.active_environment_map = None;
        self.active_camera = None;
        self.annotations.clear();
        self.node_generation += 1;
    }

    /// Removes orphaned resources not referenced by any node in the scene.
    ///
    /// Walks the scene tree to find all referenced instances, meshes, materials,
    /// and textures, then removes any that are unreferenced. The default material
    /// is always retained.
    pub fn cleanup(&mut self) {
        use std::collections::HashSet;

        // Collect all instance IDs referenced by nodes
        let referenced_instances: HashSet<InstanceId> = self
            .nodes
            .values()
            .filter_map(|node| match node.payload() {
                NodePayload::Instance(id) => Some(*id),
                _ => None,
            })
            .collect();

        // Collect mesh and material IDs referenced by retained instances
        let mut referenced_meshes = HashSet::new();
        let mut referenced_materials = HashSet::new();
        for &inst_id in &referenced_instances {
            if let Some(inst) = self.instances.get(&inst_id) {
                referenced_meshes.insert(inst.mesh());
                referenced_materials.insert(inst.material());
            }
        }

        // Collect texture IDs referenced by retained materials
        let mut referenced_textures = HashSet::new();
        for &mat_id in &referenced_materials {
            if let Some(mat) = self.materials.get(&mat_id) {
                if let Some(tex) = mat.base_color_texture() {
                    referenced_textures.insert(tex);
                }
                if let Some(tex) = mat.normal_texture() {
                    referenced_textures.insert(tex);
                }
                if let Some(tex) = mat.metallic_roughness_texture() {
                    referenced_textures.insert(tex);
                }
            }
        }

        // Remove unreferenced resources
        self.instances.retain(|id, _| referenced_instances.contains(id));
        self.meshes.retain(|id, _| referenced_meshes.contains(id));
        self.materials.retain(|id, _| referenced_materials.contains(id));
        self.textures.retain(|id, _| referenced_textures.contains(id));
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

            node.mark_bounds_dirty();

            // Move to parent
            current_id = node.parent();
        }
    }

    // ========== Node Transform Mutation API ==========

    /// Sets the payload of a node and invalidates ancestor bounds.
    pub fn set_node_payload(&mut self, node_id: NodeId, payload: NodePayload) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_payload(payload);
        self.node_generation += 1;
        self.invalidate_ancestor_bounds(node_id);
    }

    pub fn set_node_transform(&mut self, node_id: NodeId, transform: common::Transform) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_transform(transform);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the position of a node and invalidates ancestor bounds.
    pub fn set_node_position(&mut self, node_id: NodeId, position: Point3<f32>) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_position(position);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the rotation of a node and invalidates ancestor bounds.
    pub fn set_node_rotation(&mut self, node_id: NodeId, rotation: Quaternion<f32>) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_rotation(rotation);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the scale of a node and invalidates ancestor bounds.
    pub fn set_node_scale(&mut self, node_id: NodeId, scale: Vector3<f32>) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_scale(scale);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    // ========== Visibility API ==========

    /// Sets the visibility of a node and propagates invisibility to descendants.
    ///
    /// When setting to Invisible, all descendants are also marked invisible.
    /// When setting to Visible, only this node is changed.
    pub fn set_node_visibility(&mut self, node_id: NodeId, visibility: node::Visibility) {
        match visibility {
            node::Visibility::Visible => {
                let node = self.nodes.get_mut(&node_id).expect("Node not found");
                node.set_visibility(node::Visibility::Visible);
                self.invalidate_ancestor_effective_visibility(node_id);
            }
            node::Visibility::Invisible => {
                self.set_subtree_visibility_recursive(node_id, node::Visibility::Invisible);
                if let Some(node) = self.get_node(node_id)
                    && let Some(parent_id) = node.parent() {
                        self.invalidate_ancestor_effective_visibility(parent_id);
                    }
            }
        }
    }

    /// Recursively sets visibility for a node and all descendants.
    fn set_subtree_visibility_recursive(&mut self, node_id: NodeId, visibility: node::Visibility) {
        let Some(node) = self.nodes.get_mut(&node_id) else {
            return;
        };
        node.set_visibility(visibility);
        let children: Vec<NodeId> = node.children().to_vec();

        for child_id in children {
            self.set_subtree_visibility_recursive(child_id, visibility);
        }
    }

    /// Invalidates effective visibility cache for ancestors.
    fn invalidate_ancestor_effective_visibility(&self, node_id: NodeId) {
        let mut current_id = Some(node_id);
        while let Some(id) = current_id {
            let Some(node) = self.get_node(id) else {
                break;
            };
            node.mark_visibility_dirty();
            current_id = node.parent();
        }
    }

    /// Gets the effective visibility of a node with caching.
    pub fn node_effective_visibility(
        &self,
        node_id: NodeId,
    ) -> node::EffectiveVisibility {
        let node = self.get_node(node_id).expect("Node not found");

        if let Some(cached) = node.cached_effective_visibility() {
            return cached;
        }

        let effective = self.compute_effective_visibility_recursive(node_id);
        node.set_cached_effective_visibility(effective);
        effective
    }

    /// Recursively computes effective visibility.
    fn compute_effective_visibility_recursive(
        &self,
        node_id: NodeId,
    ) -> node::EffectiveVisibility {
        let node = self.get_node(node_id).expect("Node not found");

        if node.visibility() == node::Visibility::Invisible {
            return node::EffectiveVisibility::Invisible;
        }

        let mut all_visible = true;
        for &child_id in node.children() {
            let child_effective = self.node_effective_visibility(child_id);
            if child_effective != node::EffectiveVisibility::Visible {
                all_visible = false;
                break;
            }
        }

        if all_visible {
            node::EffectiveVisibility::Visible
        } else {
            node::EffectiveVisibility::Mixed
        }
    }

    // ========== Transform & Bounding Box API ==========

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
        let instance_id = match node.payload() {
            NodePayload::Instance(id) => Some(*id),
            _ => None,
        };
        let node_bounds = if let Some(instance_id) = instance_id {
            let instance = self.instances.get(&instance_id)
                .expect("Instance referenced by node not found in scene");
            let mesh = self.meshes.get(&instance.mesh())
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
            // No instance - just use merged child bounds
            merged_bounds
        };

        // Cache it
        node.set_cached_bounds(node_bounds);

        node_bounds
    }

    // ========== Annotation Reification API ==========

    /// Ensures the annotation root node exists and returns its ID.
    ///
    /// This is also used by overlay systems (e.g. gizmos) that parent their
    /// geometry under the annotation root to keep it grouped with other
    /// non-scene overlay content.
    // TODO: Once we support multiple overlay types beyond annotations and gizmos,
    // consider introducing a dedicated overlay system with its own root node
    // rather than piggy-backing on the annotation root.
    pub fn ensure_annotation_root(&mut self) -> NodeId {
        if let Some(root_id) = self.annotations.root_node()
            && self.nodes.contains_key(&root_id) {
                return root_id;
            }

        let root_id = self
            .add_default_node(None, Some("AnnotationRoot".to_string()))
            .expect("Failed to create annotation root node");
        self.annotations.set_root_node(root_id);
        root_id
    }

    /// Reifies all unreified annotations, creating scene nodes for them.
    ///
    /// # Returns
    /// The number of annotations that were reified.
    pub fn reify_annotations(&mut self) -> usize {
        // Check for stale reference-based annotations and mark them for re-reification.
        let stale_ids: Vec<AnnotationId> = self
            .annotations
            .iter()
            .filter(|a| a.is_reified())
            .filter(|a| match a {
                // TODO Phase 2: redesign light annotations with NodeId refs
                Annotation::PointLight(pl) => {
                    pl.reified_generation != Some(self.node_generation)
                }
                Annotation::SpotLight(sl) => {
                    sl.reified_generation != Some(self.node_generation)
                }
                Annotation::Normals(normals) => {
                    let current_gen = self
                        .nodes
                        .get(&normals.target_node_id)
                        .and_then(|n| match n.payload() {
                            NodePayload::Instance(id) => Some(*id),
                            _ => None,
                        })
                        .and_then(|iid| self.instances.get(&iid))
                        .and_then(|inst| self.meshes.get(&inst.mesh()))
                        .map(|m| m.generation());
                    normals.reified_generation != current_gen
                }
                _ => false,
            })
            .map(|a| a.id())
            .collect();

        // Remove old scene nodes for stale annotations
        for &id in &stale_ids {
            if let Some(annotation) = self.annotations.get(id)
                && let Some(old_node_id) = annotation.node_id() {
                    self.remove_node(old_node_id);
                }
            if let Some(ann) = self.annotations.get_mut(id) {
                ann.meta_mut().node_id = None;
            }
        }

        // Now reify all unreified annotations (including freshly un-reified stale ones)
        let unreified_ids: Vec<AnnotationId> = self
            .annotations
            .iter_unreified()
            .map(|a| a.id())
            .collect();

        let count = unreified_ids.len();
        for id in unreified_ids {
            self.reify_annotation(id);
        }
        count
    }

    /// Reifies a single annotation by ID.
    ///
    /// # Returns
    /// The NodeId of the reified annotation, or None if not found or already reified.
    pub fn reify_annotation(&mut self, id: AnnotationId) -> Option<NodeId> {
        // Check if already reified
        if let Some(annotation) = self.annotations.get(id)
            && annotation.is_reified() {
                return annotation.node_id();
            }

        let annotation = self.annotations.get(id)?.clone();
        let name = annotation.meta().name.clone();
        let meshes = self.resolve_annotation_meshes(&annotation);
        let root_id = self.ensure_annotation_root();
        let node_id = self.create_annotation_nodes(root_id, name, meshes)?;

        // Store generation for reference-based annotations
        match self.annotations.get_mut(id) {
            // TODO Phase 2: redesign light annotations with NodeId refs
            Some(Annotation::PointLight(pl)) => {
                pl.reified_generation = Some(self.node_generation);
            }
            Some(Annotation::SpotLight(sl)) => {
                sl.reified_generation = Some(self.node_generation);
            }
            Some(Annotation::Normals(normals)) => {
                let target_node_id = normals.target_node_id;
                normals.reified_generation = self
                    .nodes
                    .get(&target_node_id)
                    .and_then(|n| match n.payload() {
                        NodePayload::Instance(id) => Some(*id),
                        _ => None,
                    })
                    .and_then(|iid| self.instances.get(&iid))
                    .and_then(|inst| self.meshes.get(&inst.mesh()))
                    .map(|m| m.generation());
            }
            _ => {}
        }

        if let Some(annotation) = self.annotations.get_mut(id) {
            annotation.meta_mut().node_id = Some(node_id);
        }

        Some(node_id)
    }

    /// Resolves an annotation's mesh data, consulting scene state as needed.
    fn resolve_annotation_meshes(
        &self,
        annotation: &Annotation,
    ) -> Vec<annotation::AnnotationMeshData> {
        match annotation {
            Annotation::Line(a) => vec![a.to_mesh_data()],
            Annotation::Polyline(a) => a.to_mesh_data().into_iter().collect(),
            Annotation::Points(a) => a.to_mesh_data().into_iter().collect(),
            Annotation::Axes(a) => a.to_mesh_data(),
            Annotation::Box(a) => vec![a.to_mesh_data()],
            Annotation::Grid(a) => vec![a.to_mesh_data()],
            Annotation::PointLight(_) | Annotation::SpotLight(_) => {
                // TODO: Phase 2 — redesign light annotations using NodeId references.
                // Position/direction now come from the node world transform, not Light payload.
                todo!("Light annotations require Phase 2 redesign with NodeId references")
            }
            Annotation::Normals(normals) => {
                let Some(node) = self.nodes.get(&normals.target_node_id) else {
                    return vec![];
                };
                let instance_id = match node.payload() {
                    NodePayload::Instance(id) => *id,
                    _ => return vec![],
                };
                let Some(instance) = self.instances.get(&instance_id) else {
                    return vec![];
                };
                let Some(mesh) = self.meshes.get(&instance.mesh()) else {
                    return vec![];
                };
                let world_transform = self.nodes_transform(normals.target_node_id);
                let normal_matrix =
                    crate::common::compute_normal_matrix(&world_transform);

                let world_vertices: Vec<(Point3<f32>, Vector3<f32>)> = mesh
                    .vertices()
                    .iter()
                    .map(|v| {
                        let local_pos =
                            Point3::new(v.position[0], v.position[1], v.position[2]);
                        let local_normal =
                            Vector3::new(v.normal[0], v.normal[1], v.normal[2]);

                        let world_pos = cgmath::Transform::transform_point(
                            &world_transform,
                            local_pos,
                        );
                        let world_normal: Vector3<f32> = normal_matrix * local_normal;
                        let world_normal = world_normal.normalize();

                        (world_pos, world_normal)
                    })
                    .collect();

                normals.to_mesh_data(&world_vertices)
            }
        }
    }

    /// Helper to create a node from annotation mesh data
    fn create_annotation_node(
        &mut self,
        parent: NodeId,
        data: annotation::AnnotationMeshData,
    ) -> Option<NodeId> {
        let mesh_id = self.add_mesh(data.mesh);
        let material_id = self.add_material(data.material);

        self.add_instance_node(
            Some(parent),
            mesh_id,
            material_id,
            data.name,
            common::Transform::IDENTITY,
        )
        .ok()
    }

    /// Creates scene nodes from annotation mesh data.
    /// Single mesh: direct child of `parent`. Multiple: grouped under a named node.
    fn create_annotation_nodes(
        &mut self,
        parent: NodeId,
        name: Option<String>,
        meshes: Vec<annotation::AnnotationMeshData>,
    ) -> Option<NodeId> {
        match meshes.len() {
            0 => None,
            1 => self.create_annotation_node(parent, meshes.into_iter().next().unwrap()),
            _ => {
                let group = self.add_default_node(Some(parent), name).ok()?;
                for data in meshes {
                    self.create_annotation_node(group, data);
                }
                Some(group)
            }
        }
    }

    /// Removes an annotation and its scene node (if reified).
    pub fn remove_annotation(&mut self, id: AnnotationId) -> Option<Annotation> {
        let annotation = self.annotations.remove(id)?;
        if let Some(node_id) = annotation.node_id() {
            self.remove_node(node_id);
        }
        Some(annotation)
    }

    /// Sets visibility for all annotations.
    pub fn set_annotations_visible(&mut self, visible: bool) {
        if let Some(root_id) = self.annotations.root_node() {
            let scale = if visible { 1.0 } else { 0.0 };
            self.set_node_scale(root_id, Vector3::new(scale, scale, scale));
        }
    }
}

#[cfg(test)]
mod scene_tests;
