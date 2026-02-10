pub mod camera;
pub use wgpu_engine_common as common;
pub mod geom_query;
pub mod gltf;

pub use camera::Camera;

// Scene submodules (formerly scene/src/scene/)
pub mod annotation;
mod batch;
pub mod environment;
pub mod format;
mod instance;
mod light;
mod material;
mod mesh;
mod node;
mod texture;
mod tree;

use cgmath::{Matrix4, Point3, Quaternion, SquareMatrix, Vector3};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;

use annotation::{Annotation, AnnotationId, AnnotationManager};

// Public API exports
pub use environment::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};
pub use instance::{Instance, InstanceId};
pub use light::{Light, LightType, MAX_LIGHTS};
pub use material::{Material, MaterialId, DEFAULT_MATERIAL_ID};
pub use mesh::{Mesh, MeshDescriptor, MeshId, MeshIndex, MeshPrimitive, ObjMesh, PrimitiveType, Vertex};
pub use node::{EffectiveVisibility, Node, NodeId, Visibility};
pub use texture::{Texture, TextureId};
pub use tree::TreeVisitor;

// Items used by core crate for rendering
pub use material::MaterialProperties;
pub use batch::{DrawBatch, partition_batches};
pub use tree::{InstanceTransform, collect_instance_transforms};

use crate::common::{Aabb, RgbaColor};

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

    /// Annotation manager
    pub annotations: AnnotationManager,

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

            annotations: AnnotationManager::new(),

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
        let id = self.next_environment_map_id;
        let env_map = EnvironmentMap::from_hdr_path(id, path);
        self.next_environment_map_id += 1;
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
    pub fn collect_draw_batches(&self) -> Vec<DrawBatch> {
        use std::collections::HashMap;

        let instance_transforms = collect_instance_transforms(self);
        let mut batch_map: HashMap<(MeshId, MaterialId, PrimitiveType), DrawBatch> =
            HashMap::new();

        for inst_transform in instance_transforms {
            let Some(instance) = self.instances.get(&inst_transform.instance_id) else {
                continue;
            };
            let Some(mesh) = self.meshes.get(&instance.mesh) else {
                continue;
            };

            // Create a separate batch for each primitive type the mesh supports
            for primitive_type in [
                PrimitiveType::TriangleList,
                PrimitiveType::LineList,
                PrimitiveType::PointList,
            ] {
                if !mesh.has_primitive_type(primitive_type) {
                    continue;
                }

                let key = (instance.mesh, instance.material, primitive_type);
                batch_map
                    .entry(key)
                    .or_insert_with(|| {
                        DrawBatch::new(instance.mesh, instance.material, primitive_type)
                    })
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
    /// textures, lights, environment maps, and annotations from the scene,
    /// resetting it to an empty state.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_nodes.clear();
        self.instances.clear();
        self.meshes.clear();
        self.lights.clear();
        self.textures.clear();
        self.environment_maps.clear();
        self.active_environment_map = None;

        // Clear annotations (they're part of Scene now)
        self.annotations.clear();

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

    /// Clears scene geometry but preserves annotation data.
    ///
    /// Annotations will need to be re-reified after calling this.
    /// Use this when you want to reload scene content but keep annotations.
    pub fn clear_preserving_annotations(&mut self) {
        self.nodes.clear();
        self.root_nodes.clear();
        self.instances.clear();
        self.meshes.clear();
        self.lights.clear();
        self.textures.clear();
        self.environment_maps.clear();
        self.active_environment_map = None;

        // Mark annotations as unreified (their nodes no longer exist)
        self.annotations.mark_all_unreified();

        // Keep only the default material (ID 0), remove all others
        self.materials.retain(|&id, _| id == DEFAULT_MATERIAL_ID);

        // Reset ID counters (but keep material counter since default material exists)
        self.next_node_id = 0;
        self.next_instance_id = 0;
        self.next_mesh_id = 0;
        self.next_texture_id = 0;
        self.next_environment_map_id = 0;
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

    // ========== Annotation Reification API ==========

    /// Ensures the annotation root node exists and returns its ID.
    fn ensure_annotation_root(&mut self) -> NodeId {
        if let Some(root_id) = self.annotations.root_node() {
            if self.nodes.contains_key(&root_id) {
                return root_id;
            }
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
        if let Some(annotation) = self.annotations.get(id) {
            if annotation.is_reified() {
                return annotation.node_id();
            }
        }

        let annotation = self.annotations.get(id)?.clone();
        let root_id = self.ensure_annotation_root();

        let node_id = match &annotation {
            Annotation::Line(line) => {
                let data = line.to_mesh_data();
                self.create_annotation_node(root_id, data)
            }
            Annotation::Polyline(polyline) => {
                let data = polyline.to_mesh_data()?;
                self.create_annotation_node(root_id, data)
            }
            Annotation::Points(points) => {
                let data = points.to_mesh_data()?;
                self.create_annotation_node(root_id, data)
            }
            Annotation::Axes(axes) => {
                // Axes creates multiple children under a parent node
                let parent = self
                    .add_default_node(Some(root_id), axes.meta.name.clone())
                    .ok()?;
                for data in axes.to_mesh_data() {
                    self.create_annotation_node(parent, data);
                }
                Some(parent)
            }
            Annotation::Box(box_ann) => {
                let data = box_ann.to_mesh_data();
                self.create_annotation_node(root_id, data)
            }
            Annotation::Grid(grid) => {
                let data = grid.to_mesh_data();
                self.create_annotation_node(root_id, data)
            }
        }?;

        if let Some(annotation) = self.annotations.get_mut(id) {
            annotation.meta_mut().node_id = Some(node_id);
        }

        Some(node_id)
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
            Point3::new(0.0, 0.0, 0.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        )
        .ok()
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
            if let Some(root) = self.get_node_mut(root_id) {
                let scale = if visible { 1.0 } else { 0.0 };
                root.set_scale(Vector3::new(scale, scale, scale));
            }
        }
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

    // ========== Visibility API ==========

    /// Sets the visibility of a node and propagates invisibility to descendants.
    ///
    /// When setting to Invisible, all descendants are also marked invisible.
    /// When setting to Visible, only this node is changed.
    pub fn set_node_visibility(&mut self, node_id: NodeId, visibility: node::Visibility) {
        match visibility {
            node::Visibility::Visible => {
                let node = self.get_node_mut(node_id).expect("Node not found");
                node.set_visibility(node::Visibility::Visible);
                self.invalidate_ancestor_effective_visibility(node_id);
            }
            node::Visibility::Invisible => {
                self.set_subtree_visibility_recursive(node_id, node::Visibility::Invisible);
                if let Some(node) = self.get_node(node_id) {
                    if let Some(parent_id) = node.parent() {
                        self.invalidate_ancestor_effective_visibility(parent_id);
                    }
                }
            }
        }
    }

    /// Recursively sets visibility for a node and all descendants.
    fn set_subtree_visibility_recursive(&mut self, node_id: NodeId, visibility: node::Visibility) {
        let Some(node) = self.get_node_mut(node_id) else {
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
mod scene_tests;
