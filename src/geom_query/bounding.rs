use crate::common::Aabb;
use crate::scene::{TreeVisitor, walk_tree_recursive, Mesh, Node, Scene};
use cgmath::Point3;

/// Computes the local-space axis-aligned bounding box for a mesh.
/// Returns None if the mesh has no vertices.
pub fn compute_mesh_bounds(mesh: &Mesh) -> Option<Aabb> {
    let vertices = mesh.vertices();
    if vertices.is_empty() {
        return None;
    }

    // Extract positions from vertices
    let positions: Vec<Point3<f32>> = vertices
        .iter()
        .map(|v| Point3::new(v.position[0], v.position[1], v.position[2]))
        .collect();

    Aabb::from_points(&positions)
}

/// Visitor implementation that computes and caches bounding boxes during tree traversal.
pub struct BoundingBoxCollector<'a> {
    scene: &'a Scene,
    /// Stack of child bounds at each tree depth level
    /// Each entry contains the merged bounds of all processed children at that level
    bounds_stack: Vec<Option<Aabb>>,
}

impl<'a> BoundingBoxCollector<'a> {
    /// Creates a new bounding box collector for the given scene.
    pub fn new(scene: &'a Scene) -> Self {
        Self {
            scene,
            bounds_stack: Vec::new(),
        }
    }
}

impl<'a> TreeVisitor for BoundingBoxCollector<'a> {
    fn enter_node(&mut self, _node: &Node) {
        // Push a new level onto the stack for this node's children
        self.bounds_stack.push(None);
    }

    fn exit_node(&mut self, node: &Node) {
        // Pop the merged bounds of all children
        let child_bounds = self.bounds_stack.pop().expect("Bounds stack underflow");

        // Compute this node's bounds
        let node_bounds = if let Some(instance_id) = node.instance() {
            // Leaf node with an instance - compute from mesh
            let instance = self.scene.instances.get(&instance_id)
                .expect("Instance referenced by node not found in scene");
            let mesh = self.scene.meshes.get(&instance.mesh)
                .expect("Mesh referenced by instance not found in scene");

            // Get local-space mesh bounds and transform to world space
            let local_bounds = compute_mesh_bounds(mesh);
            let world_bounds = local_bounds.map(|bounds| {
                let world_transform = node.cached_world_transform();
                bounds.transform(&world_transform)
            });

            // Merge with child bounds if any
            match (world_bounds, child_bounds) {
                (Some(wb), Some(cb)) => Some(wb.merge(&cb)),
                (Some(wb), None) => Some(wb),
                (None, cb) => cb,
            }
        } else {
            // Branch node - just use merged child bounds
            child_bounds
        };

        // Cache the computed bounds on the node
        node.set_cached_bounds(node_bounds);

        // Merge this node's bounds into the parent level (if there is one)
        if let Some(parent_bounds) = self.bounds_stack.last_mut() {
            *parent_bounds = match (*parent_bounds, node_bounds) {
                (Some(pb), Some(nb)) => Some(pb.merge(&nb)),
                (Some(pb), None) => Some(pb),
                (None, Some(nb)) => Some(nb),
                (None, None) => None,
            };
        }
    }
}

/// Computes the world-space bounding boxes for all nodes in the scene.
/// This uses the tree visitor pattern to traverse the scene hierarchy,
/// computing bounds bottom-up and caching them on each node.
///
/// Note: This assumes world transforms have already been computed and cached.
/// Call this after calling `collect_instance_transforms` if transforms may be dirty.
pub fn compute_node_bounds(scene: &Scene) {
    let mut visitor = BoundingBoxCollector::new(scene);

    // Process each root node
    for &root_id in scene.root_nodes() {
        walk_tree_recursive(scene, root_id, &mut visitor);
    }
}
