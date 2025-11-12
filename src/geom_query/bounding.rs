use crate::common::Aabb;
use crate::scene::{TreeVisitor, walk_tree, Mesh, Node, Scene};
use cgmath::Point3;

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
        // This will be mutated if the node has children to contain their bounding.
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
            let local_bounds = mesh.bounding();
            let world_bounds = local_bounds.map(|bounds| {
                let world_transform = self.scene.nodes_transform(node.id);
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
        walk_tree(scene, root_id, &mut visitor);
    }
}
