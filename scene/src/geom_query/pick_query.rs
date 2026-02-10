use cgmath::{Matrix4, SquareMatrix};

use crate::common::Aabb;
use crate::{InstanceId, Mesh, NodeId, Scene};

/// A query that can pick objects by traversing the scene tree.
///
/// Implement this trait to create custom picking behaviors. The generic
/// traversal handles tree walking, AABB culling, and coordinate space
/// transformations - implementors only need to define the actual tests.
pub trait PickQuery: Sized {
    /// Result type returned for hits. A single mesh test may produce multiple results
    /// (e.g., ray hitting multiple triangles).
    type Result;

    /// Broad phase test: does this query potentially intersect geometry within the bounds?
    ///
    /// Return `true` if the subtree should be explored, `false` to skip it entirely.
    /// This should be a fast, conservative test - false positives are acceptable,
    /// but false negatives will cause missed results.
    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool;

    /// Transform this query to a different coordinate space.
    ///
    /// Used to transform the query from world space to local mesh space.
    fn transform(&self, matrix: &Matrix4<f32>) -> Self;

    /// Narrow phase test: test the query against a mesh and collect results.
    ///
    /// Called when a node passes the broad phase and has an instance attached.
    /// The query has already been transformed to local mesh space.
    ///
    /// # Arguments
    /// * `mesh` - The mesh to test against (in local space)
    /// * `node_id` - ID of the node being tested
    /// * `instance_id` - ID of the instance being tested
    /// * `world_transform` - The node's world transform (for result computation)
    /// * `results` - Vector to push results into
    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node_id: NodeId,
        instance_id: InstanceId,
        world_transform: &Matrix4<f32>,
        results: &mut Vec<Self::Result>,
    );
}

/// Picks all instances in the scene that match the given query.
///
/// Walks the scene tree from all root nodes, using cached bounding boxes
/// for efficient culling. The query is transformed to local space for each
/// instance test.
pub fn pick_all<Q: PickQuery>(query: &Q, scene: &Scene) -> Vec<Q::Result> {
    let mut results = Vec::new();

    for &root_id in scene.root_nodes() {
        pick_node(query, root_id, scene, &mut results);
    }

    results
}

/// Recursively tests a query against a node and its descendants.
fn pick_node<Q: PickQuery>(
    query: &Q,
    node_id: NodeId,
    scene: &Scene,
    results: &mut Vec<Q::Result>,
) {
    let node = scene
        .get_node(node_id)
        .expect("Node ID not found in scene during picking");

    // Broad phase: Test against this node's bounding box
    let Some(bounds) = scene.nodes_bounding(node_id) else {
        return; // Subtree has no geometry, skip
    };
    if !query.might_intersect_bounds(&bounds) {
        return; // Query doesn't intersect bounds, skip entire subtree
    }

    // Narrow phase: If this node has an instance, test it
    if let Some(instance_id) = node.instance() {
        let instance = scene
            .instances
            .get(&instance_id)
            .expect("Instance referenced by node not found in scene");
        let mesh = scene
            .meshes
            .get(&instance.mesh)
            .expect("Mesh referenced by instance not found in scene");

        // Get world transform and compute inverse for local space conversion
        let world_transform = scene.nodes_transform(node_id);
        let world_to_local = world_transform
            .invert()
            .expect("Node world transform is not invertible");

        // Transform query to local mesh space
        let local_query = query.transform(&world_to_local);

        // Test against mesh and collect results
        local_query.collect_mesh_hits(mesh, node_id, instance_id, &world_transform, results);
    }

    // Recurse to children
    for &child_id in node.children() {
        pick_node(query, child_id, scene, results);
    }
}
