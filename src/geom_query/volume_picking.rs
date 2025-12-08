use crate::common::ConvexPolyhedron;
use crate::scene::{InstanceId, NodeId, Scene};
use cgmath::SquareMatrix;

/// Result of a volume-instance intersection test.
#[derive(Debug, Clone)]
pub struct VolumePickResult {
    /// The node that was hit
    pub node_id: NodeId,
    /// The instance that was hit
    pub instance_id: InstanceId,
    /// Indices of triangles that intersect the volume
    pub triangle_indices: Vec<usize>,
    /// True if the entire instance is fully contained within the volume
    pub fully_contained: bool,
}

/// Recursively tests a volume against a node and its descendants.
fn pick_node_from_volume(
    volume: &ConvexPolyhedron,
    node_id: NodeId,
    scene: &Scene,
    thorough: bool,
    results: &mut Vec<VolumePickResult>,
) {
    let node = scene
        .get_node(node_id)
        .expect("Node ID not found in scene during picking");

    // Broad phase: Test against this node's bounding box
    let Some(bounds) = scene.nodes_bounding(node_id) else {
        return; // Subtree has no geometry, skip
    };
    if !volume.intersects_aabb(&bounds) {
        return; // Volume doesn't intersect bounds, skip entire subtree
    }

    // If this node has an instance, test it
    if let Some(instance_id) = node.instance() {
        let instance = scene
            .instances
            .get(&instance_id)
            .expect("Instance referenced by node not found in scene");
        let mesh = scene
            .meshes
            .get(&instance.mesh)
            .expect("Mesh referenced by instance not found in scene");

        // Get world transform for this node
        let world_transform = scene.nodes_transform(node_id);
        let world_to_local = world_transform
            .invert()
            .expect("Node world transform is not invertible");

        // Transform volume to local mesh space
        let local_volume = volume.transform(&world_to_local);

        // Test against all triangles in the mesh
        if let Some(mesh_hit) = mesh.intersect_volume(&local_volume, thorough) {
            results.push(VolumePickResult {
                node_id,
                instance_id,
                triangle_indices: mesh_hit.triangle_indices,
                fully_contained: mesh_hit.fully_contained,
            });
        }
    }

    // Recurse to children
    for &child_id in node.children() {
        pick_node_from_volume(volume, child_id, scene, thorough, results);
    }
}

/// Picks all instances intersected by a convex volume.
///
/// The volume should be in world space. The function walks the scene tree from root nodes,
/// using cached bounding boxes to eliminate large portions of the scene efficiently.
///
/// # Arguments
/// * `volume` - The convex polyhedron to test against (in world space)
/// * `scene` - The scene to pick from
/// * `thorough` - If true, uses more accurate but slower edge-triangle intersection tests.
///   This catches edge cases where the volume passes through a triangle without any triangle
///   vertices being inside and without triangle edges crossing the volume boundary.
///
/// # Returns
/// A vector of VolumePickResult for each instance that intersects the volume.
/// Each result includes whether the instance is fully contained within the volume.
pub fn pick_all_from_volume(
    volume: &ConvexPolyhedron,
    scene: &Scene,
    thorough: bool,
) -> Vec<VolumePickResult> {
    let mut results = Vec::new();

    // Walk the tree from each root node
    for &root_id in scene.root_nodes() {
        pick_node_from_volume(volume, root_id, scene, thorough, &mut results);
    }

    results
}
