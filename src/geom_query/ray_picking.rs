use crate::common::Ray;
use crate::scene::{NodeId, InstanceId, Scene};
use cgmath::{InnerSpace, Point3, SquareMatrix};

/// Result of a ray-instance intersection test.
#[derive(Debug, Clone)]
pub struct PickResult {
    /// The instance that was hit
    pub instance_id: InstanceId,
    /// Distance along the ray to the hit point
    pub distance: f32,
    /// World-space hit location
    pub hit_point: Point3<f32>,
    /// Index of the triangle that was hit (index into the mesh's index buffer / 3)
    pub triangle_index: usize,
    /// Barycentric coordinates of the hit point on the triangle (u, v, w) where w = 1 - u - v
    pub barycentric: (f32, f32, f32),
}

/// Recursively tests a ray against a node and its descendants.
fn pick_node_from_ray(
    ray: &Ray,
    node_id: NodeId,
    scene: &Scene,
    results: &mut Vec<PickResult>,
) {
    let node = scene.get_node(node_id)
        .expect("Node ID not found in scene during picking");

    // Broad phase: Test against this node's bounding box
    let Some(bounds) = scene.nodes_bounding(node_id) else {
        return; // Subtree has no geometry, skip
    };
    if bounds.intersects_ray(ray).is_none() {
        return; // Ray doesn't hit bounds, skip entire subtree
    }

    // If this node has an instance, test it
    if let Some(instance_id) = node.instance() {
        let instance = scene.instances.get(&instance_id)
            .expect("Instance referenced by node not found in scene");
        let mesh = scene.meshes.get(&instance.mesh)
            .expect("Mesh referenced by instance not found in scene");

        // Get world transform for this node
        let world_transform = scene.nodes_transform(node_id);
        let world_to_local = world_transform.invert()
            .expect("Node world transform is not invertible");

        // Transform ray to local mesh space
        let local_ray = ray.transform(&world_to_local);

        // Test against all triangles in the mesh
        let mesh_hits = mesh.intersect_ray(&local_ray);

        // Transform hits to world space and add to results
        for mesh_hit in mesh_hits {
            // Transform hit point to world space
            let world_hit_point = {
                let homogeneous = world_transform * mesh_hit.hit_point.to_homogeneous();
                Point3::from_homogeneous(homogeneous)
            };

            // Compute distance in world space from ray origin
            let distance = (world_hit_point - ray.origin).magnitude();

            results.push(PickResult {
                instance_id,
                distance,
                hit_point: world_hit_point,
                triangle_index: mesh_hit.triangle_index,
                barycentric: mesh_hit.barycentric,
            });
        }
    }

    // Recurse to children
    for &child_id in node.children() {
        pick_node_from_ray(ray, child_id, scene, results);
    }
}

/// Picks all instances intersected by a ray, sorted by distance from near to far.
///
/// The ray should be in world space. The function walks the scene tree from root nodes,
/// using cached bounding boxes to eliminate large portions of the scene efficiently.
///
/// Returns a vector of PickResult sorted by distance (closest first).
pub fn pick_all_from_ray(ray: &Ray, scene: &Scene) -> Vec<PickResult> {
    let mut results = Vec::new();

    // Walk the tree from each root node
    for &root_id in scene.root_nodes() {
        pick_node_from_ray(ray, root_id, scene, &mut results);
    }

    // Sort by distance (closest first)
    results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

    results
}
