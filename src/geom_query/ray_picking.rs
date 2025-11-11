use crate::common::{Ray, EPSILON};
use crate::scene::{NodeId, InstanceId, Scene, collect_instance_transforms};
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

/// Tests if a ray intersects a triangle using the Möller-Trumbore algorithm.
///
/// Returns Some((t, u, v)) if the ray hits the triangle, where:
/// - t: distance along the ray
/// - u, v: barycentric coordinates (w = 1 - u - v)
///
/// Returns None if there's no intersection or if the intersection is behind the ray origin.
fn intersect_ray_triangle(
    ray: &Ray,
    v0: Point3<f32>,
    v1: Point3<f32>,
    v2: Point3<f32>,
) -> Option<(f32, f32, f32)> {
    // Compute edges from v0
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    // Begin calculating determinant - also used to calculate u parameter
    let h = ray.direction.cross(edge2);
    let det = edge1.dot(h);

    // If determinant is near zero, ray lies in plane of triangle or is parallel
    if det.abs() < EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;

    // Calculate distance from v0 to ray origin
    let s = ray.origin - v0;

    // Calculate u parameter and test bounds
    let u = inv_det * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    // Prepare to test v parameter
    let q = s.cross(edge1);

    // Calculate v parameter and test bounds
    let v = inv_det * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    // At this stage we can compute t to find out where the intersection point is on the line
    let t = inv_det * edge2.dot(q);

    // Ray intersection
    if t > EPSILON {
        Some((t, u, v))
    } else {
        // Line intersection but not a ray intersection (behind ray origin)
        None
    }
}

/// Recursively tests a ray against a node and its descendants.
/// Uses cached bounding boxes to prune entire subtrees.
fn pick_node_recursive(
    ray: &Ray,
    node_id: NodeId,
    scene: &Scene,
    results: &mut Vec<PickResult>,
) {
    let node = scene.get_node(node_id)
        .expect("Node ID not found in scene during picking");

    // Broad phase: Test against this node's bounding box
    if let Some(bounds) = node.cached_bounds() {
        if bounds.intersects_ray(ray).is_none() {
            // Ray doesn't hit this node's bounds, skip entire subtree
            return;
        }
    }

    // If this node has an instance, test it
    if let Some(instance_id) = node.instance() {
        let instance = scene.instances.get(&instance_id)
            .expect("Instance referenced by node not found in scene");
        let mesh = scene.meshes.get(&instance.mesh)
            .expect("Mesh referenced by instance not found in scene");

        // Get world transform for this node
        let world_transform = node.cached_world_transform();
        let world_to_local = world_transform.invert()
            .expect("Node world transform is not invertible");

        // Transform ray to local mesh space
        let local_ray = ray.transform(&world_to_local);

        // Test against all triangles in the mesh
        let vertices = mesh.vertices();
        let indices = mesh.indices();

        // Iterate through triangles (indices come in groups of 3)
        for triangle_index in 0..(indices.len() / 3) {
            let i0 = indices[triangle_index * 3] as usize;
            let i1 = indices[triangle_index * 3 + 1] as usize;
            let i2 = indices[triangle_index * 3 + 2] as usize;

            let v0 = Point3::from(vertices[i0].position);
            let v1 = Point3::from(vertices[i1].position);
            let v2 = Point3::from(vertices[i2].position);

            // Test ray-triangle intersection in local space
            if let Some((t, u, v)) = intersect_ray_triangle(&local_ray, v0, v1, v2) {
                // Hit! Compute world-space hit point
                let local_hit_point = local_ray.point_at(t);
                let world_hit_point = {
                    let homogeneous = world_transform * local_hit_point.to_homogeneous();
                    Point3::from_homogeneous(homogeneous)
                };

                // Compute distance in world space from ray origin
                let distance = (world_hit_point - ray.origin).magnitude();

                let w = 1.0 - u - v;
                results.push(PickResult {
                    instance_id,
                    distance,
                    hit_point: world_hit_point,
                    triangle_index,
                    barycentric: (u, v, w),
                });
            }
        }
    }

    // Recurse to children
    for &child_id in node.children() {
        pick_node_recursive(ray, child_id, scene, results);
    }
}

/// Picks all instances intersected by a ray, sorted by distance from near to far.
///
/// This function performs a hierarchical two-phase picking algorithm:
/// 1. Broad phase: Test ray against node bounding boxes, pruning entire subtrees
/// 2. Narrow phase: Test ray against individual triangles for nodes that passed broad phase
///
/// The ray should be in world space. The function walks the scene tree from root nodes,
/// using cached bounding boxes to eliminate large portions of the scene efficiently.
///
/// Note: This function ensures bounding boxes and transforms are up-to-date before picking.
///
/// Returns a vector of PickResult sorted by distance (closest first).
pub fn pick_all_from_ray(ray: &Ray, scene: &Scene) -> Vec<PickResult> {
    // Ensure transforms are up-to-date (required for bounding box computation)
    collect_instance_transforms(scene);

    // Ensure bounding boxes are computed and cached
    super::bounding::compute_node_bounds(scene);

    let mut results = Vec::new();

    // Walk the tree from each root node
    for &root_id in scene.root_nodes() {
        pick_node_recursive(ray, root_id, scene, &mut results);
    }

    // Sort by distance (closest first)
    results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

    results
}
