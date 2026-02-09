use cgmath::{InnerSpace, Matrix4, Point3};

use crate::common::{Aabb, Ray};
use crate::scene::{InstanceId, Mesh, NodeId, Scene};

use super::pick_query::{pick_all, PickQuery};

/// Result of a ray-instance intersection test.
#[derive(Debug, Clone)]
pub struct RayPickResult {
    /// The node that was hit
    pub node_id: NodeId,
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

/// Ray picking query that implements the generic PickQuery trait.
///
/// Wraps a Ray and the original world-space ray for distance calculations.
pub struct RayPickQuery {
    /// The ray in current coordinate space (may be transformed to local space)
    ray: Ray,
    /// The original ray in world space (for distance calculations)
    world_ray: Ray,
}

impl RayPickQuery {
    /// Creates a new ray pick query from a world-space ray.
    pub fn new(ray: Ray) -> Self {
        Self {
            ray,
            world_ray: ray,
        }
    }
}

impl PickQuery for RayPickQuery {
    type Result = RayPickResult;

    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool {
        bounds.intersects_ray(&self.ray).is_some()
    }

    fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        Self {
            ray: self.ray.transform(matrix),
            // Keep world_ray unchanged for distance calculations
            world_ray: self.world_ray,
        }
    }

    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node_id: NodeId,
        instance_id: InstanceId,
        world_transform: &Matrix4<f32>,
        results: &mut Vec<Self::Result>,
    ) {
        // Test against all triangles in the mesh (ray is already in local space)
        let mesh_hits = mesh.intersect_ray(&self.ray);

        // Transform hits to world space and add to results
        for mesh_hit in mesh_hits {
            // Transform hit point to world space
            let world_hit_point = {
                let homogeneous = world_transform * mesh_hit.hit_point.to_homogeneous();
                Point3::from_homogeneous(homogeneous)
            };

            // Compute distance in world space from original ray origin
            let distance = (world_hit_point - self.world_ray.origin).magnitude();

            results.push(RayPickResult {
                node_id,
                instance_id,
                distance,
                hit_point: world_hit_point,
                triangle_index: mesh_hit.triangle_index,
                barycentric: mesh_hit.barycentric,
            });
        }
    }
}

/// Picks all instances intersected by a ray, sorted by distance from near to far.
///
/// The ray should be in world space. The function walks the scene tree from root nodes,
/// using cached bounding boxes to eliminate large portions of the scene efficiently.
///
/// Returns a vector of RayPickResult sorted by distance (closest first).
pub fn pick_all_from_ray(ray: &Ray, scene: &Scene) -> Vec<RayPickResult> {
    let query = RayPickQuery::new(*ray);
    let mut results = pick_all(&query, scene);

    // Sort by distance (closest first)
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}
