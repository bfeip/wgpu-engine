use cgmath::{InnerSpace, Matrix4, Point3};

use crate::common::{Aabb, Ray};
use crate::{InstanceId, Mesh, NodeId, Scene};

use super::mesh_intersection;
use super::pick_query::{pick_all, PickQuery};

/// The primitive that was hit by a ray pick query.
#[derive(Debug, Clone)]
pub enum RayHit {
    /// A triangle face was hit.
    Triangle {
        /// Index of the triangle (0-based, into the mesh's triangle list)
        triangle_index: usize,
        /// Barycentric coordinates of the hit point (u, v, w) where w = 1 - u - v
        barycentric: (f32, f32, f32),
    },
    /// A line segment was hit within the query's tolerance.
    Segment {
        /// Index of the segment (0-based pair index into the mesh's line index buffer)
        segment_index: usize,
        /// World-space perpendicular distance from the ray to the segment
        distance_to_ray: f32,
    },
}

/// Result of a ray pick query against a scene instance.
///
/// For `RayHit::Triangle`, `hit_point` is the exact ray-triangle intersection point.
/// For `RayHit::Segment`, `hit_point` is the closest point on the segment to the ray.
#[derive(Debug, Clone)]
pub struct RayPickResult {
    /// The node that was hit
    pub node_id: NodeId,
    /// The instance that was hit
    pub instance_id: InstanceId,
    /// World-space distance from the ray origin to the hit (used for depth sorting)
    pub distance: f32,
    /// World-space hit location
    pub hit_point: Point3<f32>,
    /// Which primitive was hit and its geometry-specific data
    pub hit: RayHit,
}

/// Ray picking query that implements the generic PickQuery trait.
///
/// Construct with [`RayPickQuery::new`] for face-only picking, [`RayPickQuery::lines`]
/// for line-only picking, or [`RayPickQuery::all`] to pick both faces and lines.
pub struct RayPickQuery {
    /// The ray in current coordinate space (may be transformed to local space)
    ray: Ray,
    /// The original ray in world space (for distance calculations)
    world_ray: Ray,
    pick_faces: bool,
    pick_lines: bool,
    /// World-space tolerance for line picking
    line_tolerance: f32,
    /// Tolerance scaled to the current (possibly local) coordinate space
    local_line_tolerance: f32,
}

impl RayPickQuery {
    /// Creates a face-only query. Lines are ignored.
    pub fn faces(ray: Ray) -> Self {
        Self {
            ray,
            world_ray: ray,
            pick_faces: true,
            pick_lines: false,
            line_tolerance: 0.0,
            local_line_tolerance: 0.0,
        }
    }

    /// Creates a line-only query. Triangles are ignored.
    ///
    /// `tolerance` is the maximum world-space distance between the ray and a segment
    /// for the segment to be considered a hit.
    pub fn lines(ray: Ray, tolerance: f32) -> Self {
        Self {
            ray,
            world_ray: ray,
            pick_faces: false,
            pick_lines: true,
            line_tolerance: tolerance,
            local_line_tolerance: tolerance,
        }
    }

    /// Creates a query that picks both triangle faces and line segments.
    ///
    /// `line_tolerance` is the maximum world-space distance for segment hits.
    pub fn all(ray: Ray, line_tolerance: f32) -> Self {
        Self {
            ray,
            world_ray: ray,
            pick_faces: true,
            pick_lines: true,
            line_tolerance,
            local_line_tolerance: line_tolerance,
        }
    }
}

impl PickQuery for RayPickQuery {
    type Result = RayPickResult;

    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool {
        if self.pick_faces && bounds.intersects_ray(&self.ray).is_some() {
            return true;
        }
        if self.pick_lines {
            // Inflate bounds by the line tolerance so segments near-but-outside the
            // AABB are not incorrectly culled during the broad phase.
            let t = self.local_line_tolerance;
            let inflated = Aabb::new(
                Point3::new(bounds.min.x - t, bounds.min.y - t, bounds.min.z - t),
                Point3::new(bounds.max.x + t, bounds.max.y + t, bounds.max.z + t),
            );
            return inflated.intersects_ray(&self.ray).is_some();
        }
        false
    }

    fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        // Scale the line tolerance into the local coordinate space.
        // world_to_local (= matrix) embeds the inverse of the world scale, so a
        // world-space distance d maps to d * column_magnitude in local space.
        // We take the max column magnitude to stay conservative for non-uniform scale.
        let scale = [matrix.x, matrix.y, matrix.z]
            .iter()
            .map(|col| col.truncate().magnitude())
            .fold(0.0_f32, f32::max);

        Self {
            ray: self.ray.transform(matrix),
            world_ray: self.world_ray,
            pick_faces: self.pick_faces,
            pick_lines: self.pick_lines,
            line_tolerance: self.line_tolerance,
            local_line_tolerance: self.line_tolerance * scale,
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
        if self.pick_faces {
            for mesh_hit in mesh_intersection::intersect_ray(mesh, &self.ray) {
                let world_hit_point =
                    cgmath::Transform::transform_point(world_transform, mesh_hit.hit_point);
                let distance = (world_hit_point - self.world_ray.origin).magnitude();
                results.push(RayPickResult {
                    node_id,
                    instance_id,
                    distance,
                    hit_point: world_hit_point,
                    hit: RayHit::Triangle {
                        triangle_index: mesh_hit.triangle_index,
                        barycentric: mesh_hit.barycentric,
                    },
                });
            }
        }

        if self.pick_lines {
            for line_hit in
                mesh_intersection::intersect_ray_with_lines(mesh, &self.ray, self.local_line_tolerance)
            {
                let world_closest =
                    cgmath::Transform::transform_point(world_transform, line_hit.closest_point);
                let distance = (world_closest - self.world_ray.origin).magnitude();

                // Compute world-space perpendicular distance by comparing the closest
                // point on the segment against the local-space ray point at the same t,
                // then transforming both to world space.
                let local_ray_point = self.ray.point_at(line_hit.t);
                let world_ray_point =
                    cgmath::Transform::transform_point(world_transform, local_ray_point);
                let distance_to_ray = (world_closest - world_ray_point).magnitude();

                results.push(RayPickResult {
                    node_id,
                    instance_id,
                    distance,
                    hit_point: world_closest,
                    hit: RayHit::Segment {
                        segment_index: line_hit.segment_index,
                        distance_to_ray,
                    },
                });
            }
        }
    }
}

/// Picks all instances intersected by a ray, sorted by distance (closest first).
///
/// Use [`RayPickQuery::new`] to pick faces only, [`RayPickQuery::lines`] to pick
/// segments only, or [`RayPickQuery::all`] to pick both.
pub fn pick_all_from_ray(query: &RayPickQuery, scene: &Scene) -> Vec<RayPickResult> {
    let mut results = pick_all(query, scene);
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}
