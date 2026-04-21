use cgmath::{InnerSpace, Matrix4, Point3, Vector3};

use crate::EPSILON;

/// The closest approach between a ray and a line segment.
///
/// Returned by [`Ray::closest_approach_to_segment`].
#[derive(Debug, Copy, Clone)]
pub struct SegmentApproach {
    /// Parameter along the ray at the closest approach point
    pub t: f32,
    /// Closest point on the segment to the ray
    pub closest_on_segment: Point3<f32>,
    /// Minimum 3D distance between the ray and the segment
    pub distance: f32,
}

/// A ray in 3D space, defined by an origin point and a direction vector.
#[derive(Debug, Copy, Clone)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>, // Should be normalized
}

impl Ray {
    /// Creates a new ray with the given origin and direction.
    /// The direction will be normalized automatically.
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Returns a point along the ray at parameter t.
    /// The point is calculated as: origin + t * direction
    pub fn point_at(&self, t: f32) -> Point3<f32> {
        self.origin + self.direction * t
    }

    /// Transforms the ray by the given 4x4 transformation matrix.
    pub fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        // Transform origin as a point (with w=1)
        let origin_homogeneous = matrix * self.origin.to_homogeneous();
        let new_origin = Point3::from_homogeneous(origin_homogeneous);

        // Transform direction as a vector (with w=0)
        let direction_vec4 = matrix * self.direction.extend(0.0);
        let new_direction = Vector3::new(direction_vec4.x, direction_vec4.y, direction_vec4.z);

        Self {
            origin: new_origin,
            direction: new_direction.normalize(),
        }
    }

    /// Finds the closest approach between the ray and a line segment using
    /// the Shoemake/Goldman two-segment closest-approach algorithm.
    ///
    /// Returns `Some((t, closest_on_segment, distance))` where:
    /// - `t`: parameter along the ray at the closest approach point
    /// - `closest_on_segment`: the point on the segment closest to the ray
    /// - `distance`: the minimum 3D distance between the ray and segment
    ///
    /// Returns `None` if the closest approach is behind the ray origin (t ≤ 0).
    pub fn closest_approach_to_segment(
        &self,
        p0: Point3<f32>,
        p1: Point3<f32>,
    ) -> Option<SegmentApproach> {
        let d = p1 - p0; // segment direction
        let w = self.origin - p0;

        let e = d.dot(d); // segment length squared
        let b = self.direction.dot(w);

        let seg_t;
        let ray_t;

        // b = dot(r, w) where w = origin - p0; the ray parameter at the projection of
        // p0 is -b (so degenerate/parallel cases use -b, not b).
        if e < EPSILON {
            // Degenerate segment: both endpoints are the same point — project p0 onto the ray
            seg_t = 0.0_f32;
            ray_t = -b;
        } else {
            let c = self.direction.dot(d);
            let f = d.dot(w);
            let denom = e - c * c; // = a*e - c*c where a=1 (ray direction is unit)

            if denom < EPSILON {
                // Ray and segment are parallel — project p0 onto the ray
                seg_t = 0.0_f32;
                ray_t = -b;
            } else {
                // Solving the 2×2 system from differentiating |P(s) - Q(t)|²:
                //   s = (c*f - b*e) / denom
                //   t = (f - b*c)   / denom
                let raw_ray_t = (c * f - b * e) / denom;
                let raw_seg_t = (f - b * c) / denom;
                seg_t = raw_seg_t.clamp(0.0, 1.0);
                // If seg_t was clamped, recompute ray_t against the clamped endpoint
                if (raw_seg_t - seg_t).abs() > EPSILON {
                    let clamped_point = p0 + d * seg_t;
                    ray_t = self.direction.dot(clamped_point - self.origin);
                } else {
                    ray_t = raw_ray_t;
                }
            }
        }

        if ray_t <= 0.0 {
            return None;
        }

        let closest_on_segment = p0 + d * seg_t;
        let closest_on_ray = self.point_at(ray_t);
        let distance = (closest_on_ray - closest_on_segment).magnitude();

        Some(SegmentApproach { t: ray_t, closest_on_segment, distance })
    }

    /// Tests if a ray intersects a triangle using the Möller-Trumbore algorithm.
    ///
    /// Returns Some((t, u, v)) if the ray hits the triangle, where:
    /// - t: distance along the ray
    /// - u, v: barycentric coordinates (w = 1 - u - v)
    ///
    /// Returns None if there's no intersection or if the intersection is behind the ray origin.
    pub fn intersect_triangle(
        &self,
        v0: Point3<f32>,
        v1: Point3<f32>,
        v2: Point3<f32>,
    ) -> Option<(f32, f32, f32)> {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = self.direction.cross(edge2);
        let det = edge1.dot(h);

        // Reject degenerate triangles (ray parallel to triangle plane).
        // Use a very small threshold to avoid rejecting thin-but-valid triangles.
        const DET_EPSILON: f32 = 1e-10;
        if det > -DET_EPSILON && det < DET_EPSILON {
            return None;
        }

        let s = self.origin - v0;
        let s_dot_h = s.dot(h);

        // Division-free bounds check for u parameter.
        // Equivalent to: u = s_dot_h / det; if u < 0 || u > 1 { reject }
        // but without dividing by the (potentially small) determinant.
        if det > 0.0 {
            if s_dot_h < 0.0 || s_dot_h > det {
                return None;
            }
        } else if s_dot_h > 0.0 || s_dot_h < det {
            return None;
        }

        let q = s.cross(edge1);
        let dir_dot_q = self.direction.dot(q);

        // Division-free bounds check for v and u+v parameters.
        if det > 0.0 {
            if dir_dot_q < 0.0 || s_dot_h + dir_dot_q > det {
                return None;
            }
        } else if dir_dot_q > 0.0 || s_dot_h + dir_dot_q < det {
            return None;
        }

        // Compute final values via division (only on the hit path).
        let inv_det = 1.0 / det;
        let t = inv_det * edge2.dot(q);

        if t > EPSILON {
            let u = s_dot_h * inv_det;
            let v = dir_dot_q * inv_det;
            Some((t, u, v))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, Point3, Vector3, Rad};

    #[test]
    fn test_ray_creation_normalizes_direction() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 4.0, 0.0));
        assert!((ray.direction.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_point_at() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 0.0, 0.0));
        let point = ray.point_at(5.0);
        assert!((point.x - 6.0).abs() < EPSILON);
        assert!((point.y - 2.0).abs() < EPSILON);
        assert!((point.z - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_point_at_zero() {
        let ray = Ray::new(Point3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 0.0, 0.0));
        let point = ray.point_at(0.0);
        assert_eq!(point, ray.origin);
    }

    #[test]
    fn test_ray_point_at_negative() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let point = ray.point_at(-5.0);
        assert!((point.x - -5.0).abs() < EPSILON);
        assert!((point.y - 0.0).abs() < EPSILON);
        assert!((point.z - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_transform_translation() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let translation = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0));
        let transformed = ray.transform(&translation);

        assert!((transformed.origin.x - 5.0).abs() < EPSILON);
        assert!((transformed.direction.x - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_transform_preserves_direction_normalization() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 4.0, 0.0));
        let transform = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));

        let transformed = ray.transform(&transform);

        // Direction should still be normalized
        assert!((transformed.direction.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_transform_with_rotation() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        // Rotate 90 degrees around Z axis
        let rotation = Matrix4::from_angle_z(Rad(std::f32::consts::PI / 2.0));
        let transformed = ray.transform(&rotation);

        // Direction should now point in +Y
        assert!((transformed.direction.x - 0.0).abs() < EPSILON);
        assert!((transformed.direction.y - 1.0).abs() < 0.001);
        assert!((transformed.direction.z - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_transform_with_scale() {
        let ray = Ray::new(Point3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let scale = Matrix4::from_scale(2.0);

        let transformed = ray.transform(&scale);

        // Origin should be scaled
        assert!((transformed.origin.x - 2.0).abs() < EPSILON);
        // Direction should still be normalized
        assert!((transformed.direction.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_transform_combined() {
        let ray = Ray::new(Point3::new(1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        // Combined transform: translate, rotate, scale
        let transform = Matrix4::from_translation(Vector3::new(0.0, 5.0, 0.0))
            * Matrix4::from_angle_z(Rad(std::f32::consts::PI / 2.0))
            * Matrix4::from_scale(2.0);

        let transformed = ray.transform(&transform);

        // Origin should be transformed
        assert!(transformed.origin.x.abs() < 0.01); // ~0 after rotation
        assert!((transformed.origin.y - 7.0).abs() < 0.01); // 5 + 2*1*sin(90) = 7

        // Direction should be rotated and normalized
        assert!((transformed.direction.magnitude() - 1.0).abs() < EPSILON);
    }

    // ===== Triangle Intersection Tests =====

    #[test]
    fn test_ray_triangle_intersection_hit() {
        // Simple triangle in XY plane at z=0
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray shooting from -Z toward triangle center
        let ray = Ray::new(Point3::new(0.25, 0.25, -1.0), Vector3::new(0.0, 0.0, 1.0));

        let result = ray.intersect_triangle(v0, v1, v2);
        assert!(result.is_some());

        let (t, u, v) = result.unwrap();
        assert!(t > 0.0); // Hit is in front of ray
        assert!(u >= 0.0 && u <= 1.0);
        assert!(v >= 0.0 && v <= 1.0);
        assert!(u + v <= 1.0);
    }

    #[test]
    fn test_ray_triangle_intersection_miss() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray shooting away from triangle
        let ray = Ray::new(Point3::new(0.5, 0.5, -1.0), Vector3::new(0.0, 0.0, -1.0));

        assert!(ray.intersect_triangle(v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_triangle_intersection_parallel() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray parallel to triangle plane (XY plane)
        let ray = Ray::new(Point3::new(0.5, 0.5, 1.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(ray.intersect_triangle(v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_triangle_intersection_behind_ray() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray origin is past the triangle
        let ray = Ray::new(Point3::new(0.25, 0.25, 1.0), Vector3::new(0.0, 0.0, 1.0));

        assert!(ray.intersect_triangle(v0, v1, v2).is_none());
    }

    #[test]
    fn test_ray_triangle_intersection_edge_cases() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray hitting exactly at v0
        let ray = Ray::new(Point3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 0.0, 1.0));
        let result = ray.intersect_triangle(v0, v1, v2);
        assert!(result.is_some());

        let (_t, u, v) = result.unwrap();
        // At v0: u=0, v=0
        assert!(u.abs() < EPSILON);
        assert!(v.abs() < EPSILON);
    }

    #[test]
    fn test_ray_triangle_intersection_at_vertex() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Test hitting each vertex
        let vertices = vec![
            (v0, 0.0, 0.0),
            (v1, 1.0, 0.0),
            (v2, 0.0, 1.0),
        ];

        for (vertex, expected_u, expected_v) in vertices {
            let ray = Ray::new(
                Point3::new(vertex.x, vertex.y, -1.0),
                Vector3::new(0.0, 0.0, 1.0)
            );

            let result = ray.intersect_triangle(v0, v1, v2);
            assert!(result.is_some(), "Should hit vertex {:?}", vertex);

            let (_t, u, v) = result.unwrap();
            assert!((u - expected_u).abs() < 0.01, "u mismatch at vertex {:?}", vertex);
            assert!((v - expected_v).abs() < 0.01, "v mismatch at vertex {:?}", vertex);
        }
    }

    #[test]
    fn test_ray_triangle_barycentric_coordinates() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray hitting triangle center
        let ray = Ray::new(Point3::new(1.0/3.0, 1.0/3.0, -1.0), Vector3::new(0.0, 0.0, 1.0));

        let result = ray.intersect_triangle(v0, v1, v2);
        assert!(result.is_some());

        let (_t, u, v) = result.unwrap();
        let w = 1.0 - u - v;

        // At center: u ≈ 1/3, v ≈ 1/3, w ≈ 1/3
        assert!((u - 1.0/3.0).abs() < 0.01);
        assert!((v - 1.0/3.0).abs() < 0.01);
        assert!((w - 1.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_ray_triangle_double_sided() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray from front
        let ray_front = Ray::new(Point3::new(0.25, 0.25, -1.0), Vector3::new(0.0, 0.0, 1.0));
        assert!(ray_front.intersect_triangle(v0, v1, v2).is_some());

        // Ray from back (should also hit with double-sided intersection)
        let ray_back = Ray::new(Point3::new(0.25, 0.25, 1.0), Vector3::new(0.0, 0.0, -1.0));
        assert!(ray_back.intersect_triangle(v0, v1, v2).is_some());
    }

    #[test]
    fn test_ray_triangle_distance_calculation() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Ray 5 units away from triangle
        let ray = Ray::new(Point3::new(0.25, 0.25, -5.0), Vector3::new(0.0, 0.0, 1.0));

        let result = ray.intersect_triangle(v0, v1, v2);
        assert!(result.is_some());

        let (t, _u, _v) = result.unwrap();
        // Distance should be approximately 5
        assert!((t - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_ray_hits_thin_cylinder_triangle() {
        // Simulates a single face of a 32-segment cylinder with extreme values.
        // This is the geometry formerly caused precision failures with the
        // division-based Möller–Trumbore bounds check.
        use std::f32::consts::PI;
        let radius: f32 = 1.0e-10;
        let height: f32 = 10000.0;
        let segments = 32;

        let angle0 = 2.0 * PI * 0.0 / segments as f32;
        let angle1 = 2.0 * PI * 1.0 / segments as f32;

        let v0 = Point3::new(radius * angle0.cos(), radius * angle0.sin(), 0.0);
        let v1 = Point3::new(radius * angle1.cos(), radius * angle1.sin(), 0.0);
        let v2 = Point3::new(radius * angle0.cos(), radius * angle0.sin(), height);

        // Ray aimed at the midpoint of the triangle from a diagonal direction
        let mid = Point3::new(
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        );
        let origin = Point3::new(mid.x + 1.0, mid.y + 1.0, mid.z + 1.0);
        let direction = Vector3::new(mid.x - origin.x, mid.y - origin.y, mid.z - origin.z);
        let ray = Ray::new(origin, direction);

        let result = ray.intersect_triangle(v0, v1, v2);
        assert!(
            result.is_some(),
            "Ray should hit thin cylinder triangle (r={}, segments={})",
            radius, segments
        );
    }

    #[test]
    fn test_ray_triangle_outside_bounds() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);

        // Rays that would hit the plane but miss the triangle
        let test_rays = vec![
            Ray::new(Point3::new(-0.5, 0.5, -1.0), Vector3::new(0.0, 0.0, 1.0)), // Left of triangle
            Ray::new(Point3::new(0.5, -0.5, -1.0), Vector3::new(0.0, 0.0, 1.0)), // Below triangle
            Ray::new(Point3::new(0.6, 0.6, -1.0), Vector3::new(0.0, 0.0, 1.0)),  // Above hypotenuse
        ];

        for ray in test_rays {
            assert!(
                ray.intersect_triangle(v0, v1, v2).is_none(),
                "Ray {:?} should miss triangle", ray
            );
        }
    }

    // ===== Segment Closest Approach Tests =====

    #[test]
    fn closest_approach_perpendicular() {
        // Ray along +Z from origin; segment along X at z=5
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let p0 = Point3::new(-1.0, 0.0, 5.0);
        let p1 = Point3::new(1.0, 0.0, 5.0);
        let a = ray.closest_approach_to_segment(p0, p1).unwrap();
        assert!((a.t - 5.0).abs() < EPSILON, "t should be 5, got {}", a.t);
        assert!((a.closest_on_segment.x).abs() < EPSILON);
        assert!((a.closest_on_segment.z - 5.0).abs() < EPSILON);
        assert!(a.distance < EPSILON, "dist should be 0, got {}", a.distance);
    }

    #[test]
    fn closest_approach_offset() {
        // Ray along +Z at x=0; segment parallel to ray at x=3, starting ahead of the origin
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let p0 = Point3::new(3.0, 0.0, 2.0);
        let p1 = Point3::new(3.0, 0.0, 10.0);
        let a = ray.closest_approach_to_segment(p0, p1).unwrap();
        assert!((a.distance - 3.0).abs() < EPSILON, "dist should be 3, got {}", a.distance);
    }

    #[test]
    fn closest_approach_degenerate_segment() {
        // Segment with both endpoints at the same position (zero length)
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let p = Point3::new(1.0, 0.0, 5.0);
        let a = ray.closest_approach_to_segment(p, p).unwrap();
        assert!((a.t - 5.0).abs() < EPSILON);
        assert!((a.closest_on_segment.x - 1.0).abs() < EPSILON);
        assert!((a.distance - 1.0).abs() < EPSILON);
    }

    #[test]
    fn closest_approach_clamped_to_endpoint() {
        // Segment does not extend to the closest point on the ray — should clamp to p1
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let p0 = Point3::new(5.0, 3.0, 0.0);
        let p1 = Point3::new(5.0, 10.0, 0.0); // both endpoints above the ray
        let a = ray.closest_approach_to_segment(p0, p1).unwrap();
        // Closest point on segment to ray is p0 (nearest endpoint)
        assert!((a.closest_on_segment.x - 5.0).abs() < EPSILON);
        assert!((a.closest_on_segment.y - 3.0).abs() < EPSILON);
        assert!((a.distance - 3.0).abs() < EPSILON);
    }

    #[test]
    fn closest_approach_behind_ray_returns_none() {
        // Segment is entirely behind the ray origin
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
        let p0 = Point3::new(-1.0, 0.0, -5.0);
        let p1 = Point3::new(1.0, 0.0, -5.0);
        assert!(ray.closest_approach_to_segment(p0, p1).is_none());
    }
}
