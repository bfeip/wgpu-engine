use cgmath::{InnerSpace, Matrix4, Point3, Vector3};

use crate::EPSILON;

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
        // Compute edges from v0
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;

        // Begin calculating determinant - also used to calculate u parameter
        let h = self.direction.cross(edge2);
        let det = edge1.dot(h);

        // If determinant is near zero, ray lies in plane of triangle or is parallel
        // For double-sided intersection, check if det is near zero (positive or negative)
        if det > -EPSILON && det < EPSILON {
            return None;
        }

        // For double-sided intersection, use det directly (handles both front and back faces)
        let inv_det = 1.0 / det;

        // Calculate distance from v0 to ray origin
        let s = self.origin - v0;

        // Calculate u parameter and test bounds
        let u = inv_det * s.dot(h);
        if u < 0.0 || u > 1.0 {
            return None;
        }

        // Prepare to test v parameter
        let q = s.cross(edge1);

        // Calculate v parameter and test bounds
        let v = inv_det * self.direction.dot(q);
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
}
