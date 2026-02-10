use cgmath::{EuclideanSpace, InnerSpace, Matrix, Matrix4, Point3, SquareMatrix, Vector3, Vector4};

use crate::EPSILON;

/// A plane in 3D space defined by a normal and distance from origin.
///
/// The plane equation is: normal · point + distance = 0
/// Points with negative signed distance are "inside" (on the normal's opposite side).
/// This convention is useful for convex polyhedra where "inside" means inside the volume.
#[derive(Debug, Copy, Clone)]
pub struct Plane {
    /// Unit normal pointing "outside" the half-space
    pub normal: Vector3<f32>,
    /// Signed distance from origin along the normal
    pub distance: f32,
}

impl Plane {
    /// Creates a new plane from a normal vector and a point on the plane.
    /// The normal will be normalized automatically.
    pub fn new(normal: Vector3<f32>, point: Point3<f32>) -> Self {
        let normal = normal.normalize();
        let distance = -normal.dot(point.to_vec());
        Self { normal, distance }
    }

    /// Creates a plane from the coefficients of the plane equation ax + by + cz + d = 0.
    /// The coefficients will be normalized so that (a, b, c) becomes a unit vector.
    pub fn from_coefficients(a: f32, b: f32, c: f32, d: f32) -> Self {
        let length = (a * a + b * b + c * c).sqrt();
        if length < EPSILON {
            // Degenerate plane, return a default
            return Self {
                normal: Vector3::new(0.0, 1.0, 0.0),
                distance: 0.0,
            };
        }
        Self {
            normal: Vector3::new(a / length, b / length, c / length),
            distance: d / length,
        }
    }

    /// Computes the signed distance from a point to the plane.
    ///
    /// - Positive: point is on the "outside" (same side as normal)
    /// - Zero: point is on the plane
    /// - Negative: point is on the "inside" (opposite side from normal)
    pub fn signed_distance(&self, point: Point3<f32>) -> f32 {
        self.normal.dot(point.to_vec()) + self.distance
    }

    /// Returns true if the point is on the inside (negative side) of the plane.
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        self.signed_distance(point) <= EPSILON
    }

    /// Transforms the plane by a 4x4 transformation matrix.
    ///
    /// For correct plane transformation, we need to use the inverse-transpose
    /// of the matrix to transform the plane normal correctly under non-uniform scaling.
    pub fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        // Planes transform by the inverse-transpose of the matrix.
        // A plane can be represented as a 4D vector (a, b, c, d) where ax + by + cz + d = 0.
        // If M transforms points, then M^(-T) transforms planes.
        let inv_transpose = matrix
            .invert()
            .unwrap_or(Matrix4::identity())
            .transpose();

        let plane_vec = Vector4::new(self.normal.x, self.normal.y, self.normal.z, self.distance);
        let transformed = inv_transpose * plane_vec;

        // Re-normalize the result
        Self::from_coefficients(transformed.x, transformed.y, transformed.z, transformed.w)
    }

    /// Finds the intersection point of a line segment with the plane.
    /// Returns Some((t, point)) where t is the parameter (0 to 1) and point is the intersection.
    /// Returns None if the segment is parallel to the plane or doesn't intersect.
    pub fn intersect_segment(
        &self,
        start: Point3<f32>,
        end: Point3<f32>,
    ) -> Option<(f32, Point3<f32>)> {
        let direction = end - start;
        let denom = self.normal.dot(direction);

        // Check if segment is parallel to plane
        if denom.abs() < EPSILON {
            return None;
        }

        let t = -(self.normal.dot(start.to_vec()) + self.distance) / denom;

        // Check if intersection is within segment bounds
        if t < 0.0 || t > 1.0 {
            return None;
        }

        let point = start + direction * t;
        Some((t, point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Rad;

    #[test]
    fn test_plane_from_normal_and_point() {
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::new(0.0, 5.0, 0.0));

        // Normal should be normalized (already unit)
        assert!((plane.normal.magnitude() - 1.0).abs() < EPSILON);

        // Point on plane should have zero distance
        assert!(plane.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
        assert!(plane.signed_distance(Point3::new(10.0, 5.0, -3.0)).abs() < EPSILON);

        // Points above should be positive
        assert!(plane.signed_distance(Point3::new(0.0, 6.0, 0.0)) > 0.0);

        // Points below should be negative
        assert!(plane.signed_distance(Point3::new(0.0, 4.0, 0.0)) < 0.0);
    }

    #[test]
    fn test_plane_from_coefficients() {
        // Plane y = 5, or 0x + 1y + 0z - 5 = 0
        let plane = Plane::from_coefficients(0.0, 1.0, 0.0, -5.0);

        assert!((plane.normal.y - 1.0).abs() < EPSILON);
        assert!(plane.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_plane_from_coefficients_normalizes() {
        // Non-normalized coefficients: 0, 2, 0, -10 (equivalent to y = 5)
        let plane = Plane::from_coefficients(0.0, 2.0, 0.0, -10.0);

        assert!((plane.normal.magnitude() - 1.0).abs() < EPSILON);
        assert!((plane.normal.y - 1.0).abs() < EPSILON);
        assert!(plane.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_plane_signed_distance() {
        // XZ plane at origin (normal pointing +Y)
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        assert!((plane.signed_distance(Point3::new(0.0, 0.0, 0.0))).abs() < EPSILON);
        assert!((plane.signed_distance(Point3::new(0.0, 3.0, 0.0)) - 3.0).abs() < EPSILON);
        assert!((plane.signed_distance(Point3::new(0.0, -2.0, 0.0)) - -2.0).abs() < EPSILON);
    }

    #[test]
    fn test_plane_contains_point() {
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Points on or below the plane are "inside"
        assert!(plane.contains_point(Point3::new(0.0, 0.0, 0.0)));
        assert!(plane.contains_point(Point3::new(0.0, -1.0, 0.0)));
        assert!(plane.contains_point(Point3::new(5.0, -0.5, 3.0)));

        // Points above are "outside"
        assert!(!plane.contains_point(Point3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn test_plane_transform_translation() {
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());
        let translation = Matrix4::from_translation(Vector3::new(0.0, 5.0, 0.0));

        let transformed = plane.transform(&translation);

        // Normal should remain the same
        assert!((transformed.normal.y - 1.0).abs() < EPSILON);

        // Plane should now pass through y=5
        assert!(transformed.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_plane_transform_rotation() {
        // XZ plane (normal +Y)
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Rotate 90 degrees around X axis: Y -> Z
        let rotation = Matrix4::from_angle_x(Rad(std::f32::consts::FRAC_PI_2));
        let transformed = plane.transform(&rotation);

        // Normal should now point in +Z direction
        assert!(transformed.normal.x.abs() < 0.01);
        assert!(transformed.normal.y.abs() < 0.01);
        assert!((transformed.normal.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_plane_intersect_segment_hit() {
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        let start = Point3::new(0.0, -1.0, 0.0);
        let end = Point3::new(0.0, 1.0, 0.0);

        let result = plane.intersect_segment(start, end);
        assert!(result.is_some());

        let (t, point) = result.unwrap();
        assert!((t - 0.5).abs() < EPSILON);
        assert!(point.y.abs() < EPSILON);
    }

    #[test]
    fn test_plane_intersect_segment_miss_parallel() {
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Segment parallel to plane
        let start = Point3::new(0.0, 1.0, 0.0);
        let end = Point3::new(1.0, 1.0, 0.0);

        assert!(plane.intersect_segment(start, end).is_none());
    }

    #[test]
    fn test_plane_intersect_segment_miss_outside() {
        let plane = Plane::new(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Segment entirely above plane
        let start = Point3::new(0.0, 1.0, 0.0);
        let end = Point3::new(0.0, 2.0, 0.0);

        assert!(plane.intersect_segment(start, end).is_none());
    }
}
