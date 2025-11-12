use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Point3, Quaternion, SquareMatrix, Vector3};

/// Epsilon value for floating-point comparisons
pub const EPSILON: f32 = 1e-6;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RgbaColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32
}

/// Computes the normal matrix from a world transform matrix.
///
/// The normal matrix is the inverse-transpose of the upper-left 3x3 portion
/// of the transform matrix. This is necessary for correct normal transformation
/// when non-uniform scaling is present.
///
/// If the matrix is not invertible, returns an identity matrix.
pub fn compute_normal_matrix(world_transform: &Matrix4<f32>) -> Matrix3<f32> {
    // Extract the upper-left 3x3 matrix
    let mat3 = Matrix3::from([
        [world_transform[0][0], world_transform[0][1], world_transform[0][2]],
        [world_transform[1][0], world_transform[1][1], world_transform[1][2]],
        [world_transform[2][0], world_transform[2][1], world_transform[2][2]],
    ]);

    // Compute inverse-transpose
    match mat3.invert() {
        Some(inv) => inv.transpose(),
        None => {
            // If not invertible, use identity (shouldn't happen in practice)
            Matrix3::identity()
        }
    }
}

/// Decomposes a 4x4 transformation matrix into translation, rotation, and scale components.
///
/// # Arguments
/// * `matrix` - A 4x4 transformation matrix (column-major)
///
/// # Returns
/// A tuple of (translation, rotation, scale)
pub fn decompose_matrix(matrix: &Matrix4<f32>) -> (Point3<f32>, Quaternion<f32>, Vector3<f32>) {
    // Extract translation from the last column
    let translation = Point3::new(matrix[3][0], matrix[3][1], matrix[3][2]);

    // Extract basis vectors (first three columns)
    let basis_x = Vector3::new(matrix[0][0], matrix[0][1], matrix[0][2]);
    let basis_y = Vector3::new(matrix[1][0], matrix[1][1], matrix[1][2]);
    let basis_z = Vector3::new(matrix[2][0], matrix[2][1], matrix[2][2]);

    // Extract scale from the length of each basis vector
    let scale = Vector3::new(
        basis_x.magnitude(),
        basis_y.magnitude(),
        basis_z.magnitude(),
    );

    // Normalize basis vectors to remove scale, giving us the rotation matrix
    let rot_x = basis_x / scale.x;
    let rot_y = basis_y / scale.y;
    let rot_z = basis_z / scale.z;

    // Build rotation matrix from normalized basis vectors
    let rotation_matrix = Matrix3::from([rot_x.into(), rot_y.into(), rot_z.into()]);

    // Convert rotation matrix to quaternion
    let rotation: Quaternion<f32> = rotation_matrix.into();

    (translation, rotation, scale)
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
        if det.abs() < EPSILON {
            return None;
        }

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

/// An axis-aligned bounding box (AABB) in 3D space.
#[derive(Debug, Copy, Clone)]
pub struct Aabb {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl Aabb {
    /// Creates a new AABB from min and max points.
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }

    /// Creates an AABB that encompasses all the given points.
    /// Returns None if the points slice is empty.
    pub fn from_points(points: &[Point3<f32>]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min = points[0];
        let mut max = points[0];

        for point in points.iter().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            min.z = min.z.min(point.z);

            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
            max.z = max.z.max(point.z);
        }

        Some(Self { min, max })
    }

    pub fn corners(&self) -> [Point3<f32>; 8] {
        [
            Point3::new(self.min.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Transforms the AABB by the given 4x4 transformation matrix.
    /// This handles rotation/scaling/shearing by transforming all 8 corners
    /// and computing a new axis-aligned bounding box.
    pub fn transform(&self, matrix: &Matrix4<f32>) -> Self {
        // Transform all 8 corners of the box
        let corners = self.corners();

        let transformed_corners: Vec<Point3<f32>> = corners
            .iter()
            .map(|corner| {
                let homogeneous = matrix * corner.to_homogeneous();
                Point3::from_homogeneous(homogeneous)
            })
            .collect();

        // Unwrap is safe because we know we have 8 corners
        Self::from_points(&transformed_corners).unwrap()
    }

    /// Tests if a ray intersects this AABB using the slab method.
    /// Returns the t parameter of the intersection point if it hits, None otherwise.
    /// If the ray originates inside the box, returns Some(0.0).
    pub fn intersects_ray(&self, ray: &Ray) -> Option<f32> {
        #[derive(Copy, Clone)]
        enum Axis { X, Y, Z }

        let mut tmin = f32::NEG_INFINITY;
        let mut tmax = f32::INFINITY;

        // Test intersection with each pair of parallel planes
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let (origin_component, dir_component, min_component, max_component) = match axis {
                Axis::X => (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
                Axis::Y => (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
                Axis::Z => (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
            };

            if dir_component.abs() < EPSILON {
                // Ray is parallel to the slab
                if origin_component < min_component || origin_component > max_component {
                    return None;
                }
            } else {
                // Compute intersection t values
                let inv_dir = 1.0 / dir_component;
                let mut t1 = (min_component - origin_component) * inv_dir;
                let mut t2 = (max_component - origin_component) * inv_dir;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }

                tmin = tmin.max(t1);
                tmax = tmax.min(t2);

                if tmin > tmax {
                    return None;
                }
            }
        }

        // Return the near intersection point (or 0 if inside)
        if tmin >= 0.0 {
            Some(tmin)
        } else if tmax >= 0.0 {
            Some(0.0) // Ray origin is inside the box
        } else {
            None // Box is behind the ray
        }
    }

    /// Expands the AABB to include the given point.
    pub fn expand(&self, point: Point3<f32>) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            ),
            max: Point3::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            ),
        }
    }

    /// Merges this AABB with another, returning the bounding box that encompasses both.
    pub fn merge(&self, other: &Aabb) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Matrix4, Point3, Vector3};

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
    fn test_ray_transform() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let translation = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0));
        let transformed = ray.transform(&translation);

        assert!((transformed.origin.x - 5.0).abs() < EPSILON);
        assert!((transformed.direction.x - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_aabb_from_points() {
        let points = vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, 4.0, 2.0),
            Point3::new(2.0, 1.0, 5.0),
        ];
        let aabb = Aabb::from_points(&points).unwrap();

        assert_eq!(aabb.min, Point3::new(-1.0, 1.0, 2.0));
        assert_eq!(aabb.max, Point3::new(2.0, 4.0, 5.0));
    }

    #[test]
    fn test_aabb_from_empty_points() {
        let points: Vec<Point3<f32>> = vec![];
        assert!(Aabb::from_points(&points).is_none());
    }

    #[test]
    fn test_aabb_expand() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let expanded = aabb.expand(Point3::new(2.0, -1.0, 0.5));

        assert_eq!(expanded.min, Point3::new(0.0, -1.0, 0.0));
        assert_eq!(expanded.max, Point3::new(2.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_merge() {
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let aabb2 = Aabb::new(Point3::new(0.5, 0.5, 0.5), Point3::new(2.0, 2.0, 2.0));
        let merged = aabb1.merge(&aabb2);

        assert_eq!(merged.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(merged.max, Point3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_ray_aabb_intersection_hit() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        let t = aabb.intersects_ray(&ray);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < EPSILON); // Should hit at t=4
    }

    #[test]
    fn test_ray_aabb_intersection_miss() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, 5.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(aabb.intersects_ray(&ray).is_none());
    }

    #[test]
    fn test_ray_aabb_intersection_inside() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        let t = aabb.intersects_ray(&ray);
        assert!(t.is_some());
        assert!((t.unwrap() - 0.0).abs() < EPSILON); // Origin inside returns 0
    }

    #[test]
    fn test_ray_aabb_intersection_behind_ray() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(aabb.intersects_ray(&ray).is_none()); // Box is behind the ray
    }

    #[test]
    fn test_ray_aabb_parallel_to_plane() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        // Ray parallel to X axis, but outside the box in Y
        let ray = Ray::new(Point3::new(-5.0, 5.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(aabb.intersects_ray(&ray).is_none());
    }
}