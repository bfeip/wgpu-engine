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

    /// Creates a ray from a screen-space point, unprojecting it to world space.
    ///
    /// The ray originates at the camera's eye position and points through the specified
    /// screen pixel into the 3D world. This is useful for mouse picking and selection.
    ///
    /// # Arguments
    /// * `screen_x` - X coordinate in screen space (0 = left edge)
    /// * `screen_y` - Y coordinate in screen space (0 = top edge)
    /// * `screen_width` - Width of the screen/viewport in pixels
    /// * `screen_height` - Height of the screen/viewport in pixels
    /// * `camera` - Camera to use for unprojection
    ///
    /// # Returns
    /// A ray originating at the camera's eye position, pointing through the screen point
    /// into the 3D world.
    pub fn from_screen_point(
        screen_x: f32,
        screen_y: f32,
        screen_width: u32,
        screen_height: u32,
        camera: &crate::camera::Camera,
    ) -> Self {
        // Unproject points at near and far planes
        let world_near = camera.unproject_point_screen(
            screen_x,
            screen_y,
            0.0, // Near plane
            screen_width,
            screen_height,
        ).expect("Camera view-projection matrix should be invertible");

        let world_far = camera.unproject_point_screen(
            screen_x,
            screen_y,
            1.0, // Far plane
            screen_width,
            screen_height,
        ).expect("Camera view-projection matrix should be invertible");

        // Create ray from camera eye through the unprojected points
        // Direction is from near point to far point
        let direction = (world_far - world_near).normalize();

        Self {
            origin: world_near,
            direction,
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
        // For double-sided intersection, we check absolute value but then normalize
        if det.abs() < EPSILON {
            return None;
        }

        // For double-sided intersection, use absolute value of det to avoid sign flips
        let inv_det = 1.0 / det.abs();

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
    use cgmath::{Matrix3, Matrix4, Point3, Vector3, EuclideanSpace};

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

    // ===== compute_normal_matrix Tests =====

    #[test]
    fn test_normal_matrix_identity() {
        let identity = Matrix4::<f32>::identity();
        let normal_mat = compute_normal_matrix(&identity);
        let expected = Matrix3::<f32>::identity();

        // Compare each element
        for i in 0..3 {
            for j in 0..3 {
                let diff: f32 = normal_mat[i][j] - expected[i][j];
                assert!(diff.abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_normal_matrix_uniform_scale() {
        // Uniform scale by 2
        let scale = Matrix4::from_scale(2.0);
        let normal_mat = compute_normal_matrix(&scale);

        // For uniform scale s, normal matrix should be scaled by 1/s
        let expected_scale = 0.5;
        assert!((normal_mat[0][0] - expected_scale).abs() < EPSILON);
        assert!((normal_mat[1][1] - expected_scale).abs() < EPSILON);
        assert!((normal_mat[2][2] - expected_scale).abs() < EPSILON);
    }

    #[test]
    fn test_normal_matrix_non_uniform_scale() {
        use cgmath::Matrix4;
        // Non-uniform scale
        let scale = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let normal_mat = compute_normal_matrix(&scale);

        // Normal matrix should correct for non-uniform scaling
        // inv-transpose of diagonal should give reciprocals
        assert!((normal_mat[0][0] - 0.5).abs() < EPSILON);
        assert!((normal_mat[1][1] - (1.0/3.0)).abs() < EPSILON);
        assert!((normal_mat[2][2] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_normal_matrix_rotation() {
        use cgmath::{Matrix4, Rad};
        // Rotation should preserve the rotation in normal matrix
        let rotation = Matrix4::from_angle_z(Rad(std::f32::consts::PI / 4.0));
        let normal_mat = compute_normal_matrix(&rotation);

        // For pure rotation, normal matrix should equal rotation matrix (upper 3x3)
        for i in 0..3 {
            for j in 0..3 {
                assert!((normal_mat[i][j] - rotation[i][j]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_normal_matrix_combined_transforms() {
        use cgmath::{Matrix4, Rad};
        // Translation doesn't affect normal matrix, but rotation and scale do
        // Transform: T(10,20,30) * R_y(90°) * S(2)
        let transform = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0))
            * Matrix4::from_angle_y(Rad(std::f32::consts::PI / 2.0))
            * Matrix4::from_scale(2.0);

        let normal_mat = compute_normal_matrix(&transform);

        // Normal matrix = inverse-transpose of upper 3x3
        // For this transform:
        // - Translation is ignored (doesn't affect 3x3 portion)
        // - 90° rotation around Y: X→-Z, Y→Y, Z→X
        // - Uniform scale by 2 → normal matrix scaled by 1/2 = 0.5
        //
        // Expected normal matrix (in row-major for clarity):
        //   Row 0: [ ~0    0   0.5]
        //   Row 1: [  0   0.5   0 ]
        //   Row 2: [-0.5  0   ~0 ]
        //
        // In cgmath (column-major), matrix[col][row]:

        // Check key elements that define the transformation
        assert!((normal_mat[0][0]).abs() < EPSILON);           // col 0, row 0 ≈ 0
        assert!((normal_mat[0][2] - -0.5).abs() < 0.001);      // col 0, row 2 ≈ -0.5
        assert!((normal_mat[1][1] - 0.5).abs() < EPSILON);     // col 1, row 1 ≈ 0.5
        assert!((normal_mat[2][0] - 0.5).abs() < 0.001);       // col 2, row 0 ≈ 0.5
        assert!((normal_mat[2][2]).abs() < EPSILON);           // col 2, row 2 ≈ 0

        // Verify the matrix represents correct normal transformation behavior
        // A normal pointing in +X should transform correctly under rotation + scale
        let normal_x = Vector3::new(1.0, 0.0, 0.0);
        let transformed_x = normal_mat * normal_x;
        // After 90° Y rotation and scale: X → -Z direction, scaled by 0.5
        assert!((transformed_x.x).abs() < 0.001);
        assert!((transformed_x.y).abs() < 0.001);
        assert!((transformed_x.z - -0.5).abs() < 0.001);
    }

    #[test]
    fn test_normal_matrix_non_invertible() {
        // Zero scale matrix is non-invertible
        let zero_scale = Matrix4::from_scale(0.0);
        let normal_mat = compute_normal_matrix(&zero_scale);

        // Should return identity for non-invertible matrix
        let identity = Matrix3::<f32>::identity();
        for i in 0..3 {
            for j in 0..3 {
                let diff: f32 = normal_mat[i][j] - identity[i][j];
                assert!(diff.abs() < EPSILON);
            }
        }
    }

    // ===== decompose_matrix Tests =====

    #[test]
    fn test_decompose_identity() {
        let identity = Matrix4::identity();
        let (translation, rotation, scale) = decompose_matrix(&identity);

        assert!((translation.x - 0.0).abs() < EPSILON);
        assert!((translation.y - 0.0).abs() < EPSILON);
        assert!((translation.z - 0.0).abs() < EPSILON);

        // Identity quaternion is (w=1, x=0, y=0, z=0) or (s=1, v=0)
        assert!((rotation.s - 1.0).abs() < EPSILON || (rotation.s + 1.0).abs() < EPSILON);

        assert!((scale.x - 1.0).abs() < EPSILON);
        assert!((scale.y - 1.0).abs() < EPSILON);
        assert!((scale.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_translation_only() {
        let translation_vec = Vector3::new(5.0, 10.0, 15.0);
        let matrix = Matrix4::from_translation(translation_vec);
        let (translation, _rotation, scale) = decompose_matrix(&matrix);

        assert!((translation.x - 5.0).abs() < EPSILON);
        assert!((translation.y - 10.0).abs() < EPSILON);
        assert!((translation.z - 15.0).abs() < EPSILON);

        assert!((scale.x - 1.0).abs() < EPSILON);
        assert!((scale.y - 1.0).abs() < EPSILON);
        assert!((scale.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_rotation_only() {
        use cgmath::Rad;
        let matrix = Matrix4::from_angle_y(Rad(std::f32::consts::PI / 2.0));
        let (translation, rotation, scale) = decompose_matrix(&matrix);

        assert!((translation.x - 0.0).abs() < EPSILON);
        assert!((translation.y - 0.0).abs() < EPSILON);
        assert!((translation.z - 0.0).abs() < EPSILON);

        // Scale should be 1
        assert!((scale.x - 1.0).abs() < EPSILON);
        assert!((scale.y - 1.0).abs() < EPSILON);
        assert!((scale.z - 1.0).abs() < EPSILON);

        // Rotation should be 90° around Y axis
        // Quaternion for rotation around Y by θ: (cos(θ/2), 0, sin(θ/2), 0)
        // For θ = π/2: (cos(π/4), 0, sin(π/4), 0) = (√2/2, 0, √2/2, 0)
        let sqrt2_over_2 = std::f32::consts::SQRT_2 / 2.0;
        assert!((rotation.s - sqrt2_over_2).abs() < 0.001);  // scalar part
        assert!((rotation.v.x - 0.0).abs() < EPSILON);        // x component
        assert!((rotation.v.y - sqrt2_over_2).abs() < 0.001); // y component
        assert!((rotation.v.z - 0.0).abs() < EPSILON);        // z component
    }

    #[test]
    fn test_decompose_scale_only() {
        let matrix = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let (translation, _rotation, scale) = decompose_matrix(&matrix);

        assert!((translation.x - 0.0).abs() < EPSILON);
        assert!((translation.y - 0.0).abs() < EPSILON);
        assert!((translation.z - 0.0).abs() < EPSILON);

        assert!((scale.x - 2.0).abs() < EPSILON);
        assert!((scale.y - 3.0).abs() < EPSILON);
        assert!((scale.z - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_trs_composition() {
        use cgmath::Rad;
        // Create a TRS matrix
        let t = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        let r = Matrix4::from_angle_z(Rad(std::f32::consts::PI / 4.0));
        let s = Matrix4::from_scale(2.0);
        let trs = t * r * s;

        let (translation, rotation, scale) = decompose_matrix(&trs);

        assert!((translation.x - 1.0).abs() < EPSILON);
        assert!((translation.y - 2.0).abs() < EPSILON);
        assert!((translation.z - 3.0).abs() < EPSILON);

        assert!((scale.x - 2.0).abs() < EPSILON);
        assert!((scale.y - 2.0).abs() < EPSILON);
        assert!((scale.z - 2.0).abs() < EPSILON);

        // Rotation should be 45° around Z axis
        // Quaternion for rotation around Z by θ: (cos(θ/2), 0, 0, sin(θ/2))
        // For θ = π/4: (cos(π/8), 0, 0, sin(π/8))
        let half_angle = std::f32::consts::PI / 8.0;
        let expected_s = half_angle.cos();  // cos(π/8) ≈ 0.9239
        let expected_z = half_angle.sin();  // sin(π/8) ≈ 0.3827

        assert!((rotation.s - expected_s).abs() < 0.001);     // scalar part
        assert!((rotation.v.x - 0.0).abs() < EPSILON);        // x component
        assert!((rotation.v.y - 0.0).abs() < EPSILON);        // y component
        assert!((rotation.v.z - expected_z).abs() < 0.001);   // z component
    }

    #[test]
    fn test_decompose_negative_scale() {
        let matrix = Matrix4::from_nonuniform_scale(-1.0, 2.0, 3.0);
        let (_translation, _rotation, scale) = decompose_matrix(&matrix);

        // Negative scale should be preserved in magnitude
        assert!((scale.x.abs() - 1.0).abs() < EPSILON);
        assert!((scale.y - 2.0).abs() < EPSILON);
        assert!((scale.z - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_recompose_identity() {
        // Decompose and recompose should give us back the original (approximately)
        let original = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0))
            * Matrix4::from_scale(2.0);

        let (translation, rotation, scale) = decompose_matrix(&original);

        // Recompose: T * R * S
        let t_mat = Matrix4::from_translation(translation.to_vec());
        let r_mat = Matrix4::from(Matrix3::from(rotation));
        let s_mat = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);
        let recomposed = t_mat * r_mat * s_mat;

        // Compare matrices (they should be very close)
        for i in 0..4 {
            for j in 0..4 {
                let diff: f32 = original[i][j] - recomposed[i][j];
                assert!(diff.abs() < 0.001);
            }
        }
    }

    // ===== Additional Ray Tests =====

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
    fn test_ray_transform_preserves_direction_normalization() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 4.0, 0.0));
        let transform = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));

        let transformed = ray.transform(&transform);

        // Direction should still be normalized
        assert!((transformed.direction.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_ray_transform_with_rotation() {
        use cgmath::Rad;
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

    // ===== Additional AABB Tests =====

    #[test]
    fn test_aabb_corners() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 3.0));
        let corners = aabb.corners();

        assert_eq!(corners.len(), 8);

        // Verify min corner
        assert_eq!(corners[0], Point3::new(0.0, 0.0, 0.0));
        // Verify max corner
        assert_eq!(corners[7], Point3::new(1.0, 2.0, 3.0));

        // Verify all corners are within bounds
        for corner in corners.iter() {
            assert!(corner.x >= aabb.min.x && corner.x <= aabb.max.x);
            assert!(corner.y >= aabb.min.y && corner.y <= aabb.max.y);
            assert!(corner.z >= aabb.min.z && corner.z <= aabb.max.z);
        }
    }

    #[test]
    fn test_aabb_transform_rotation() {
        use cgmath::Rad;
        // Unit cube at origin
        let aabb = Aabb::new(Point3::new(-0.5, -0.5, -0.5), Point3::new(0.5, 0.5, 0.5));

        // Rotate 45 degrees around Z
        let rotation = Matrix4::from_angle_z(Rad(std::f32::consts::PI / 4.0));
        let transformed = aabb.transform(&rotation);

        // Rotated AABB should be larger in X and Y (but same in Z)
        let original_size_x = aabb.max.x - aabb.min.x;
        let transformed_size_x = transformed.max.x - transformed.min.x;

        assert!(transformed_size_x > original_size_x);

        // Z size should be unchanged
        let original_size_z = aabb.max.z - aabb.min.z;
        let transformed_size_z = transformed.max.z - transformed.min.z;
        assert!((transformed_size_z - original_size_z).abs() < EPSILON);
    }

    #[test]
    fn test_aabb_transform_scale() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let scale = Matrix4::from_scale(2.0);

        let transformed = aabb.transform(&scale);

        assert_eq!(transformed.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(transformed.max, Point3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_aabb_transform_translation() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let translation = Matrix4::from_translation(Vector3::new(5.0, 10.0, 15.0));

        let transformed = aabb.transform(&translation);

        assert_eq!(transformed.min, Point3::new(5.0, 10.0, 15.0));
        assert_eq!(transformed.max, Point3::new(6.0, 11.0, 16.0));
    }

    #[test]
    fn test_aabb_transform_combined() {
        use cgmath::Rad;
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));

        // Combine translation, rotation, and scale
        let transform = Matrix4::from_translation(Vector3::new(10.0, 0.0, 0.0))
            * Matrix4::from_angle_z(Rad(std::f32::consts::PI / 4.0))
            * Matrix4::from_scale(2.0);

        let transformed = aabb.transform(&transform);

        // Should have moved and grown
        assert!(transformed.max.x > aabb.max.x);
        assert!(transformed.min.x > aabb.min.x);
    }

    #[test]
    fn test_aabb_merge_disjoint() {
        // Two boxes that don't overlap
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let aabb2 = Aabb::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));

        let merged = aabb1.merge(&aabb2);

        // Should encompass both
        assert_eq!(merged.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(merged.max, Point3::new(6.0, 6.0, 6.0));
    }

    #[test]
    fn test_aabb_merge_contained() {
        // Small box inside large box
        let large = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let small = Aabb::new(Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0));

        let merged = large.merge(&small);

        // Should be same as large box
        assert_eq!(merged.min, large.min);
        assert_eq!(merged.max, large.max);
    }

    #[test]
    fn test_aabb_expand_internal_point() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        // Point already inside
        let internal_point = Point3::new(5.0, 5.0, 5.0);
        let expanded = aabb.expand(internal_point);

        // Should be unchanged
        assert_eq!(expanded.min, aabb.min);
        assert_eq!(expanded.max, aabb.max);
    }

    #[test]
    fn test_aabb_intersects_ray_grazing() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

        // Ray just touching the top edge
        let ray = Ray::new(Point3::new(-1.0, 1.0, 0.5), Vector3::new(1.0, 0.0, 0.0));

        let result = aabb.intersects_ray(&ray);
        assert!(result.is_some());
    }

    #[test]
    fn test_aabb_intersects_ray_from_corner() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

        // Ray starting at corner, shooting out
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(-1.0, -1.0, -1.0));

        let result = aabb.intersects_ray(&ray);
        // Should return 0 since origin is at corner (inside/on surface)
        assert!(result.is_some());
        assert!((result.unwrap() - 0.0).abs() < EPSILON);
    }
}