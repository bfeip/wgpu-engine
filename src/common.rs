use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Point3, Quaternion, SquareMatrix, Vector3};

/// Epsilon value for floating-point comparisons
pub const EPSILON: f32 = 1e-6;

mod ray;
mod aabb;
mod plane;
mod convex_polyhedron;

// Re-export common types
pub use ray::Ray;
pub use aabb::Aabb;
pub use plane::Plane;
pub use convex_polyhedron::ConvexPolyhedron;

/// An RGBA color, with values between 0.0 and 1.0
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


#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{EuclideanSpace, Matrix3, Matrix4, Vector3};

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

}