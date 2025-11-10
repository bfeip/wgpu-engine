use cgmath::{InnerSpace, Matrix, Matrix3, Matrix4, Point3, Quaternion, SquareMatrix, Vector3};

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