use cgmath::{Matrix3, Matrix4, SquareMatrix};

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
    let mat3 = Matrix3::new(
        world_transform[0][0],
        world_transform[0][1],
        world_transform[0][2],
        world_transform[1][0],
        world_transform[1][1],
        world_transform[1][2],
        world_transform[2][0],
        world_transform[2][1],
        world_transform[2][2],
    );

    // Compute inverse-transpose
    match mat3.invert() {
        Some(inv) => {
            // Transpose by accessing columns as rows
            Matrix3::new(
                inv[0][0], inv[1][0], inv[2][0], // First row = first column of inv
                inv[0][1], inv[1][1], inv[2][1], // Second row = second column of inv
                inv[0][2], inv[1][2], inv[2][2], // Third row = third column of inv
            )
        }
        None => {
            // If not invertible, use identity (shouldn't happen in practice)
            Matrix3::identity()
        }
    }
}