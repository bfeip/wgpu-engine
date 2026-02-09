pub use wgpu_engine_scene::Camera;

/// Extension trait adding GPU-specific functionality to Camera.
pub(crate) trait CameraExt {
    /// Creates a GPU-compatible uniform buffer representation of this camera.
    fn to_uniform(&self) -> CameraUniform;
}

impl CameraExt for Camera {
    fn to_uniform(&self) -> CameraUniform {
        let mut ret = CameraUniform::new();
        ret.update_view_proj(self);
        ret
    }
}

/// GPU uniform buffer layout for camera data.
///
/// This struct is `#[repr(C)]` and implements `bytemuck::Pod` for direct
/// memory mapping to GPU buffers. It contains the view-projection matrix
/// and eye position, uploaded to bind group 0 for use by shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    /// Combined view-projection matrix (64 bytes, 4x4 f32).
    view_proj: [[f32; 4]; 4],
    /// Camera eye position in world space (for view direction calculation in PBR).
    eye_position: [f32; 3],
    /// Padding for 16-byte alignment.
    _padding: u32,
}

impl CameraUniform {
    /// Creates a new camera uniform initialized to identity matrix and origin.
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            eye_position: [0.0, 0.0, 0.0],
            _padding: 0,
        }
    }

    /// Updates the uniform from the given camera.
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
        self.eye_position = [camera.eye.x, camera.eye.y, camera.eye.z];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Point3, Vector3, Matrix4, SquareMatrix};

    const EPSILON: f32 = 1e-6;

    fn create_test_camera() -> Camera {
        Camera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    #[test]
    fn test_camera_uniform_new() {
        let uniform = CameraUniform::new();
        let identity = Matrix4::<f32>::identity();

        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(uniform.view_proj[i][j], identity[i][j]);
            }
        }
    }

    #[test]
    fn test_camera_uniform_update() {
        let camera = create_test_camera();
        let mut uniform = CameraUniform::new();

        uniform.update_view_proj(&camera);

        let expected_vp = camera.build_view_projection_matrix();

        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(uniform.view_proj[i][j], expected_vp[i][j]);
            }
        }
    }

    #[test]
    fn test_camera_uniform_layout() {
        use std::mem;
        assert_eq!(mem::size_of::<CameraUniform>(), 80);
        assert_eq!(mem::align_of::<CameraUniform>(), 4);
    }

    #[test]
    fn test_camera_to_uniform() {
        let camera = create_test_camera();
        let uniform = camera.to_uniform();

        let expected_vp = camera.build_view_projection_matrix();
        for i in 0..4 {
            for j in 0..4 {
                assert!((uniform.view_proj[i][j] - expected_vp[i][j]).abs() < EPSILON);
            }
        }
    }
}
