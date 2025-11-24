#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    pub target: cgmath::Point3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }

    /// Returns the camera's forward vector
    pub fn forward(&self) -> cgmath::Vector3<f32> {
        use cgmath::InnerSpace;
        (self.target - self.eye).normalize()
    }

    /// Returns the right vector of the camera
    pub fn right(&self) -> cgmath::Vector3<f32> {
        use cgmath::InnerSpace;
        self.forward().cross(self.up).normalize()
    }

    /// Returns length of the camera's look vector
    /// (the distance from the camera eye to the target)
    pub fn length(&self) -> f32 {
        use cgmath::MetricSpace;
        self.eye.distance(self.target)
    }

    pub fn to_uniform(&self) -> CameraUniform {
        let mut ret = CameraUniform::new();
        ret.update_view_proj(&self);
        return ret;
    }

    /// Projects a 3D world-space point to normalized device coordinates (NDC).
    ///
    /// # Arguments
    /// * `world_point` - Point in world space
    ///
    /// # Returns
    /// A 3D point in NDC space where:
    /// - X and Y are in range [-1, 1] (left/right, bottom/top)
    /// - Z is in range [0, 1] (near/far in WGPU depth convention)
    pub fn project_point_ndc(&self, world_point: cgmath::Point3<f32>) -> cgmath::Point3<f32> {
        let vp = self.build_view_projection_matrix();
        let homogeneous = vp * world_point.to_homogeneous();

        // Perform perspective division
        cgmath::Point3::from_homogeneous(homogeneous)
    }

    /// Unprojects a point from normalized device coordinates (NDC) to world space.
    ///
    /// # Arguments
    /// * `ndc_point` - Point in NDC space where:
    ///   - X and Y are in range [-1, 1] (left/right, bottom/top)
    ///   - Z is in range [0, 1] (near/far in WGPU depth convention)
    ///
    /// # Returns
    /// A 3D point in world space, or None if the view-projection matrix is not invertible.
    pub fn unproject_point_ndc(&self, ndc_point: cgmath::Point3<f32>) -> Option<cgmath::Point3<f32>> {
        use cgmath::SquareMatrix;

        let viewproj = self.build_view_projection_matrix();

        let inv_vp = viewproj.invert()?;

        // Convert NDC point to homogeneous coordinates
        let homogeneous = inv_vp * ndc_point.to_homogeneous();

        // Perform perspective division
        Some(cgmath::Point3::from_homogeneous(homogeneous))
    }

    /// Projects a 3D world-space point to screen-space pixel coordinates.
    ///
    /// # Arguments
    /// * `world_point` - Point in world space
    /// * `screen_width` - Width of the screen/viewport in pixels
    /// * `screen_height` - Height of the screen/viewport in pixels
    ///
    /// # Returns
    /// A 3D point in screen space where:
    /// - X is in range [0, screen_width] (left to right)
    /// - Y is in range [0, screen_height] (top to bottom)
    /// - Z is the depth value in range [0, 1]
    pub fn project_point_screen(
        &self,
        world_point: cgmath::Point3<f32>,
        screen_width: u32,
        screen_height: u32,
    ) -> cgmath::Point3<f32> {
        let ndc = self.project_point_ndc(world_point);

        // Convert NDC to screen coordinates
        // NDC: [-1, 1] × [-1, 1], Y-up
        // Screen: [0, width] × [0, height], Y-down
        let screen_x = (ndc.x + 1.0) * 0.5 * screen_width as f32;
        let screen_y = (1.0 - ndc.y) * 0.5 * screen_height as f32; // Flip Y
        let screen_z = ndc.z; // Keep depth as-is

        cgmath::Point3::new(screen_x, screen_y, screen_z)
    }

    /// Unprojects a screen-space pixel coordinate to a point in world space.
    ///
    /// # Arguments
    /// * `screen_x` - X coordinate in screen space (0 = left edge)
    /// * `screen_y` - Y coordinate in screen space (0 = top edge)
    /// * `depth` - Depth value in range [0, 1] (0 = near plane, 1 = far plane)
    /// * `screen_width` - Width of the screen/viewport in pixels
    /// * `screen_height` - Height of the screen/viewport in pixels
    ///
    /// # Returns
    /// A 3D point in world space, or None if the view-projection matrix is not invertible.
    pub fn unproject_point_screen(
        &self,
        screen_x: f32,
        screen_y: f32,
        depth: f32,
        screen_width: u32,
        screen_height: u32,
    ) -> Option<cgmath::Point3<f32>> {
        // Convert screen coordinates to NDC
        // Screen: [0, width] × [0, height], Y-down
        // NDC: [-1, 1] × [-1, 1], Y-up
        let ndc_x = (screen_x / screen_width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / screen_height as f32) * 2.0; // Flip Y
        let ndc_z = depth;

        let ndc_point = cgmath::Point3::new(ndc_x, ndc_y, ndc_z);
        self.unproject_point_ndc(ndc_point)
    }
}


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Point3, Vector3, InnerSpace, Matrix4, SquareMatrix};

    const EPSILON: f32 = 1e-6;

    // Helper function to create a basic test camera
    fn create_test_camera() -> Camera {
        Camera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    // ===== Camera Struct Tests =====

    #[test]
    fn test_camera_forward() {
        let camera = create_test_camera();
        let forward = camera.forward();

        // Eye at (0,0,5), target at (0,0,0)
        // Forward should point from eye to target: (0,0,-5) normalized = (0,0,-1)
        assert!((forward.x - 0.0).abs() < EPSILON);
        assert!((forward.y - 0.0).abs() < EPSILON);
        assert!((forward.z - -1.0).abs() < EPSILON);

        // Should be unit length
        let magnitude = forward.magnitude();
        assert!((magnitude - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_camera_right() {
        let camera = create_test_camera();
        let forward = camera.forward();
        let right = camera.right();

        // Right should be perpendicular to forward
        let dot_forward = forward.dot(right);
        assert!(dot_forward.abs() < EPSILON);

        // Right should be perpendicular to up
        let dot_up = camera.up.dot(right);
        assert!(dot_up.abs() < EPSILON);

        // For our test camera setup (forward = -Z, up = +Y), right should be +X
        assert!((right.x - 1.0).abs() < EPSILON);
        assert!((right.y - 0.0).abs() < EPSILON);
        assert!((right.z - 0.0).abs() < EPSILON);

        // Should be unit length
        let magnitude = right.magnitude();
        assert!((magnitude - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_camera_length() {
        let camera = Camera {
            eye: Point3::new(3.0, 4.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        // Distance from (3,4,0) to (0,0,0) is sqrt(9 + 16) = 5.0
        let length = camera.length();
        assert!((length - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_camera_length_zero() {
        let camera = Camera {
            eye: Point3::new(1.0, 2.0, 3.0),
            target: Point3::new(1.0, 2.0, 3.0), // Same as eye
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        // Distance should be zero when eye == target
        let length = camera.length();
        assert!(length.abs() < EPSILON);
    }

    #[test]
    fn test_build_view_projection_identity() {
        let camera = Camera {
            eye: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let vp = camera.build_view_projection_matrix();

        // Should produce a valid matrix (not NaN or infinity)
        for i in 0..4 {
            for j in 0..4 {
                assert!(vp[i][j].is_finite());
            }
        }

        // For camera at origin looking at -Z, the view matrix should be close to identity
        // (though projection will transform it)
        // Main check: matrix is valid and determinant is non-zero
        let det = vp.determinant();
        assert!(det.abs() > EPSILON);
    }

    #[test]
    fn test_build_view_projection_aspect_ratio() {
        let mut camera1 = create_test_camera();
        camera1.aspect = 16.0 / 9.0;

        let mut camera2 = create_test_camera();
        camera2.aspect = 4.0 / 3.0;

        let vp1 = camera1.build_view_projection_matrix();
        let vp2 = camera2.build_view_projection_matrix();

        // Different aspect ratios should produce different matrices
        let mut found_difference = false;
        for i in 0..4 {
            for j in 0..4 {
                if (vp1[i][j] - vp2[i][j]).abs() > EPSILON {
                    found_difference = true;
                    break;
                }
            }
        }
        assert!(found_difference, "Aspect ratio should affect the view-projection matrix");
    }

    #[test]
    fn test_build_view_projection_fov() {
        let mut camera1 = create_test_camera();
        camera1.fovy = 45.0;

        let mut camera2 = create_test_camera();
        camera2.fovy = 90.0;

        let vp1 = camera1.build_view_projection_matrix();
        let vp2 = camera2.build_view_projection_matrix();

        // Different FOVs should produce different matrices
        let mut found_difference = false;
        for i in 0..4 {
            for j in 0..4 {
                if (vp1[i][j] - vp2[i][j]).abs() > EPSILON {
                    found_difference = true;
                    break;
                }
            }
        }
        assert!(found_difference, "FOV should affect the view-projection matrix");
    }

    // ===== CameraUniform Tests =====

    #[test]
    fn test_camera_uniform_new() {
        let uniform = CameraUniform::new();
        let identity = Matrix4::<f32>::identity();

        // Should initialize to identity matrix
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

        // Uniform should now contain the camera's view-projection matrix
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(uniform.view_proj[i][j], expected_vp[i][j]);
            }
        }
    }

    #[test]
    fn test_camera_uniform_layout() {
        use std::mem;

        // Verify size is correct for GPU (4x4 matrix of f32)
        assert_eq!(mem::size_of::<CameraUniform>(), 64);

        // Verify alignment is appropriate for GPU
        assert_eq!(mem::align_of::<CameraUniform>(), 4);
    }

    // ===== OpenGL-to-WGPU Matrix Tests =====

    #[test]
    fn test_depth_remapping() {
        use cgmath::Vector4;

        let m = OPENGL_TO_WGPU_MATRIX;

        // The matrix transforms clip-space coordinates (homogeneous, before perspective division)
        // Matrix correctly applies: z' = 0.5*z + 0.5*w, w' = w
        // This remaps OpenGL NDC [-1, 1] to WGPU NDC [0, 1]
        // After perspective division: z_ndc = z'/w' = (0.5*z + 0.5*w)/w = 0.5*(z/w) + 0.5

        // Test with W=1 (orthographic-like case)
        // Z=-1 in OpenGL clip space -> should map to Z=0 in WGPU NDC after division
        let near_clip = Vector4::new(0.0, 0.0, -1.0, 1.0);
        let near_result = m * near_clip;

        // In homogeneous coords: z' = 0.5*(-1) + 0.5*1 = 0.0, w' = 1.0
        assert!((near_result.z - 0.0).abs() < EPSILON);
        assert!((near_result.w - 1.0).abs() < EPSILON);

        // After perspective division: z_ndc = 0.0 / 1.0 = 0.0 (WGPU near plane)
        let z_ndc = near_result.z / near_result.w;
        assert!((z_ndc - 0.0).abs() < EPSILON);

        // Test with Z=1 (OpenGL far plane)
        let far_clip = Vector4::new(0.0, 0.0, 1.0, 1.0);
        let far_result = m * far_clip;
        // z' = 0.5*1 + 0.5*1 = 1.0, w' = 1.0
        assert!((far_result.z - 1.0).abs() < EPSILON);
        assert!((far_result.w - 1.0).abs() < EPSILON);

        // After perspective division: z_ndc = 1.0 / 1.0 = 1.0 (WGPU far plane)
        let z_ndc_far = far_result.z / far_result.w;
        assert!((z_ndc_far - 1.0).abs() < EPSILON);

        // Test with Z=0 (OpenGL mid plane)
        let mid_clip = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let mid_result = m * mid_clip;
        // z' = 0.5*0 + 0.5*1 = 0.5, w' = 1.0
        assert!((mid_result.z - 0.5).abs() < EPSILON);
        assert!((mid_result.w - 1.0).abs() < EPSILON);

        // After perspective division: z_ndc = 0.5 / 1.0 = 0.5 (WGPU mid plane)
        let z_ndc_mid = mid_result.z / mid_result.w;
        assert!((z_ndc_mid - 0.5).abs() < EPSILON);

        // Verify X and Y are unchanged
        let test_point = Vector4::new(3.5, -2.7, 0.0, 1.0);
        let transformed = m * test_point;
        assert!((transformed.x - 3.5).abs() < EPSILON);
        assert!((transformed.y - -2.7).abs() < EPSILON);
    }
}