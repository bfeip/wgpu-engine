/// Matrix to convert from OpenGL clip-space depth [-1, 1] to WGPU depth [0, 1].
///
/// WGPU uses a different depth convention than OpenGL:
/// - OpenGL NDC depth: [-1, 1] (near to far)
/// - WGPU NDC depth: [0, 1] (near to far)
///
/// This matrix remaps Z: `z' = 0.5 * z + 0.5`
#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// A camera that defines the viewpoint and projection for rendering.
///
/// The camera combines view parameters (position, orientation) with projection
/// parameters (field of view, clipping planes) to produce a view-projection matrix
/// used by the GPU to transform world-space coordinates to clip-space.
///
/// # Example
///
/// ```
/// use cgmath::{Point3, Vector3};
/// use wgpu_engine::Camera;
///
/// let camera = Camera {
///     eye: Point3::new(0.0, 0.0, 5.0),
///     target: Point3::new(0.0, 0.0, 0.0),
///     up: Vector3::new(0.0, 1.0, 0.0),
///     aspect: 16.0 / 9.0,
///     fovy: 45.0,
///     znear: 0.1,
///     zfar: 100.0,
///     ortho: false,
/// };
/// ```
pub struct Camera {
    /// The position of the camera in world space.
    pub eye: cgmath::Point3<f32>,
    /// The point the camera is looking at in world space.
    pub target: cgmath::Point3<f32>,
    /// The up direction vector (typically Y-up: `(0, 1, 0)`).
    pub up: cgmath::Vector3<f32>,
    /// The aspect ratio of the viewport (width / height).
    pub aspect: f32,
    /// Vertical field of view in degrees (used for perspective projection).
    pub fovy: f32,
    /// Distance to the near clipping plane.
    pub znear: f32,
    /// Distance to the far clipping plane.
    pub zfar: f32,
    /// When true, use orthographic projection instead of perspective.
    ///
    /// The orthographic view size is derived from the camera distance and fovy,
    /// so zoom (changing distance) works naturally for both projection modes.
    pub ortho: bool,
}

impl Camera {
    /// Builds the combined view-projection matrix for this camera.
    ///
    /// The resulting matrix transforms world-space coordinates to clip-space,
    /// ready for the GPU rasterizer. It combines:
    /// - View matrix: transforms world space to camera/view space
    /// - Projection matrix: transforms view space to clip space (perspective or orthographic)
    /// - Depth remapping: converts OpenGL depth convention to WGPU convention
    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = if self.ortho {
            // Derive orthographic bounds from camera distance and fovy.
            // This allows zoom (changing distance) to work naturally.
            let half_height = self.length() * (self.fovy.to_radians() / 2.0).tan();
            let half_width = half_height * self.aspect;
            cgmath::ortho(-half_width, half_width, -half_height, half_height, self.znear, self.zfar)
        } else {
            cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar)
        };

        OPENGL_TO_WGPU_MATRIX * proj * view
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

    /// Creates a GPU-compatible uniform buffer representation of this camera.
    pub(crate) fn to_uniform(&self) -> CameraUniform {
        let mut ret = CameraUniform::new();
        ret.update_view_proj(self);
        ret
    }

    /// Adjusts the camera to fit a bounding box in view.
    ///
    /// Positions the camera so the entire bounding box is visible while maintaining
    /// the current view direction (from eye towards target). The camera is moved
    /// along this direction to ensure the bounds fit within the field of view.
    ///
    /// # Arguments
    /// * `bounds` - The axis-aligned bounding box to fit in view
    pub fn fit_to_bounds(&mut self, bounds: &crate::common::Aabb) {
        use cgmath::{InnerSpace, MetricSpace};

        let center = bounds.center();
        let (size_x, size_y, size_z) = bounds.size();

        // Compute the bounding sphere radius (half the diagonal of the AABB)
        let bounding_radius = (size_x * size_x + size_y * size_y + size_z * size_z).sqrt() / 2.0;

        // Calculate the distance needed to fit the bounding sphere in view
        // Using the vertical field of view and accounting for aspect ratio
        let half_fov_rad = (self.fovy / 2.0).to_radians();

        // Calculate distance for vertical fit
        let vertical_distance = bounding_radius / half_fov_rad.sin();

        // Calculate distance for horizontal fit (accounting for aspect ratio)
        let half_hfov_rad = (half_fov_rad.tan() * self.aspect).atan();
        let horizontal_distance = bounding_radius / half_hfov_rad.sin();

        // Use the larger distance to ensure the object fits in both dimensions
        let distance = vertical_distance.max(horizontal_distance);

        // Get current view direction (or default to -Z if eye == target)
        let view_dir = if self.eye.distance(self.target) < 1e-6 {
            cgmath::Vector3::new(0.0, 0.0, -1.0)
        } else {
            (self.target - self.eye).normalize()
        };

        // Position camera at the calculated distance from the center
        self.target = center;
        self.eye = center - view_dir * distance;

        // Adjust near/far planes to encompass the scene
        // Near plane: at least 1/1000th of the distance, but not less than 0.001
        // Far plane: at least twice the distance plus the bounding radius
        self.znear = (distance * 0.001).max(0.001);
        self.zfar = (distance + bounding_radius) * 2.0;
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


/// GPU uniform buffer layout for camera data.
///
/// This struct is `#[repr(C)]` and implements `bytemuck::Pod` for direct
/// memory mapping to GPU buffers. It contains only the view-projection matrix,
/// which is uploaded to bind group 0 for use by shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    /// Combined view-projection matrix (64 bytes, 4x4 f32).
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    /// Creates a new camera uniform initialized to the identity matrix.
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    /// Updates the view-projection matrix from the given camera.
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
            ortho: false,
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
            ortho: false,
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
            ortho: false,
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
            ortho: false,
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

    // ===== Projection/Unprojection Tests =====

    #[test]
    fn test_project_unproject_ndc_roundtrip() {
        let camera = create_test_camera();

        // Test various world points
        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),    // Origin (camera target)
            Point3::new(1.0, 0.0, 0.0),    // Right
            Point3::new(0.0, 1.0, 0.0),    // Up
            Point3::new(0.0, 0.0, -1.0),   // Forward from target
            Point3::new(2.5, 1.5, -3.0),   // Arbitrary point
        ];

        for original_point in test_points {
            // Project to NDC
            let ndc = camera.project_point_ndc(original_point);

            // Unproject back to world space
            let unprojected = camera.unproject_point_ndc(ndc)
                .expect("Failed to unproject point");

            // Should get back the original point (within floating point precision)
            assert!(
                (unprojected.x - original_point.x).abs() < 1e-4,
                "X mismatch: original={}, unprojected={}", original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-4,
                "Y mismatch: original={}, unprojected={}", original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-4,
                "Z mismatch: original={}, unprojected={}", original_point.z, unprojected.z
            );
        }
    }

    #[test]
    fn test_project_camera_target_to_ndc_center() {
        let camera = create_test_camera();

        // The camera target should project to the center of NDC (0, 0)
        let ndc = camera.project_point_ndc(camera.target);

        // X and Y should be at center (0, 0)
        assert!(
            ndc.x.abs() < 1e-4,
            "Target should project to NDC center X, got {}", ndc.x
        );
        assert!(
            ndc.y.abs() < 1e-4,
            "Target should project to NDC center Y, got {}", ndc.y
        );

        // Z should be somewhere between 0 and 1 (in front of camera)
        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "NDC Z should be in [0, 1], got {}", ndc.z);
    }

    #[test]
    fn test_project_ndc_bounds() {
        let camera = create_test_camera();

        // Points in front of the camera should have NDC Z between 0 and 1
        let point_in_front = Point3::new(0.0, 0.0, 2.0);
        let ndc = camera.project_point_ndc(point_in_front);

        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "Point in frustum should have NDC Z in [0, 1]");
    }

    #[test]
    fn test_project_ndc_depth_ordering() {
        let camera = create_test_camera();

        // Points closer to camera should have smaller NDC Z values
        let point_near = Point3::new(0.0, 0.0, 1.0);   // Closer
        let point_far = Point3::new(0.0, 0.0, -2.0);   // Farther

        let ndc_near = camera.project_point_ndc(point_near);
        let ndc_far = camera.project_point_ndc(point_far);

        // Closer point should have smaller Z (closer to 0)
        assert!(
            ndc_near.z < ndc_far.z,
            "Closer point should have smaller NDC Z: near={}, far={}",
            ndc_near.z, ndc_far.z
        );
    }

    #[test]
    fn test_unproject_ndc_center() {
        let camera = create_test_camera();

        // NDC center (0, 0) at mid-depth should unproject to a point on camera's forward ray
        let ndc_center = Point3::new(0.0, 0.0, 0.5);
        let world_point = camera.unproject_point_ndc(ndc_center)
            .expect("Failed to unproject NDC center");

        // The unprojected point should lie on the line from camera eye towards target
        let to_point = (world_point - camera.eye).normalize();
        let forward = camera.forward();

        // Vectors should be parallel (dot product close to 1 or -1)
        let dot = to_point.dot(forward);
        assert!(
            (dot.abs() - 1.0).abs() < 1e-4,
            "Unprojected center should lie on camera forward ray, dot={}",
            dot
        );
    }

    #[test]
    fn test_unproject_ndc_corners() {
        let camera = create_test_camera();

        // Test all four corners of NDC at mid-depth
        let corners = vec![
            Point3::new(-1.0, -1.0, 0.5), // Bottom-left
            Point3::new(1.0, -1.0, 0.5),  // Bottom-right
            Point3::new(-1.0, 1.0, 0.5),  // Top-left
            Point3::new(1.0, 1.0, 0.5),   // Top-right
        ];

        for ndc_corner in corners {
            let world_point = camera.unproject_point_ndc(ndc_corner)
                .expect("Failed to unproject corner");

            // All unprojected points should be valid (finite)
            assert!(world_point.x.is_finite());
            assert!(world_point.y.is_finite());
            assert!(world_point.z.is_finite());

            // Round-trip should work
            let reprojected = camera.project_point_ndc(world_point);
            assert!((reprojected.x - ndc_corner.x).abs() < 1e-4);
            assert!((reprojected.y - ndc_corner.y).abs() < 1e-4);
            assert!((reprojected.z - ndc_corner.z).abs() < 1e-4);
        }
    }

    #[test]
    fn test_project_screen_coordinates() {
        let camera = create_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        // Camera target should project to screen center
        let screen_point = camera.project_point_screen(
            camera.target,
            screen_width,
            screen_height
        );

        // Should be at center of screen
        let center_x = screen_width as f32 / 2.0;
        let center_y = screen_height as f32 / 2.0;

        assert!(
            (screen_point.x - center_x).abs() < 1.0,
            "Target should project to screen center X: expected {}, got {}",
            center_x, screen_point.x
        );
        assert!(
            (screen_point.y - center_y).abs() < 1.0,
            "Target should project to screen center Y: expected {}, got {}",
            center_y, screen_point.y
        );

        // Depth should be in valid range
        assert!(screen_point.z >= 0.0 && screen_point.z <= 1.0);
    }

    #[test]
    fn test_project_unproject_screen_roundtrip() {
        let camera = create_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-0.5, 2.0, -1.0),
        ];

        for original_point in test_points {
            // Project to screen
            let screen = camera.project_point_screen(
                original_point,
                screen_width,
                screen_height
            );

            // Unproject back to world
            let unprojected = camera.unproject_point_screen(
                screen.x,
                screen.y,
                screen.z,
                screen_width,
                screen_height
            ).expect("Failed to unproject screen point");

            // Should match original
            assert!(
                (unprojected.x - original_point.x).abs() < 1e-3,
                "Screen roundtrip X mismatch: original={}, unprojected={}",
                original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-3,
                "Screen roundtrip Y mismatch: original={}, unprojected={}",
                original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-3,
                "Screen roundtrip Z mismatch: original={}, unprojected={}",
                original_point.z, unprojected.z
            );
        }
    }

    #[test]
    fn test_screen_coordinate_bounds() {
        let camera = create_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        // Test screen corners map to NDC corners
        let test_cases = vec![
            // (screen_x, screen_y, expected_ndc_x, expected_ndc_y)
            (0.0, 0.0, -1.0, 1.0),                                    // Top-left
            (screen_width as f32, 0.0, 1.0, 1.0),                     // Top-right
            (0.0, screen_height as f32, -1.0, -1.0),                  // Bottom-left
            (screen_width as f32, screen_height as f32, 1.0, -1.0),   // Bottom-right
            (screen_width as f32 / 2.0, screen_height as f32 / 2.0, 0.0, 0.0), // Center
        ];

        for (screen_x, screen_y, expected_ndc_x, expected_ndc_y) in test_cases {
            let world_point = camera.unproject_point_screen(
                screen_x,
                screen_y,
                0.5, // Mid-depth
                screen_width,
                screen_height
            ).expect("Failed to unproject screen corner");

            // Project back to NDC to verify
            let ndc = camera.project_point_ndc(world_point);

            assert!(
                (ndc.x - expected_ndc_x).abs() < 1e-4,
                "Screen ({}, {}) should map to NDC X {}, got {}",
                screen_x, screen_y, expected_ndc_x, ndc.x
            );
            assert!(
                (ndc.y - expected_ndc_y).abs() < 1e-4,
                "Screen ({}, {}) should map to NDC Y {}, got {}",
                screen_x, screen_y, expected_ndc_y, ndc.y
            );
        }
    }

    #[test]
    fn test_screen_y_flip() {
        let camera = create_test_camera();
        let screen_width = 800;
        let screen_height = 600;

        // Top of screen (Y=0) should correspond to positive NDC Y
        let top_screen = camera.unproject_point_screen(
            400.0, 0.0, 0.5,
            screen_width, screen_height
        ).expect("Failed to unproject top");

        // Bottom of screen (Y=height) should correspond to negative NDC Y
        let bottom_screen = camera.unproject_point_screen(
            400.0, screen_height as f32, 0.5,
            screen_width, screen_height
        ).expect("Failed to unproject bottom");

        let top_ndc = camera.project_point_ndc(top_screen);
        let bottom_ndc = camera.project_point_ndc(bottom_screen);

        // Top screen Y should have positive NDC Y
        assert!(top_ndc.y > 0.5, "Top of screen should have positive NDC Y");

        // Bottom screen Y should have negative NDC Y
        assert!(bottom_ndc.y < -0.5, "Bottom of screen should have negative NDC Y");
    }

    #[test]
    fn test_depth_range_screen() {
        let camera = create_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        // Test different depth values
        let center_x = screen_width as f32 / 2.0;
        let center_y = screen_height as f32 / 2.0;

        let near_point = camera.unproject_point_screen(
            center_x, center_y, 0.0,
            screen_width, screen_height
        ).expect("Failed to unproject near");

        let mid_point = camera.unproject_point_screen(
            center_x, center_y, 0.5,
            screen_width, screen_height
        ).expect("Failed to unproject mid");

        let far_point = camera.unproject_point_screen(
            center_x, center_y, 1.0,
            screen_width, screen_height
        ).expect("Failed to unproject far");

        // Near point should be closer to camera than far point
        let dist_near = (near_point - camera.eye).magnitude();
        let dist_mid = (mid_point - camera.eye).magnitude();
        let dist_far = (far_point - camera.eye).magnitude();

        assert!(dist_near < dist_mid, "Near point should be closer than mid");
        assert!(dist_mid < dist_far, "Mid point should be closer than far");
    }

    #[test]
    fn test_projection_consistency_across_methods() {
        let camera = create_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        let world_point = Point3::new(1.5, -0.5, 1.0);

        // Project using both methods
        let ndc = camera.project_point_ndc(world_point);
        let screen = camera.project_point_screen(world_point, screen_width, screen_height);

        // Manual conversion from screen to NDC
        let ndc_from_screen_x = (screen.x / screen_width as f32) * 2.0 - 1.0;
        let ndc_from_screen_y = 1.0 - (screen.y / screen_height as f32) * 2.0;

        // Should match
        assert!(
            (ndc.x - ndc_from_screen_x).abs() < 1e-4,
            "NDC X should match: direct={}, from_screen={}",
            ndc.x, ndc_from_screen_x
        );
        assert!(
            (ndc.y - ndc_from_screen_y).abs() < 1e-4,
            "NDC Y should match: direct={}, from_screen={}",
            ndc.y, ndc_from_screen_y
        );
        assert!(
            (ndc.z - screen.z).abs() < EPSILON,
            "Depth should match"
        );
    }

    // ===== Orthographic Projection Tests =====

    fn create_ortho_test_camera() -> Camera {
        Camera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: true,
        }
    }

    #[test]
    fn test_ortho_build_view_projection_valid() {
        let camera = create_ortho_test_camera();
        let vp = camera.build_view_projection_matrix();

        // Should produce a valid matrix (not NaN or infinity)
        for i in 0..4 {
            for j in 0..4 {
                assert!(vp[i][j].is_finite(), "Matrix element [{i}][{j}] is not finite");
            }
        }

        // Determinant should be non-zero (invertible)
        let det = vp.determinant();
        assert!(det.abs() > EPSILON, "Ortho matrix should be invertible");
    }

    #[test]
    fn test_ortho_project_unproject_ndc_roundtrip() {
        let camera = create_ortho_test_camera();

        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),    // Origin (camera target)
            Point3::new(1.0, 0.0, 0.0),    // Right
            Point3::new(0.0, 1.0, 0.0),    // Up
            Point3::new(0.0, 0.0, -1.0),   // Forward from target
            Point3::new(2.5, 1.5, -3.0),   // Arbitrary point
        ];

        for original_point in test_points {
            let ndc = camera.project_point_ndc(original_point);
            let unprojected = camera.unproject_point_ndc(ndc)
                .expect("Failed to unproject ortho point");

            assert!(
                (unprojected.x - original_point.x).abs() < 1e-4,
                "Ortho X mismatch: original={}, unprojected={}", original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-4,
                "Ortho Y mismatch: original={}, unprojected={}", original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-4,
                "Ortho Z mismatch: original={}, unprojected={}", original_point.z, unprojected.z
            );
        }
    }

    #[test]
    fn test_ortho_target_projects_to_center() {
        let camera = create_ortho_test_camera();

        let ndc = camera.project_point_ndc(camera.target);

        // Target should project to center (0, 0)
        assert!(ndc.x.abs() < 1e-4, "Ortho target should project to NDC center X, got {}", ndc.x);
        assert!(ndc.y.abs() < 1e-4, "Ortho target should project to NDC center Y, got {}", ndc.y);
        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "NDC Z should be in [0, 1], got {}", ndc.z);
    }

    #[test]
    fn test_ortho_depth_ordering() {
        let camera = create_ortho_test_camera();

        let point_near = Point3::new(0.0, 0.0, 1.0);   // Closer
        let point_far = Point3::new(0.0, 0.0, -2.0);   // Farther

        let ndc_near = camera.project_point_ndc(point_near);
        let ndc_far = camera.project_point_ndc(point_far);

        // Closer point should have smaller Z
        assert!(
            ndc_near.z < ndc_far.z,
            "Ortho closer point should have smaller NDC Z: near={}, far={}",
            ndc_near.z, ndc_far.z
        );
    }

    #[test]
    fn test_ortho_no_perspective_distortion() {
        let camera = create_ortho_test_camera();

        // In orthographic projection, objects at different Z distances
        // should project to the same X/Y in NDC (no foreshortening)
        let point_near = Point3::new(1.0, 1.0, 2.0);
        let point_far = Point3::new(1.0, 1.0, -5.0);

        let ndc_near = camera.project_point_ndc(point_near);
        let ndc_far = camera.project_point_ndc(point_far);

        // X and Y should be identical regardless of Z distance
        assert!(
            (ndc_near.x - ndc_far.x).abs() < 1e-5,
            "Ortho X should be same at different depths: near={}, far={}",
            ndc_near.x, ndc_far.x
        );
        assert!(
            (ndc_near.y - ndc_far.y).abs() < 1e-5,
            "Ortho Y should be same at different depths: near={}, far={}",
            ndc_near.y, ndc_far.y
        );
    }

    #[test]
    fn test_ortho_vs_perspective_different_matrices() {
        let mut camera = create_test_camera();

        let persp_vp = camera.build_view_projection_matrix();

        camera.ortho = true;
        let ortho_vp = camera.build_view_projection_matrix();

        // Matrices should be different
        let mut found_difference = false;
        for i in 0..4 {
            for j in 0..4 {
                if (persp_vp[i][j] - ortho_vp[i][j]).abs() > EPSILON {
                    found_difference = true;
                    break;
                }
            }
        }
        assert!(found_difference, "Perspective and orthographic matrices should differ");
    }

    #[test]
    fn test_ortho_screen_roundtrip() {
        let camera = create_ortho_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-0.5, 2.0, -1.0),
        ];

        for original_point in test_points {
            let screen = camera.project_point_screen(original_point, screen_width, screen_height);
            let unprojected = camera.unproject_point_screen(
                screen.x, screen.y, screen.z,
                screen_width, screen_height
            ).expect("Failed to unproject ortho screen point");

            assert!(
                (unprojected.x - original_point.x).abs() < 1e-3,
                "Ortho screen roundtrip X mismatch: original={}, unprojected={}",
                original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-3,
                "Ortho screen roundtrip Y mismatch: original={}, unprojected={}",
                original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-3,
                "Ortho screen roundtrip Z mismatch: original={}, unprojected={}",
                original_point.z, unprojected.z
            );
        }
    }
}