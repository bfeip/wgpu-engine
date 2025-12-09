use crate::common::RgbaColor;

pub struct Light {
    position: cgmath::Vector3<f32>,
    color: RgbaColor,
}

impl Light {
    pub fn new(position: cgmath::Vector3<f32>, color: RgbaColor) -> Self {
        Self {
            position,
            color
        }
    }

    pub(crate) fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: self.position.into(),
            _padding: 0,
            color: bytemuck::cast(self.color)
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightUniform {
    position: [f32; 3],
    _padding: u32, // 16 byte spacing required
    color: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Vector3;
    use crate::common::EPSILON;

    // ===== LightUniform Tests =====

    #[test]
    fn test_light_uniform_from_light() {
        let position = Vector3::new(1.0, 2.0, 3.0);
        let color = RgbaColor { r: 0.5, g: 0.6, b: 0.7, a: 1.0 };
        let light = Light::new(position, color);

        let uniform = light.to_uniform();

        // Position should match
        assert!((uniform.position[0] - 1.0).abs() < EPSILON);
        assert!((uniform.position[1] - 2.0).abs() < EPSILON);
        assert!((uniform.position[2] - 3.0).abs() < EPSILON);

        // Padding should be zero
        assert_eq!(uniform._padding, 0);

        // Color should match
        assert!((uniform.color[0] - 0.5).abs() < EPSILON);
        assert!((uniform.color[1] - 0.6).abs() < EPSILON);
        assert!((uniform.color[2] - 0.7).abs() < EPSILON);
        assert!((uniform.color[3] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_layout() {
        use std::mem;

        // Should be 32 bytes total:
        // - position: [f32; 3] = 12 bytes
        // - _padding: u32 = 4 bytes
        // - color: [f32; 4] = 16 bytes
        // Total = 32 bytes
        assert_eq!(mem::size_of::<LightUniform>(), 32);

        // Alignment should be at least 4 (for f32 and u32)
        let alignment = mem::align_of::<LightUniform>();
        assert!(alignment >= 4);
    }
}