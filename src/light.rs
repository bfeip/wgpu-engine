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

    pub fn to_uniform(&self) -> LightUniform {
        LightUniform {
            position: self.position.into(),
            _padding: 0,
            color: bytemuck::cast(self.color)
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    position: [f32; 3],
    _padding: u32, // 16 byte spacing required
    color: [f32; 4],
}