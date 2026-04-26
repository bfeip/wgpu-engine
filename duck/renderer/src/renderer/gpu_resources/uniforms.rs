use crate::scene::{Camera, Light, LightType, Material, MAX_LIGHTS};

/// GPU uniform buffer layout for camera data.
///
/// Contains the view-projection matrix and eye position, uploaded to
/// bind group 0 for use by shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix (64 bytes, 4x4 f32).
    view_proj: [[f32; 4]; 4],
    /// Camera eye position in world space (for view direction calculation in PBR).
    eye_position: [f32; 3],
    /// Padding for 16-byte alignment.
    _padding: u32,
}

impl CameraUniform {
    /// Creates a `CameraUniform` from a scene `Camera`.
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.build_view_projection_matrix().into(),
            eye_position: [camera.eye.x, camera.eye.y, camera.eye.z],
            _padding: 0,
        }
    }

    /// Creates a new camera uniform initialized to identity matrix and origin.
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
            eye_position: [0.0, 0.0, 0.0],
            _padding: 0,
        }
    }

}

/// GPU-ready instance data for instanced rendering.
///
/// Contains a world transform matrix and normal matrix packed for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstance {
    pub transform: [[f32; 4]; 4],
    pub normal_mat: [[f32; 3]; 3],
}

/// GPU-compatible representation of a single light for shader uniforms.
///
/// This struct is laid out to match WGSL uniform buffer alignment requirements.
/// vec3<f32> types require 16-byte alignment in WGSL, so scalar fields are
/// grouped at the start to pack efficiently.
///
/// # Memory Layout (64 bytes total)
///
/// | Offset | Size | Field          | Notes                              |
/// |--------|------|----------------|------------------------------------|
/// | 0      | 4    | light_type     | 0=Point, 1=Directional, 2=Spot     |
/// | 4      | 4    | range          | 0 = infinite range                 |
/// | 8      | 4    | inner_cone_cos | Spot: cos(inner angle)             |
/// | 12     | 4    | outer_cone_cos | Spot: cos(outer angle)             |
/// | 16     | 12   | position       | Point/Spot: world position         |
/// | 28     | 4    | intensity      | Light intensity multiplier         |
/// | 32     | 12   | direction      | Directional/Spot: light direction  |
/// | 44     | 4    | _padding1      | Alignment padding                  |
/// | 48     | 12   | color          | RGB color                          |
/// | 60     | 4    | _padding2      | Alignment padding                  |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    light_type: u32,
    range: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    position: [f32; 3],
    intensity: f32,
    direction: [f32; 3],
    _padding1: f32,
    color: [f32; 3],
    _padding2: f32,
}

impl LightUniform {
    /// Creates a `LightUniform` from a scene `Light`.
    pub fn from_light(light: &Light) -> Self {
        match light {
            Light::Point {
                position,
                color,
                intensity,
                range,
                ..
            } => LightUniform {
                light_type: LightType::Point as u32,
                range: *range,
                inner_cone_cos: 0.0,
                outer_cone_cos: 0.0,
                position: (*position).into(),
                intensity: *intensity,
                direction: [0.0, 0.0, 0.0],
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
            Light::Directional {
                direction,
                color,
                intensity,
                ..
            } => LightUniform {
                light_type: LightType::Directional as u32,
                range: 0.0,
                inner_cone_cos: 0.0,
                outer_cone_cos: 0.0,
                position: [0.0, 0.0, 0.0],
                intensity: *intensity,
                direction: (*direction).into(),
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
            Light::Spot {
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone_angle,
                outer_cone_angle,
                ..
            } => LightUniform {
                light_type: LightType::Spot as u32,
                range: *range,
                inner_cone_cos: inner_cone_angle.cos(),
                outer_cone_cos: outer_cone_angle.cos(),
                position: (*position).into(),
                intensity: *intensity,
                direction: (*direction).into(),
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
        }
    }
}

/// GPU-compatible array of lights with count.
///
/// # Memory Layout (528 bytes total)
///
/// | Offset | Size     | Field       |
/// |--------|----------|-------------|
/// | 0      | 4        | light_count |
/// | 4      | 12       | _padding    |
/// | 16     | 64 * 8   | lights      |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightsArrayUniform {
    pub light_count: u32,
    _padding: [u32; 3],
    pub lights: [LightUniform; MAX_LIGHTS],
}

impl LightsArrayUniform {
    /// Creates a lights array uniform from a slice of lights.
    ///
    /// Only the first `MAX_LIGHTS` lights will be used.
    pub fn from_lights(lights: &[Light]) -> Self {
        let mut uniform = Self {
            light_count: lights.len().min(MAX_LIGHTS) as u32,
            _padding: [0; 3],
            lights: [bytemuck::Zeroable::zeroed(); MAX_LIGHTS],
        };
        for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
            uniform.lights[i] = LightUniform::from_light(light);
        }
        uniform
    }
}

/// PBR material parameters for GPU uniform buffer.
///
/// This struct is sent to the shader and contains all scalar factors
/// plus flags indicating which textures are present.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrUniform {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub texture_flags: u32,
    pub alpha_cutoff: f32,
    pub _padding: [u32; 3],
}

impl PbrUniform {
    pub const FLAG_HAS_BASE_COLOR_TEXTURE: u32 = 1 << 0;
    pub const FLAG_HAS_NORMAL_TEXTURE: u32 = 1 << 1;
    pub const FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE: u32 = 1 << 2;

    /// Creates a `PbrUniform` from a scene `Material`.
    pub fn from_material(material: &Material) -> Self {
        let mut texture_flags = 0u32;
        if material.base_color_texture().is_some() {
            texture_flags |= Self::FLAG_HAS_BASE_COLOR_TEXTURE;
        }
        if material.normal_texture().is_some() {
            texture_flags |= Self::FLAG_HAS_NORMAL_TEXTURE;
        }
        if material.metallic_roughness_texture().is_some() {
            texture_flags |= Self::FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE;
        }

        let base_color = material.base_color_factor();
        PbrUniform {
            base_color_factor: [base_color.r, base_color.g, base_color.b, base_color.a],
            metallic_factor: material.metallic_factor(),
            roughness_factor: material.roughness_factor(),
            normal_scale: material.normal_scale(),
            texture_flags,
            alpha_cutoff: material.alpha_cutoff(),
            _padding: [0; 3],
        }
    }
}

/// GPU uniform for screen-space selection outline rendering.
/// Must match the layout in `outline_screenspace.wesl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::renderer) struct OutlineUniform {
    pub(in crate::renderer) color: [f32; 4],
    pub(in crate::renderer) width_pixels: f32,
    pub(in crate::renderer) screen_width: f32,
    pub(in crate::renderer) screen_height: f32,
    pub(in crate::renderer) _padding: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::common::{RgbaColor, EPSILON};
    use cgmath::Vector3;

    #[test]
    fn test_light_uniform_from_point_light() {
        let position = Vector3::new(1.0, 2.0, 3.0);
        let color = RgbaColor { r: 0.5, g: 0.6, b: 0.7, a: 1.0 };
        let light = Light::point(position, color, 2.0);

        let uniform = LightUniform::from_light(&light);

        assert!((uniform.position[0] - 1.0).abs() < EPSILON);
        assert!((uniform.position[1] - 2.0).abs() < EPSILON);
        assert!((uniform.position[2] - 3.0).abs() < EPSILON);
        assert_eq!(uniform.light_type, LightType::Point as u32);
        assert!((uniform.color[0] - 0.5).abs() < EPSILON);
        assert!((uniform.color[1] - 0.6).abs() < EPSILON);
        assert!((uniform.color[2] - 0.7).abs() < EPSILON);
        assert!((uniform.intensity - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_directional_light() {
        let direction = Vector3::new(0.0, -1.0, 0.0);
        let color = RgbaColor { r: 1.0, g: 1.0, b: 0.9, a: 1.0 };
        let light = Light::directional(direction, color, 1.5);

        let uniform = LightUniform::from_light(&light);

        assert_eq!(uniform.light_type, LightType::Directional as u32);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.intensity - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_spotlight() {
        let position = Vector3::new(0.0, 5.0, 0.0);
        let direction = Vector3::new(0.0, -1.0, 0.0);
        let color = RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
        let inner_angle = std::f32::consts::PI / 6.0;
        let outer_angle = std::f32::consts::PI / 4.0;
        let light = Light::spot(position, direction, color, 3.0, inner_angle, outer_angle);

        let uniform = LightUniform::from_light(&light);

        assert_eq!(uniform.light_type, LightType::Spot as u32);
        assert!((uniform.position[1] - 5.0).abs() < EPSILON);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.inner_cone_cos - inner_angle.cos()).abs() < EPSILON);
        assert!((uniform.outer_cone_cos - outer_angle.cos()).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_layout() {
        assert_eq!(std::mem::size_of::<LightUniform>(), 64);
        assert_eq!(std::mem::size_of::<LightsArrayUniform>(), 528);
    }

    #[test]
    fn test_lights_array_uniform_from_lights() {
        let lights = vec![
            Light::point(
                Vector3::new(1.0, 0.0, 0.0),
                RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 },
                1.0,
            ),
            Light::directional(
                Vector3::new(0.0, -1.0, 0.0),
                RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                1.0,
            ),
        ];

        let uniform = LightsArrayUniform::from_lights(&lights);

        assert_eq!(uniform.light_count, 2);
        assert_eq!(uniform.lights[0].light_type, LightType::Point as u32);
        assert_eq!(uniform.lights[1].light_type, LightType::Directional as u32);
    }
}
