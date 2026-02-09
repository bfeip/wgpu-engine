use crate::common::RgbaColor;

/// Maximum number of lights supported in the scene.
pub const MAX_LIGHTS: usize = 8;

/// Light type identifiers for GPU shader discrimination.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LightType {
    /// Point light (radiates in all directions from a position).
    Point = 0,
    /// Directional light (parallel rays, like sunlight).
    Directional = 1,
    /// Spotlight (cone of light from a position in a direction).
    Spot = 2,
}

/// A light source in the scene.
#[derive(Debug, Clone)]
pub enum Light {
    /// Point light that radiates in all directions from a position.
    Point {
        position: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        /// Maximum range of the light. 0.0 means infinite range.
        range: f32,
    },
    /// Directional light with parallel rays (like sunlight).
    Directional {
        /// Direction the light is pointing (will be normalized).
        direction: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
    },
    /// Spotlight with a cone of light.
    Spot {
        position: cgmath::Vector3<f32>,
        /// Direction the spotlight is pointing (will be normalized).
        direction: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        /// Maximum range of the light. 0.0 means infinite range.
        range: f32,
        /// Inner cone angle in radians (full intensity).
        inner_cone_angle: f32,
        /// Outer cone angle in radians (zero intensity).
        outer_cone_angle: f32,
    },
}

impl Light {
    /// Creates a new point light with explicit intensity.
    pub fn point(position: cgmath::Vector3<f32>, color: RgbaColor, intensity: f32) -> Self {
        Self::Point {
            position,
            color,
            intensity,
            range: 0.0,
        }
    }

    /// Creates a new point light with explicit intensity and range.
    pub fn point_with_range(
        position: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        range: f32,
    ) -> Self {
        Self::Point {
            position,
            color,
            intensity,
            range,
        }
    }

    /// Creates a new directional light.
    pub fn directional(
        direction: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
    ) -> Self {
        use cgmath::InnerSpace;
        Self::Directional {
            direction: direction.normalize(),
            color,
            intensity,
        }
    }

    /// Creates a new spotlight.
    ///
    /// # Arguments
    /// * `position` - World-space position of the spotlight
    /// * `direction` - Direction the spotlight is pointing
    /// * `color` - Light color (RGB)
    /// * `intensity` - Light intensity multiplier
    /// * `inner_cone_angle` - Inner cone angle in radians (full intensity region)
    /// * `outer_cone_angle` - Outer cone angle in radians (falloff to zero)
    pub fn spot(
        position: cgmath::Vector3<f32>,
        direction: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Self {
        use cgmath::InnerSpace;
        Self::Spot {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range: 0.0,
            inner_cone_angle,
            outer_cone_angle,
        }
    }

    /// Creates a new spotlight with explicit range.
    pub fn spot_with_range(
        position: cgmath::Vector3<f32>,
        direction: cgmath::Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        range: f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Self {
        use cgmath::InnerSpace;
        Self::Spot {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range,
            inner_cone_angle,
            outer_cone_angle,
        }
    }

    /// Converts the light to a GPU-compatible uniform structure.
    pub fn to_uniform(&self) -> LightUniform {
        match self {
            Light::Point {
                position,
                color,
                intensity,
                range,
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
    /// Creates an empty lights array uniform.
    pub fn new() -> Self {
        Self {
            light_count: 0,
            _padding: [0; 3],
            lights: [LightUniform::zeroed(); MAX_LIGHTS],
        }
    }

    /// Creates a lights array uniform from a slice of lights.
    ///
    /// Only the first `MAX_LIGHTS` lights will be used.
    pub fn from_lights(lights: &[Light]) -> Self {
        let mut uniform = Self::new();
        uniform.light_count = lights.len().min(MAX_LIGHTS) as u32;
        for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
            uniform.lights[i] = light.to_uniform();
        }
        uniform
    }
}

impl LightUniform {
    /// Creates a zeroed light uniform.
    fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::EPSILON;
    use cgmath::Vector3;

    #[test]
    fn test_light_uniform_from_point_light() {
        let position = Vector3::new(1.0, 2.0, 3.0);
        let color = RgbaColor {
            r: 0.5,
            g: 0.6,
            b: 0.7,
            a: 1.0,
        };
        let light = Light::point(position, color, 2.0);

        let uniform = light.to_uniform();

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
        let color = RgbaColor {
            r: 1.0,
            g: 1.0,
            b: 0.9,
            a: 1.0,
        };
        let light = Light::directional(direction, color, 1.5);

        let uniform = light.to_uniform();

        assert_eq!(uniform.light_type, LightType::Directional as u32);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.intensity - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_spotlight() {
        let position = Vector3::new(0.0, 5.0, 0.0);
        let direction = Vector3::new(0.0, -1.0, 0.0);
        let color = RgbaColor {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let inner_angle = std::f32::consts::PI / 6.0; // 30 degrees
        let outer_angle = std::f32::consts::PI / 4.0; // 45 degrees
        let light = Light::spot(position, direction, color, 3.0, inner_angle, outer_angle);

        let uniform = light.to_uniform();

        assert_eq!(uniform.light_type, LightType::Spot as u32);
        assert!((uniform.position[1] - 5.0).abs() < EPSILON);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.inner_cone_cos - inner_angle.cos()).abs() < EPSILON);
        assert!((uniform.outer_cone_cos - outer_angle.cos()).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_layout() {
        use std::mem;

        // LightUniform should be 64 bytes
        assert_eq!(mem::size_of::<LightUniform>(), 64);

        // LightsArrayUniform should be 16 + 64*8 = 528 bytes
        assert_eq!(mem::size_of::<LightsArrayUniform>(), 528);
    }

    #[test]
    fn test_lights_array_uniform_from_lights() {
        let lights = vec![
            Light::point(
                Vector3::new(1.0, 0.0, 0.0),
                RgbaColor {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                1.0,
            ),
            Light::directional(
                Vector3::new(0.0, -1.0, 0.0),
                RgbaColor {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                },
                1.0,
            ),
        ];

        let uniform = LightsArrayUniform::from_lights(&lights);

        assert_eq!(uniform.light_count, 2);
        assert_eq!(uniform.lights[0].light_type, LightType::Point as u32);
        assert_eq!(uniform.lights[1].light_type, LightType::Directional as u32);
    }
}
