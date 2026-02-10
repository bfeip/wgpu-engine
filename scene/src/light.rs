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

}

