use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Point3, SquareMatrix, Transform, Vector3};

use crate::Camera;
use crate::CoordinateSpace;
use crate::common::RgbaColor;

/// Maximum number of lights supported in the scene.
pub const MAX_LIGHTS: usize = 8;

/// Light type identifiers.
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
        /// World or camera-space position.
        position: Vector3<f32>,
        /// Light color.
        color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
        /// Maximum range of the light. 0.0 means infinite range.
        range: f32,
        /// Coordinate space for this light's position.
        space: CoordinateSpace,
    },
    /// Directional light with parallel rays (like sunlight).
    Directional {
        /// Direction the light is pointing (will be normalized).
        direction: Vector3<f32>,
        /// Light color.
        color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
        /// Coordinate space for this light's direction.
        space: CoordinateSpace,
    },
    /// Spotlight with a cone of light.
    Spot {
        /// World or camera-space position.
        position: Vector3<f32>,
        /// Direction the spotlight is pointing (will be normalized).
        direction: Vector3<f32>,
        /// Light color.
        color: RgbaColor,
        /// Intensity multiplier.
        intensity: f32,
        /// Maximum range of the light. 0.0 means infinite range.
        range: f32,
        /// Inner cone angle in radians (full intensity).
        inner_cone_angle: f32,
        /// Outer cone angle in radians (zero intensity).
        outer_cone_angle: f32,
        /// Coordinate space for this light's position and direction.
        space: CoordinateSpace,
    },
}

impl Light {
    /// Creates a new point light with explicit intensity.
    pub fn point(position: Vector3<f32>, color: RgbaColor, intensity: f32) -> Self {
        Self::Point {
            position,
            color,
            intensity,
            range: 0.0,
            space: CoordinateSpace::World,
        }
    }

    /// Creates a new point light with explicit intensity and range.
    pub fn point_with_range(
        position: Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        range: f32,
    ) -> Self {
        Self::Point {
            position,
            color,
            intensity,
            range,
            space: CoordinateSpace::World,
        }
    }

    /// Creates a new directional light.
    pub fn directional(direction: Vector3<f32>, color: RgbaColor, intensity: f32) -> Self {
        Self::Directional {
            direction: direction.normalize(),
            color,
            intensity,
            space: CoordinateSpace::World,
        }
    }

    /// Creates a new spotlight.
    pub fn spot(
        position: Vector3<f32>,
        direction: Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Self {
        Self::Spot {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range: 0.0,
            inner_cone_angle,
            outer_cone_angle,
            space: CoordinateSpace::World,
        }
    }

    /// Creates a new spotlight with explicit range.
    pub fn spot_with_range(
        position: Vector3<f32>,
        direction: Vector3<f32>,
        color: RgbaColor,
        intensity: f32,
        range: f32,
        inner_cone_angle: f32,
        outer_cone_angle: f32,
    ) -> Self {
        Self::Spot {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            range,
            inner_cone_angle,
            outer_cone_angle,
            space: CoordinateSpace::World,
        }
    }

    /// Sets the coordinate space for this light. Use as a builder:
    /// ```ignore
    /// Light::point(pos, color, 1.0).in_space(CoordinateSpace::Camera)
    /// ```
    pub fn in_space(mut self, space: CoordinateSpace) -> Self {
        match &mut self {
            Light::Point { space: s, .. } => *s = space,
            Light::Directional { space: s, .. } => *s = space,
            Light::Spot { space: s, .. } => *s = space,
        }
        self
    }

    /// Returns the coordinate space of this light.
    pub fn space(&self) -> CoordinateSpace {
        match self {
            Light::Point { space, .. } => *space,
            Light::Directional { space, .. } => *space,
            Light::Spot { space, .. } => *space,
        }
    }

    /// Resolves this light to world space given a camera.
    ///
    /// If the light is already in world space, returns a clone unchanged.
    /// For camera-space lights, transforms positions and directions using
    /// the camera's inverse view matrix.
    pub fn in_worldspace(&self, camera: &Camera) -> Light {
        match self.space() {
            CoordinateSpace::World => self.clone(),
            CoordinateSpace::Camera => {
                let view = Matrix4::look_at_rh(camera.eye, camera.target, camera.up);
                let inv_view = view.invert().unwrap_or(Matrix4::identity());
                self.transform(inv_view, CoordinateSpace::World)
            }
        }
    }

    /// Returns a new light with positions and directions transformed by the given matrix.
    ///
    /// Positions are transformed as points (with translation).
    /// Directions are transformed as vectors (rotation only, no translation).
    /// The resulting light is assigned the given coordinate space.
    pub fn transform(&self, matrix: Matrix4<f32>, space: CoordinateSpace) -> Light {
        match self {
            Light::Point {
                position,
                color,
                intensity,
                range,
                ..
            } => {
                let new_pos = matrix.transform_point(Point3::from_vec(*position));
                Light::Point {
                    position: new_pos.to_vec(),
                    color: *color,
                    intensity: *intensity,
                    range: *range,
                    space,
                }
            }
            Light::Directional {
                direction,
                color,
                intensity,
                ..
            } => {
                let new_dir = matrix.transform_vector(*direction).normalize();
                Light::Directional {
                    direction: new_dir,
                    color: *color,
                    intensity: *intensity,
                    space,
                }
            }
            Light::Spot {
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone_angle,
                outer_cone_angle,
                ..
            } => {
                let new_pos = matrix.transform_point(Point3::from_vec(*position));
                let new_dir = matrix.transform_vector(*direction).normalize();
                Light::Spot {
                    position: new_pos.to_vec(),
                    direction: new_dir,
                    color: *color,
                    intensity: *intensity,
                    range: *range,
                    inner_cone_angle: *inner_cone_angle,
                    outer_cone_angle: *outer_cone_angle,
                    space,
                }
            }
        }
    }
}
