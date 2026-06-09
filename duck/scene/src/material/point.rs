use crate::common::RgbaColor;

/// Unique identifier for a [`PointMaterial`].
pub type PointMaterialId = crate::Id<PointMaterial>;

/// Shading for point primitives.
///
/// Points are rendered unlit at a constant color.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointMaterial {
    /// Unique identifier for this material
    pub id: PointMaterialId,

    /// Point color
    color: RgbaColor,

    /// Generation counter (for change tracking)
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::initial_generation"))]
    generation: u64,
}

impl PointMaterial {
    /// Create a new point material with the given color.
    pub fn new(color: RgbaColor) -> Self {
        Self {
            id: PointMaterialId::new(),
            color,
            generation: crate::initial_generation(),
        }
    }

    /// Get the point color.
    pub fn color(&self) -> RgbaColor {
        self.color
    }

    /// Returns the generation counter. Increments on any mutation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Set the point color (chainable).
    pub fn with_color(mut self, color: RgbaColor) -> Self {
        self.set_color(color);
        self
    }

    /// Set the point color, marking the material as dirty.
    pub fn set_color(&mut self, color: RgbaColor) {
        self.color = color;
        self.generation += 1;
    }
}
