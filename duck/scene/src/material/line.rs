use crate::common::RgbaColor;

/// Unique identifier for a [`LineMaterial`].
pub type LineMaterialId = crate::Id<LineMaterial>;

/// Shading for line (edge / wireframe) primitives.
///
/// Lines are rendered unlit at a constant color.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineMaterial {
    /// Unique identifier for this material
    pub id: LineMaterialId,

    /// Line color
    color: RgbaColor,

    /// Generation counter (for change tracking)
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::initial_generation"))]
    generation: u64,
}

impl LineMaterial {
    /// Create a new line material with the given color.
    pub fn new(color: RgbaColor) -> Self {
        Self {
            id: LineMaterialId::new(),
            color,
            generation: crate::initial_generation(),
        }
    }

    /// Get the line color.
    pub fn color(&self) -> RgbaColor {
        self.color
    }

    /// Returns the generation counter. Increments on any mutation.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Set the line color (chainable).
    pub fn with_color(mut self, color: RgbaColor) -> Self {
        self.set_color(color);
        self
    }

    /// Set the line color, marking the material as dirty.
    pub fn set_color(&mut self, color: RgbaColor) {
        self.color = color;
        self.generation += 1;
    }
}
