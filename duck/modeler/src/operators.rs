mod boolean;
mod sphere;

pub use boolean::BooleanOperator;
pub use sphere::SphereOperator;

use duck_engine_common::{Plane, RgbaColor};
use duck_engine_scene::cad::CadTessellationOptions;

use crate::grid::GridConfig;

pub struct ConstructionOptions {
    pub geometry_preview_options: CadTessellationOptions,
    pub construction_plane: Plane,
    pub grid: GridConfig,
}

impl ConstructionOptions {
    pub fn new() -> Self {
        let geometry_preview_options = CadTessellationOptions {
            tessellation_tolerance: 0.01,
            scale_factor: 1.0,
            face_color: RgbaColor { r: 0.55, g: 0.65, b: 0.9, a: 1.0 },
            edge_color: RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            include_edges: true,
        };
        let construction_plane = Plane::xz();
        let grid = GridConfig::default();
        Self { geometry_preview_options, construction_plane, grid }
    }
}