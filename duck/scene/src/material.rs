//! Per-primitive material types.
//!
//! Rather than one monolithic material, shading is split into three independent,
//! top-level types — [`FaceMaterial`], [`LineMaterial`], [`PointMaterial`] — each
//! with its own id, scene collection, and generation counter. An [`crate::Instance`]
//! references up to three of them, one per primitive kind it draws.

use bitflags::bitflags;

mod face;
mod line;
mod point;

pub use face::{FaceMaterial, FaceMaterialId};
pub use line::{LineMaterial, LineMaterialId};
pub use point::{PointMaterial, PointMaterialId};

/// Default roughness factor when not specified
pub const DEFAULT_ROUGHNESS: f32 = 0.5;
/// Default metallic factor when not specified
pub const DEFAULT_METALLIC: f32 = 0.0;
/// Default normal scale when not specified
pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;
/// Default alpha cutoff for mask mode (per glTF spec)
pub const DEFAULT_ALPHA_CUTOFF: f32 = 0.5;

/// Alpha rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AlphaMode {
    /// Fully opaque, alpha channel ignored.
    #[default]
    Opaque,
    /// Binary alpha test: alpha >= cutoff is fully opaque, otherwise discarded.
    Mask,
    /// Standard alpha blending (source alpha, one minus source alpha).
    Blend,
}

bitflags! {
    /// Additional face-material options
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    pub struct MaterialFlags: u32 {
        /// No special flags
        const NONE = 0;
        /// Disable back-face culling and flip normals for back faces
        const DOUBLE_SIDED = 1 << 1;
        /// Disables face lighting. Faces will appear at a constant luminance
        const DO_NOT_LIGHT = 1 << 2;
    }
}

/// Material properties helpful to know during shader generation.
///
/// Used by `ShaderGenerator` and `PipelineManager` to determine which shader
/// variant and pipeline state to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialProperties {
    /// Whether lighting calculations should be applied
    pub has_lighting: bool,
    /// Whether the material is double-sided (disables back-face culling, flips normals)
    pub double_sided: bool,
    /// Alpha rendering mode
    pub alpha_mode: AlphaMode,
    /// Whether the material binds a base-color texture
    pub base_color_texture: bool,
    /// Whether the material binds a normal-map texture (lit materials only)
    pub normal_texture: bool,
    /// Whether the material binds a metallic-roughness texture (lit materials only)
    pub metallic_roughness_texture: bool,
}

impl MaterialProperties {
    /// Fixed properties used for untextured line and point primitives: unlit,
    /// opaque, no textures.
    pub const UNLIT_OPAQUE: MaterialProperties = MaterialProperties {
        has_lighting: false,
        double_sided: false,
        alpha_mode: AlphaMode::Opaque,
        base_color_texture: false,
        normal_texture: false,
        metallic_roughness_texture: false,
    };
}
