//! Bind group ABI for the standard scene shader library.
//!
//! These slot assignments are a convention of *this* crate's WESL shaders
//! (mirrored in `shaders/constants.wesl`), not of rendering in general. The
//! scene-agnostic core imposes no bind group layout; a workflow built directly
//! on the core may use any slots it likes. Custom passes that reuse the standard
//! camera/lights/material/IBL bind groups should reference these constants
//! rather than hard-coding literals.

/// Camera uniform (view-projection + eye position).
pub const GROUP_CAMERA: u32 = 0;
/// Lights uniform array.
pub const GROUP_LIGHTS: u32 = 1;
/// Material parameters (color or PBR).
pub const GROUP_MATERIAL: u32 = 2;
/// IBL environment (irradiance, prefiltered, BRDF LUT).
pub const GROUP_IBL: u32 = 3;
