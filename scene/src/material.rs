use bitflags::bitflags;

use crate::common::RgbaColor;

use super::mesh::PrimitiveType;
use super::texture::TextureId;

/// Default roughness factor when not specified
pub const DEFAULT_ROUGHNESS: f32 = 0.5;
/// Default metallic factor when not specified
pub const DEFAULT_METALLIC: f32 = 0.0;
/// Default normal scale when not specified
pub const DEFAULT_NORMAL_SCALE: f32 = 1.0;

bitflags! {
    /// Additional material rendering flags for extensibility
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MaterialFlags: u32 {
        /// No special flags
        const NONE = 0;
        /// Enable alpha blending (TODO)
        const ALPHA_BLEND = 1 << 0;
        /// Disable back-face culling (TODO)
        const DOUBLE_SIDED = 1 << 1;
        /// Disables face lighting. Faces will appear at a constant luminance
        const DO_NOT_LIGHT = 1 << 2;
    }
}

/// Material properties that determine shader generation and rendering behavior.
///
/// This drives:
/// - Shader generation (ShaderGenerator uses these for conditional compilation)
/// - Pipeline creation (different properties = different pipelines)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialProperties {
    /// Whether this material has a base color texture
    pub has_base_color_texture: bool,
    /// Whether this material has a normal map
    pub has_normal_map: bool,
    /// Whether this material has a metallic-roughness texture
    pub has_metallic_roughness_texture: bool,
    /// Whether lighting calculations should be applied
    pub has_lighting: bool,
}

/// The ID of the default material created automatically by the Scene.
///
/// This material is always available at a sentinel ID (`u32::MAX`) and provides
/// fallback rendering for faces (magenta), lines (black), and points (black).
/// Using `u32::MAX` ensures it never collides with user-assigned material IDs
/// which are assigned sequentially starting from 0.
pub const DEFAULT_MATERIAL_ID: MaterialId = u32::MAX;

/// Unique identifier for materials.
///
/// Material IDs are assigned sequentially by the Scene starting from 0.
pub type MaterialId = u32;

/// Material that can be rendered as faces, lines, or points.
///
/// Supports physically-based rendering for faces with base color, normal maps,
/// and metallic-roughness textures. Missing textures fall back to scalar factors.
///
/// # Examples
///
/// ```
/// use wgpu_engine::scene::{Material, Scene};
/// use wgpu_engine::common::RgbaColor;
///
/// // Create a simple colored material
/// let material = Material::new()
///     .with_base_color_factor(RgbaColor::RED)
///     .with_line_color(RgbaColor::BLACK);
///
/// // Add to scene
/// let mut scene = Scene::new();
/// let mat_id = scene.add_material(material);
///
/// // PBR materials can also have textures:
/// // material.with_base_color_texture(texture_id)
/// //         .with_metallic_factor(0.0)
/// //         .with_roughness_factor(0.5);
/// ```
#[derive(Clone)]
pub struct Material {
    /// Unique identifier for this material
    pub id: MaterialId,

    /// Base color texture (albedo)
    base_color_texture: Option<TextureId>,
    /// Normal map texture
    normal_texture: Option<TextureId>,
    /// Metallic-roughness texture (G=roughness, B=metallic per glTF spec)
    metallic_roughness_texture: Option<TextureId>,

    /// Base color factor (multiplied with texture if present)
    base_color_factor: RgbaColor,
    /// Metallic factor (0.0 = dielectric, 1.0 = metal)
    metallic_factor: f32,
    /// Roughness factor (0.0 = smooth, 1.0 = rough)
    roughness_factor: f32,
    /// Normal map scale
    normal_scale: f32,

    /// Line color
    line_color: Option<RgbaColor>,

    /// Point color
    point_color: Option<RgbaColor>,

    /// Rendering flags
    flags: MaterialFlags,

    // Generation counters per primitive type (for GPU sync tracking)
    face_generation: u64,
    line_generation: u64,
    point_generation: u64,
}

impl Material {
    /// Create a new material with default values.
    ///
    /// Defaults: white base color, metallic=0.0, roughness=0.5, no textures.
    pub fn new() -> Self {
        Self {
            id: 0, // Assigned by Scene
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            base_color_factor: RgbaColor::WHITE,
            metallic_factor: DEFAULT_METALLIC,
            roughness_factor: DEFAULT_ROUGHNESS,
            normal_scale: DEFAULT_NORMAL_SCALE,
            line_color: None,
            point_color: None,
            flags: MaterialFlags::NONE,
            face_generation: 1,
            line_generation: 1,
            point_generation: 1,
        }
    }

    // ========== Getter methods ==========

    /// Get the base color texture ID.
    pub fn base_color_texture(&self) -> Option<TextureId> {
        self.base_color_texture
    }

    /// Get the normal map texture ID.
    pub fn normal_texture(&self) -> Option<TextureId> {
        self.normal_texture
    }

    /// Get the metallic-roughness texture ID.
    pub fn metallic_roughness_texture(&self) -> Option<TextureId> {
        self.metallic_roughness_texture
    }

    /// Get the base color factor.
    pub fn base_color_factor(&self) -> RgbaColor {
        self.base_color_factor
    }

    /// Get the metallic factor.
    pub fn metallic_factor(&self) -> f32 {
        self.metallic_factor
    }

    /// Get the roughness factor.
    pub fn roughness_factor(&self) -> f32 {
        self.roughness_factor
    }

    /// Get the normal map scale.
    pub fn normal_scale(&self) -> f32 {
        self.normal_scale
    }

    /// Get the line color.
    pub fn line_color(&self) -> Option<RgbaColor> {
        self.line_color
    }

    /// Get the point color.
    pub fn point_color(&self) -> Option<RgbaColor> {
        self.point_color
    }

    /// Get the flags.
    pub fn flags(&self) -> MaterialFlags {
        self.flags
    }

    // ========== Builder methods (chainable) ==========

    /// Set the base color texture.
    pub fn with_base_color_texture(mut self, texture_id: TextureId) -> Self {
        self.base_color_texture = Some(texture_id);
        self.face_generation += 1;
        self
    }

    /// Set the normal map texture.
    pub fn with_normal_texture(mut self, texture_id: TextureId) -> Self {
        self.normal_texture = Some(texture_id);
        self.face_generation += 1;
        self
    }

    /// Set the metallic-roughness texture.
    pub fn with_metallic_roughness_texture(mut self, texture_id: TextureId) -> Self {
        self.metallic_roughness_texture = Some(texture_id);
        self.face_generation += 1;
        self
    }

    /// Set the base color factor (multiplied with texture if present).
    pub fn with_base_color_factor(mut self, color: RgbaColor) -> Self {
        self.base_color_factor = color;
        self.face_generation += 1;
        self
    }

    /// Set the metallic factor (0.0 = dielectric, 1.0 = metal).
    pub fn with_metallic_factor(mut self, metallic: f32) -> Self {
        self.metallic_factor = metallic;
        self.face_generation += 1;
        self
    }

    /// Set the roughness factor (0.0 = smooth, 1.0 = rough).
    pub fn with_roughness_factor(mut self, roughness: f32) -> Self {
        self.roughness_factor = roughness;
        self.face_generation += 1;
        self
    }

    /// Set the normal map scale.
    pub fn with_normal_scale(mut self, scale: f32) -> Self {
        self.normal_scale = scale;
        self.face_generation += 1;
        self
    }

    /// Set the line color.
    pub fn with_line_color(mut self, color: RgbaColor) -> Self {
        self.line_color = Some(color);
        self.line_generation += 1;
        self
    }

    /// Set the point color.
    pub fn with_point_color(mut self, color: RgbaColor) -> Self {
        self.point_color = Some(color);
        self.point_generation += 1;
        self
    }

    pub fn with_flags(mut self, flags: MaterialFlags) -> Self {
        self.flags = flags;
        self
    }

    // ========== Mutation methods (increment generation) ==========

    /// Set the base color texture, incrementing the face generation.
    pub fn set_base_color_texture(&mut self, texture_id: TextureId) {
        self.base_color_texture = Some(texture_id);
        self.face_generation += 1;
    }

    /// Set the base color factor, marking the material as dirty.
    pub fn set_base_color_factor(&mut self, color: RgbaColor) {
        self.base_color_factor = color;
        self.face_generation += 1;
    }

    /// Set the normal map texture, marking the material as dirty.
    pub fn set_normal_texture(&mut self, texture_id: TextureId) {
        self.normal_texture = Some(texture_id);
        self.face_generation += 1;
    }

    /// Set the metallic-roughness texture, marking the material as dirty.
    pub fn set_metallic_roughness_texture(&mut self, texture_id: TextureId) {
        self.metallic_roughness_texture = Some(texture_id);
        self.face_generation += 1;
    }

    /// Set the metallic factor, marking the material as dirty.
    pub fn set_metallic_factor(&mut self, metallic: f32) {
        self.metallic_factor = metallic;
        self.face_generation += 1;
    }

    /// Set the roughness factor, marking the material as dirty.
    pub fn set_roughness_factor(&mut self, roughness: f32) {
        self.roughness_factor = roughness;
        self.face_generation += 1;
    }

    /// Set the normal map scale, marking the material as dirty.
    pub fn set_normal_scale(&mut self, scale: f32) {
        self.normal_scale = scale;
        self.face_generation += 1;
    }

    /// Set the line color, marking the material as dirty.
    pub fn set_line_color(&mut self, color: RgbaColor) {
        self.line_color = Some(color);
        self.line_generation += 1;
    }

    /// Set the point color, marking the material as dirty.
    pub fn set_point_color(&mut self, color: RgbaColor) {
        self.point_color = Some(color);
        self.point_generation += 1;
    }

    pub fn set_flags(&mut self, flags: MaterialFlags) {
        self.flags = flags;
    }

    // ========== Query methods ==========

    /// Returns the generation counter for a primitive type.
    ///
    /// This value increments on any mutation to the material data for that primitive type.
    /// Used by renderers to track when GPU resources need updating.
    pub fn generation(&self, primitive_type: PrimitiveType) -> u64 {
        match primitive_type {
            PrimitiveType::TriangleList => self.face_generation,
            PrimitiveType::LineList => self.line_generation,
            PrimitiveType::PointList => self.point_generation,
        }
    }

    /// Get the material properties for a given primitive type.
    ///
    /// This is used by ShaderGenerator and PipelineManager to determine
    /// which shader variant to use.
    pub fn get_properties(&self, primitive_type: PrimitiveType) -> MaterialProperties {
        let faces_have_lighting = !self.flags.contains(MaterialFlags::DO_NOT_LIGHT);
        match primitive_type {
            PrimitiveType::TriangleList => MaterialProperties {
                has_base_color_texture: self.base_color_texture.is_some(),
                has_normal_map: self.normal_texture.is_some(),
                has_metallic_roughness_texture: self.metallic_roughness_texture.is_some(),
                has_lighting: faces_have_lighting,
            },
            PrimitiveType::LineList | PrimitiveType::PointList => MaterialProperties {
                has_base_color_texture: false,
                has_normal_map: false,
                has_metallic_roughness_texture: false,
                has_lighting: false,
            },
        }
    }

    /// Check if the material has data for a given primitive type.
    ///
    /// For faces, PBR materials always have data (at minimum the base_color_factor).
    pub fn has_primitive_data(&self, primitive_type: PrimitiveType) -> bool {
        match primitive_type {
            // PBR materials always have face data (base_color_factor is always set)
            PrimitiveType::TriangleList => true,
            PrimitiveType::LineList => self.line_color.is_some(),
            PrimitiveType::PointList => self.point_color.is_some(),
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}
