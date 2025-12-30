use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

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
        const NONE = 0b0000;
        /// Enable alpha blending (TODO)
        const ALPHA_BLEND = 0b0001;
        /// Disable back-face culling (TODO)
        const DOUBLE_SIDED = 0b0010;
    }
}

/// PBR material parameters for GPU uniform buffer.
///
/// This struct is sent to the shader and contains all scalar factors
/// plus flags indicating which textures are present.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct PbrUniform {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub texture_flags: u32,
}

impl PbrUniform {
    pub const FLAG_HAS_BASE_COLOR_TEXTURE: u32 = 1 << 0;
    pub const FLAG_HAS_NORMAL_TEXTURE: u32 = 1 << 1;
    pub const FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE: u32 = 1 << 2;
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
    /// Additional rendering flags
    pub flags: MaterialFlags,
}

/// The ID of the default material created automatically by the Scene.
///
/// This material is always available with ID 0 and provides fallback
/// rendering for faces (magenta), lines (black), and points (black).
pub const DEFAULT_MATERIAL_ID: MaterialId = 0;

/// Unique identifier for materials.
///
/// Material IDs are assigned sequentially by the Scene starting from 1
/// (ID 0 is reserved for the default material).
pub type MaterialId = u32;

/// GPU resources for a single primitive type (face, line, or point)
pub(crate) struct MaterialGpuResources {
    pub bind_group: wgpu::BindGroup,
    pub _buffer: Option<wgpu::Buffer>, // For color materials
}

/// PBR material that can be rendered as faces, lines, or points.
///
/// Supports physically-based rendering with base color, normal maps,
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
pub struct Material {
    /// Unique identifier for this material
    pub id: MaterialId,

    // PBR textures (optional, for face rendering)
    /// Base color texture (albedo)
    pub base_color_texture: Option<TextureId>,
    /// Normal map texture
    pub normal_texture: Option<TextureId>,
    /// Metallic-roughness texture (G=roughness, B=metallic per glTF spec)
    pub metallic_roughness_texture: Option<TextureId>,

    // PBR scalar factors (always present, used as multipliers or fallbacks)
    /// Base color factor (multiplied with texture if present)
    pub base_color_factor: RgbaColor,
    /// Metallic factor (0.0 = dielectric, 1.0 = metal)
    pub metallic_factor: f32,
    /// Roughness factor (0.0 = smooth, 1.0 = rough)
    pub roughness_factor: f32,
    /// Normal map scale
    pub normal_scale: f32,

    // Line rendering data (no lighting, no PBR)
    /// Line color
    pub line_color: Option<RgbaColor>,

    // Point rendering data (no lighting, no PBR)
    /// Point color
    pub point_color: Option<RgbaColor>,

    // GPU resources per primitive type (created lazily)
    pub(crate) face_gpu: Option<MaterialGpuResources>,
    pub(crate) line_gpu: Option<MaterialGpuResources>,
    pub(crate) point_gpu: Option<MaterialGpuResources>,

    // Dirty flags per primitive type
    face_dirty: bool,
    line_dirty: bool,
    point_dirty: bool,
}

impl Material {
    /// Create a new material with default PBR values.
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
            face_gpu: None,
            line_gpu: None,
            point_gpu: None,
            face_dirty: true,
            line_dirty: true,
            point_dirty: true,
        }
    }

    // ========== Builder methods (chainable) ==========

    /// Set the base color texture.
    pub fn with_base_color_texture(mut self, texture_id: TextureId) -> Self {
        self.base_color_texture = Some(texture_id);
        self.face_dirty = true;
        self
    }

    /// Set the normal map texture.
    pub fn with_normal_texture(mut self, texture_id: TextureId) -> Self {
        self.normal_texture = Some(texture_id);
        self.face_dirty = true;
        self
    }

    /// Set the metallic-roughness texture.
    pub fn with_metallic_roughness_texture(mut self, texture_id: TextureId) -> Self {
        self.metallic_roughness_texture = Some(texture_id);
        self.face_dirty = true;
        self
    }

    /// Set the base color factor (multiplied with texture if present).
    pub fn with_base_color_factor(mut self, color: RgbaColor) -> Self {
        self.base_color_factor = color;
        self.face_dirty = true;
        self
    }

    /// Set the metallic factor (0.0 = dielectric, 1.0 = metal).
    pub fn with_metallic_factor(mut self, metallic: f32) -> Self {
        self.metallic_factor = metallic;
        self.face_dirty = true;
        self
    }

    /// Set the roughness factor (0.0 = smooth, 1.0 = rough).
    pub fn with_roughness_factor(mut self, roughness: f32) -> Self {
        self.roughness_factor = roughness;
        self.face_dirty = true;
        self
    }

    /// Set the normal map scale.
    pub fn with_normal_scale(mut self, scale: f32) -> Self {
        self.normal_scale = scale;
        self.face_dirty = true;
        self
    }

    /// Set the line color.
    pub fn with_line_color(mut self, color: RgbaColor) -> Self {
        self.line_color = Some(color);
        self.line_dirty = true;
        self
    }

    /// Set the point color.
    pub fn with_point_color(mut self, color: RgbaColor) -> Self {
        self.point_color = Some(color);
        self.point_dirty = true;
        self
    }

    // ========== Mutation methods (set dirty flags) ==========

    /// Set the base color texture, marking the material as dirty.
    pub fn set_base_color_texture(&mut self, texture_id: TextureId) {
        self.base_color_texture = Some(texture_id);
        self.face_dirty = true;
    }

    /// Set the base color factor, marking the material as dirty.
    pub fn set_base_color_factor(&mut self, color: RgbaColor) {
        self.base_color_factor = color;
        self.face_dirty = true;
    }

    /// Set the line color, marking the material as dirty.
    pub fn set_line_color(&mut self, color: RgbaColor) {
        self.line_color = Some(color);
        self.line_dirty = true;
    }

    /// Set the point color, marking the material as dirty.
    pub fn set_point_color(&mut self, color: RgbaColor) {
        self.point_color = Some(color);
        self.point_dirty = true;
    }

    // ========== Query methods ==========

    /// Get the material properties for a given primitive type.
    ///
    /// This is used by ShaderGenerator and PipelineManager to determine
    /// which shader variant to use.
    pub fn get_properties(&self, primitive_type: PrimitiveType) -> MaterialProperties {
        match primitive_type {
            PrimitiveType::TriangleList => MaterialProperties {
                has_base_color_texture: self.base_color_texture.is_some(),
                has_normal_map: self.normal_texture.is_some(),
                has_metallic_roughness_texture: self.metallic_roughness_texture.is_some(),
                has_lighting: true,
                flags: MaterialFlags::NONE,
            },
            PrimitiveType::LineList | PrimitiveType::PointList => MaterialProperties {
                has_base_color_texture: false,
                has_normal_map: false,
                has_metallic_roughness_texture: false,
                has_lighting: false,
                flags: MaterialFlags::NONE,
            },
        }
    }

    /// Build the PBR uniform for GPU upload.
    pub fn build_pbr_uniform(&self) -> PbrUniform {
        let mut texture_flags = 0u32;
        if self.base_color_texture.is_some() {
            texture_flags |= PbrUniform::FLAG_HAS_BASE_COLOR_TEXTURE;
        }
        if self.normal_texture.is_some() {
            texture_flags |= PbrUniform::FLAG_HAS_NORMAL_TEXTURE;
        }
        if self.metallic_roughness_texture.is_some() {
            texture_flags |= PbrUniform::FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE;
        }

        PbrUniform {
            base_color_factor: [
                self.base_color_factor.r,
                self.base_color_factor.g,
                self.base_color_factor.b,
                self.base_color_factor.a,
            ],
            metallic_factor: self.metallic_factor,
            roughness_factor: self.roughness_factor,
            normal_scale: self.normal_scale,
            texture_flags,
        }
    }

    /// Check if GPU resources need to be created or updated for a primitive type.
    pub(crate) fn needs_gpu_resources(&self, primitive_type: PrimitiveType) -> bool {
        match primitive_type {
            PrimitiveType::TriangleList => {
                self.face_gpu.is_none() || self.face_dirty
            }
            PrimitiveType::LineList => {
                self.line_gpu.is_none() || self.line_dirty
            }
            PrimitiveType::PointList => {
                self.point_gpu.is_none() || self.point_dirty
            }
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

    /// Get the GPU resources for a primitive type.
    ///
    /// Returns `None` if GPU resources haven't been created yet.
    pub(crate) fn get_gpu(&self, primitive_type: PrimitiveType) -> Option<&MaterialGpuResources> {
        match primitive_type {
            PrimitiveType::TriangleList => self.face_gpu.as_ref(),
            PrimitiveType::LineList => self.line_gpu.as_ref(),
            PrimitiveType::PointList => self.point_gpu.as_ref(),
        }
    }

    /// Bind this material's resources for the given primitive type.
    ///
    /// # Panics
    /// Panics if GPU resources haven't been initialized for this primitive type.
    pub(crate) fn bind(&self, pass: &mut wgpu::RenderPass, primitive_type: PrimitiveType) {
        debug_assert!(!self.needs_gpu_resources(primitive_type), "Material resources out of date");
        let gpu = self.get_gpu(primitive_type)
            .expect("Material GPU resources not initialized");
        pass.set_bind_group(2, &gpu.bind_group, &[]);
    }

    /// Mark dirty flags as clean for a primitive type.
    pub(crate) fn mark_clean(&mut self, primitive_type: PrimitiveType) {
        match primitive_type {
            PrimitiveType::TriangleList => self.face_dirty = false,
            PrimitiveType::LineList => self.line_dirty = false,
            PrimitiveType::PointList => self.point_dirty = false,
        }
    }

    /// Set GPU resources for a primitive type.
    pub(crate) fn set_gpu(&mut self, primitive_type: PrimitiveType, gpu: MaterialGpuResources) {
        match primitive_type {
            PrimitiveType::TriangleList => self.face_gpu = Some(gpu),
            PrimitiveType::LineList => self.line_gpu = Some(gpu),
            PrimitiveType::PointList => self.point_gpu = Some(gpu),
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::new()
    }
}
