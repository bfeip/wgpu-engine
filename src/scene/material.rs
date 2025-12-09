use bitflags::bitflags;

use crate::common::RgbaColor;

use super::mesh::PrimitiveType;
use super::texture::TextureId;

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

/// Material properties that determine shader generation and rendering behavior
///
/// This drives:
/// - Shader generation (ShaderGenerator uses these for conditional compilation)
/// - Pipeline creation (different properties = different pipelines)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MaterialProperties {
    /// Whether this material uses a texture (vs solid color)
    pub has_texture: bool,
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
    pub buffer: Option<wgpu::Buffer>, // For color materials
}

/// Unified material that can be rendered as faces, lines, or points.
///
/// Materials can be created without GPU resources and will have their
/// GPU resources created lazily during rendering. Dirty flags track
/// when GPU resources need to be created or updated.
///
/// # Examples
///
/// ```ignore
/// // Create a material with face color (no GPU needed)
/// let material = Material::new()
///     .with_face_color(RgbaColor::RED)
///     .with_line_color(RgbaColor::BLACK);
///
/// // Add to scene
/// let mat_id = scene.add_material(material);
///
/// // GPU resources are created automatically during rendering
/// ```
pub struct Material {
    /// Unique identifier for this material
    pub id: MaterialId,

    // Face rendering data (triangles with lighting)
    /// Face color (if not using texture)
    pub face_color: Option<RgbaColor>,
    /// Face texture ID (if not using solid color) - references Scene's textures
    pub face_texture: Option<TextureId>,

    // Line rendering data (no lighting)
    /// Line color
    pub line_color: Option<RgbaColor>,

    // Point rendering data (no lighting)
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
    /// Create a new empty material.
    ///
    /// The material has no colors or textures set. Use builder methods
    /// like `with_face_color()` to configure it.
    pub fn new() -> Self {
        Self {
            id: 0, // Assigned by Scene
            face_color: None,
            face_texture: None,
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

    /// Set the face color (for solid color face rendering).
    ///
    /// Note: face_color and face_texture are mutually exclusive.
    /// Setting face_color clears any face_texture.
    pub fn with_face_color(mut self, color: RgbaColor) -> Self {
        self.face_color = Some(color);
        self.face_texture = None;
        self.face_dirty = true;
        self
    }

    /// Set the face texture ID (for textured face rendering).
    ///
    /// Note: face_color and face_texture are mutually exclusive.
    /// Setting face_texture clears any face_color.
    pub fn with_face_texture(mut self, texture_id: TextureId) -> Self {
        self.face_texture = Some(texture_id);
        self.face_color = None;
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

    /// Set the face color, marking the material as dirty.
    pub fn set_face_color(&mut self, color: RgbaColor) {
        self.face_color = Some(color);
        self.face_texture = None;
        self.face_dirty = true;
    }

    /// Set the face texture ID, marking the material as dirty.
    pub fn set_face_texture(&mut self, texture_id: TextureId) {
        self.face_texture = Some(texture_id);
        self.face_color = None;
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
                has_texture: self.face_texture.is_some(),
                has_lighting: true,
                flags: MaterialFlags::NONE,
            },
            PrimitiveType::LineList | PrimitiveType::PointList => MaterialProperties {
                has_texture: false,
                has_lighting: false,
                flags: MaterialFlags::NONE,
            },
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

    /// Check if any GPU resources need creation or update.
    pub(crate) fn needs_any_gpu_resources(&self) -> bool {
        self.needs_gpu_resources(PrimitiveType::TriangleList)
            || self.needs_gpu_resources(PrimitiveType::LineList)
            || self.needs_gpu_resources(PrimitiveType::PointList)
    }

    /// Check if the material has data for a given primitive type.
    pub fn has_primitive_data(&self, primitive_type: PrimitiveType) -> bool {
        match primitive_type {
            PrimitiveType::TriangleList => {
                self.face_color.is_some() || self.face_texture.is_some()
            }
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
