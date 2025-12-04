use std::collections::HashMap;

use bitflags::bitflags;
use bytemuck::bytes_of;
use wgpu::{
    util::{
        BufferInitDescriptor,
        DeviceExt
    }
};

use crate::{
    common::RgbaColor,
    texture::Texture,
    scene::PrimitiveType
};

/// Default magenta color for face materials
const DEFAULT_FACE_COLOR: RgbaColor = RgbaColor {
    r: 1.0,
    g: 0.3,
    b: 1.0,
    a: 1.0
};

/// Default black color for line materials
const DEFAULT_LINE_COLOR: RgbaColor = RgbaColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0
};

/// Default black color for point materials
const DEFAULT_POINT_COLOR: RgbaColor = RgbaColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0
};

bitflags! {
    /// Additional material rendering flags for extensibility
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MaterialFlags: u32 {
        /// No special flags
        const NONE = 0b0000;
        /// Enable alpha blending (future)
        const ALPHA_BLEND = 0b0001;
        /// Disable back-face culling (future)
        const DOUBLE_SIDED = 0b0010;
    }
}

/// Material properties that determine shader generation and rendering behavior
///
/// This is the single source of truth that drives:
/// - Shader generation (ShaderGenerator uses these for conditional compilation)
/// - Bind group layout generation (BindGroupGenerator creates layouts from these)
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

impl MaterialProperties {
    /// Convert from legacy MaterialType (temporary for Phase 2 compatibility)
    pub(crate) fn from_material_type(material_type: MaterialType) -> Self {
        match material_type {
            MaterialType::FaceColor => Self {
                has_texture: false,
                has_lighting: true,
                flags: MaterialFlags::NONE,
            },
            MaterialType::FaceTexture => Self {
                has_texture: true,
                has_lighting: true,
                flags: MaterialFlags::NONE,
            },
            MaterialType::LineColor | MaterialType::PointColor => Self {
                has_texture: false,
                has_lighting: false,
                flags: MaterialFlags::NONE,
            },
        }
    }

    /// Create properties for a lit color material (faces)
    pub fn face_color() -> Self {
        Self {
            has_texture: false,
            has_lighting: true,
            flags: MaterialFlags::NONE,
        }
    }

    /// Create properties for a lit texture material (faces)
    pub fn face_texture() -> Self {
        Self {
            has_texture: true,
            has_lighting: true,
            flags: MaterialFlags::NONE,
        }
    }

    /// Create properties for an unlit color material (lines/points)
    pub fn unlit_color() -> Self {
        Self {
            has_texture: false,
            has_lighting: false,
            flags: MaterialFlags::NONE,
        }
    }
}

/// Default materials created automatically by the MaterialManager.
///
/// These materials are always available with fixed IDs (0, 1, 2) and provide
/// fallback rendering options when custom materials aren't specified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultMaterial {
    /// Default magenta face material (ID = 0)
    Face = 0,
    /// Default black line material (ID = 1)
    Line,
    /// Default black point material (ID = 2)
    Point
}

impl Into<MaterialId> for DefaultMaterial {
    fn into(self) -> MaterialId {
        self as MaterialId
    }
}

/// Unique identifier for materials.
///
/// Material IDs are assigned sequentially by the MaterialManager starting from 3
/// (IDs 0-2 are reserved for default materials).
pub type MaterialId = u32;

/// Categorizes materials by their rendering type.
///
/// Different material types may require different shader pipelines and bind group layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialType {
    /// Solid color material for face primitives
    FaceColor,
    /// Textured material for face primitives
    FaceTexture,
    /// Solid color material for line primitives
    LineColor,
    /// Solid color material for point primitives
    PointColor,
}

/// Unified material that can be rendered as faces, lines, or points
///
/// A single Material instance contains rendering data for all primitive types:
/// - Faces: Can use either a color or texture (with lighting)
/// - Lines: Uses a solid color (no lighting)
/// - Points: Uses a solid color (no lighting)
///
/// At draw time, the appropriate data is bound based on the mesh's primitive type.
pub struct Material {
    /// Unique identifier for this material
    pub id: MaterialId,

    // Face rendering data (triangles with lighting)
    /// Face color (if not using texture)
    pub face_color: Option<RgbaColor>,
    /// Face texture (if not using solid color)
    pub face_texture: Option<Texture>,

    // Line rendering data (no lighting)
    /// Line color
    pub line_color: Option<RgbaColor>,

    // Point rendering data (no lighting)
    /// Point color
    pub point_color: Option<RgbaColor>,

    // GPU resources per primitive type
    face_bind_group: Option<wgpu::BindGroup>,
    line_bind_group: Option<wgpu::BindGroup>,
    point_bind_group: Option<wgpu::BindGroup>,

    // Uniform buffers for colors
    face_buffer: Option<wgpu::Buffer>,
    line_buffer: Option<wgpu::Buffer>,
    point_buffer: Option<wgpu::Buffer>,
}

impl Material {
    /// Get the material properties for a given primitive type
    ///
    /// This is used by ShaderGenerator and PipelineManager to determine
    /// which shader variant to use.
    pub fn get_properties(&self, primitive_type: PrimitiveType) -> MaterialProperties {
        match primitive_type {
            PrimitiveType::TriangleList => {
                if self.face_texture.is_some() {
                    MaterialProperties::face_texture()
                } else {
                    MaterialProperties::face_color()
                }
            },
            PrimitiveType::LineList | PrimitiveType::PointList => {
                MaterialProperties::unlit_color()
            },
        }
    }

    /// Bind this material's resources for the given primitive type
    pub fn bind(&self, pass: &mut wgpu::RenderPass, primitive_type: PrimitiveType) {
        let bind_group = match primitive_type {
            PrimitiveType::TriangleList => self.face_bind_group.as_ref()
                .expect("Material missing face bind group"),
            PrimitiveType::LineList => self.line_bind_group.as_ref()
                .expect("Material missing line bind group"),
            PrimitiveType::PointList => self.point_bind_group.as_ref()
                .expect("Material missing point bind group"),
        };

        pass.set_bind_group(2, bind_group, &[]);
    }

    /// Returns the legacy MaterialType for backwards compatibility during migration
    pub fn material_type(&self) -> MaterialType {
        // For backwards compatibility, report the face material type
        if self.face_texture.is_some() {
            MaterialType::FaceTexture
        } else {
            MaterialType::FaceColor
        }
    }

    /// Create a builder for constructing a unified Material
    ///
    /// Note: Use MaterialManager::create_material() instead to properly
    /// register the material and assign it a unique ID.
    pub fn builder() -> MaterialBuilder {
        MaterialBuilder::new()
    }
}

/// Builder for creating unified Material instances
///
/// Allows flexible construction of materials with optional rendering data
/// for faces, lines, and points. GPU resources (buffers and bind groups)
/// are created automatically based on which data is provided.
///
/// # Usage
///
/// Use MaterialManager::create_material() to create and register a material:
/// ```ignore
/// let id = material_manager.create_material(
///     &device,
///     Material::builder()
///         .with_face_color(color)
///         .with_line_color(line_color)
/// );
/// ```
pub struct MaterialBuilder {
    face_color: Option<RgbaColor>,
    face_texture: Option<Texture>,
    line_color: Option<RgbaColor>,
    point_color: Option<RgbaColor>,
}

impl MaterialBuilder {
    /// Create a new MaterialBuilder
    pub fn new() -> Self {
        Self {
            face_color: None,
            face_texture: None,
            line_color: None,
            point_color: None,
        }
    }

    /// Set the face color (for solid color face rendering)
    ///
    /// Note: face_color and face_texture are mutually exclusive.
    /// If both are set, face_texture takes precedence.
    pub fn with_face_color(mut self, color: RgbaColor) -> Self {
        self.face_color = Some(color);
        self
    }

    /// Set the face texture (for textured face rendering)
    ///
    /// Note: face_color and face_texture are mutually exclusive.
    /// If both are set, face_texture takes precedence.
    pub fn with_face_texture(mut self, texture: Texture) -> Self {
        self.face_texture = Some(texture);
        self
    }

    /// Set the line color
    pub fn with_line_color(mut self, color: RgbaColor) -> Self {
        self.line_color = Some(color);
        self
    }

    /// Set the point color
    pub fn with_point_color(mut self, color: RgbaColor) -> Self {
        self.point_color = Some(color);
        self
    }

    /// Build the Material with the given ID, creating all necessary GPU resources
    ///
    /// This is an internal method. Use MaterialManager::create_material() instead.
    ///
    /// This creates uniform buffers and bind groups for each primitive type
    /// that has data. It uses the BindGroupGenerator to ensure bind group
    /// layouts match the shader expectations.
    pub(crate) fn build(
        self,
        id: MaterialId,
        device: &wgpu::Device,
        bind_group_generator: &mut crate::shaders::BindGroupGenerator,
    ) -> Material {
        let mut face_buffer = None;
        let mut face_bind_group = None;
        let mut line_buffer = None;
        let mut line_bind_group = None;
        let mut point_buffer = None;
        let mut point_bind_group = None;

        // Create face resources (either color or texture)
        if self.face_texture.is_some() || self.face_color.is_some() {
            if let Some(ref texture) = self.face_texture {
                // Texture-based face rendering
                let layout = bind_group_generator.get_or_generate_layout(
                    device,
                    &MaterialProperties::face_texture(),
                );

                face_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Material Face Texture Bind Group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&texture.sampler),
                        },
                    ],
                }));
            } else if let Some(color) = self.face_color {
                // Color-based face rendering
                let buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("Material Face Color Buffer"),
                    contents: bytes_of(&color),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let layout = bind_group_generator.get_or_generate_layout(
                    device,
                    &MaterialProperties::face_color(),
                );

                face_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Material Face Color Bind Group"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                }));

                face_buffer = Some(buffer);
            }
        }

        // Create line resources
        if let Some(color) = self.line_color {
            let buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Material Line Color Buffer"),
                contents: bytes_of(&color),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let layout = bind_group_generator.get_or_generate_layout(
                device,
                &MaterialProperties::unlit_color(),
            );

            line_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Material Line Color Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            }));

            line_buffer = Some(buffer);
        }

        // Create point resources
        if let Some(color) = self.point_color {
            let buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Material Point Color Buffer"),
                contents: bytes_of(&color),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let layout = bind_group_generator.get_or_generate_layout(
                device,
                &MaterialProperties::unlit_color(),
            );

            point_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Material Point Color Bind Group"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            }));

            point_buffer = Some(buffer);
        }

        Material {
            id,
            face_color: self.face_color,
            face_texture: self.face_texture,
            line_color: self.line_color,
            point_color: self.point_color,
            face_bind_group,
            line_bind_group,
            point_bind_group,
            face_buffer,
            line_buffer,
            point_buffer,
        }
    }
}

impl Default for MaterialBuilder {
    fn default() -> Self {
        Self::new()
    }
}
/// Manages material creation, storage, and GPU resource allocation.
///
/// The MaterialManager maintains a registry of all materials and their associated
/// GPU resources (buffers, textures, bind groups). It uses a BindGroupGenerator
/// to ensure bind group layouts stay synchronized with shader expectations.
///
/// # Default Materials
///
/// The manager automatically creates three default materials (IDs 0-2):
/// - Face: Magenta color for debugging missing materials
/// - Line: Black color for line primitives
/// - Point: Black color for point primitives
pub struct MaterialManager {
    materials: HashMap<MaterialId, Material>,
    next_id: MaterialId,

    /// Bind group generator (derives layouts from MaterialProperties)
    bind_group_generator: crate::shaders::BindGroupGenerator,
}

impl MaterialManager {
    /// Creates a new MaterialManager with default materials.
    ///
    /// This initializes the material registry with three default materials (IDs 0-2):
    /// - Face (ID 0): Magenta color material for faces
    /// - Line (ID 1): Black color material for lines
    /// - Point (ID 2): Black color material for points
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let mut manager = Self {
            materials: HashMap::new(),
            next_id: 0,
            bind_group_generator: crate::shaders::BindGroupGenerator::new(),
        };

        // Create default materials with fixed IDs (0, 1, 2)
        // Default face material (ID 0, magenta for debugging)
        let face_id = manager.create_material(
            device,
            Material::builder().with_face_color(DEFAULT_FACE_COLOR),
        );
        assert_eq!(face_id, DefaultMaterial::Face as u32);

        // Default line material (ID 1, black)
        let line_id = manager.create_material(
            device,
            Material::builder().with_line_color(DEFAULT_LINE_COLOR),
        );
        assert_eq!(line_id, DefaultMaterial::Line as u32);

        // Default point material (ID 2, black)
        let point_id = manager.create_material(
            device,
            Material::builder().with_point_color(DEFAULT_POINT_COLOR),
        );
        assert_eq!(point_id, DefaultMaterial::Point as u32);

        manager
    }

    /// Creates and registers a new material from a MaterialBuilder
    ///
    /// This is the primary way to create materials. It handles ID generation,
    /// GPU resource creation, and material registration automatically.
    ///
    /// # Arguments
    /// * `device` - The WGPU device for creating GPU resources
    /// * `builder` - A MaterialBuilder configured with the desired material properties
    ///
    /// # Returns
    /// The unique MaterialId for the created material
    ///
    /// # Example
    /// ```ignore
    /// let id = material_manager.create_material(
    ///     &device,
    ///     Material::builder()
    ///         .with_face_color(RgbaColor::RED)
    ///         .with_line_color(RgbaColor::BLACK)
    /// );
    /// ```
    pub(crate) fn create_material(
        &mut self,
        device: &wgpu::Device,
        builder: MaterialBuilder,
    ) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;

        let material = builder.build(id, device, &mut self.bind_group_generator);
        self.materials.insert(id, material);
        id
    }

    /// Gets a bind group layout for the given material properties
    ///
    /// This is useful for creating pipelines with the correct bind group layouts.
    /// The layout is cached internally by the BindGroupGenerator.
    pub(crate) fn get_bind_group_layout(
        &mut self,
        device: &wgpu::Device,
        properties: &MaterialProperties,
    ) -> &wgpu::BindGroupLayout {
        self.bind_group_generator.get_or_generate_layout(device, properties)
    }

    /// Retrieves a material by its ID.
    ///
    /// # Arguments
    /// * `id` - The MaterialId to look up
    ///
    /// # Returns
    /// A reference to the Material if found, None otherwise
    pub fn get(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(&id)
    }

    /// Binds a material's resources to a render pass for a specific primitive type.
    ///
    /// This sets the material's bind group at slot 2 in the render pass,
    /// making the material's uniforms and textures available to the shaders.
    ///
    /// # Arguments
    /// * `id` - The MaterialId to bind
    /// * `pass` - The render pass to bind to
    /// * `primitive_type` - The primitive type being rendered (TriangleList, LineList, or PointList)
    pub(crate) fn bind(&self, id: MaterialId, pass: &mut wgpu::RenderPass, primitive_type: PrimitiveType) {
        let material = self.materials
            .get(&id)
            .expect("Attempt to bind non-existent material");
        material.bind(pass, primitive_type);
    }
}