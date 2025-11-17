use std::{collections::HashMap, path::Path};

use anyhow::Ok;
use bytemuck::bytes_of;
use wgpu::{
    util::{
        BufferInitDescriptor,
        DeviceExt
    }
};

use crate::{
    common::RgbaColor, texture::Texture
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

/// A material variant that encapsulates all supported material types.
///
/// This enum provides a unified interface for working with different material types.
pub enum Material {
    /// Solid color material for face primitives
    FaceColor(FaceColorMaterial),
    /// Textured material for face primitives
    FaceTexture(FaceTextureMaterial),
    /// Solid color material for line primitives
    LineColor(LineColorMaterial),
    /// Solid color material for point primitives
    PointColor(PointColorMaterial),
}

impl Material {
    /// Returns the unique identifier for this material.
    pub fn id(&self) -> MaterialId {
        match self {
            Self::FaceColor(material) => material.id,
            Self::FaceTexture(material) => material.id,
            Self::LineColor(material) => material.id,
            Self::PointColor(material) => material.id,
        }
    }

    /// Binds this material's resources to the render pass at bind group 2.
    pub fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        match self {
            Self::FaceColor(material) => material.bind(pass),
            Self::FaceTexture(material) => material.bind(pass),
            Self::LineColor(material) => material.bind(pass),
            Self::PointColor(material) => material.bind(pass),
        }
    }

    /// Returns the type category of this material.
    pub fn material_type(&self) -> MaterialType {
        match self {
            Self::FaceColor(_) => MaterialType::FaceColor,
            Self::FaceTexture(_) => MaterialType::FaceTexture,
            Self::LineColor(_) => MaterialType::LineColor,
            Self::PointColor(_) => MaterialType::PointColor,
        }
    }
}


/// Solid color material for rendering face primitives.
///
/// Uses a uniform buffer to store the diffuse color and applies
/// simple diffuse lighting in the fragment shader.
pub struct FaceColorMaterial {
    /// Unique identifier for this material
    pub id: MaterialId,
    /// Diffuse color (RGBA)
    pub diffuse: RgbaColor,

    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup
}

impl FaceColorMaterial {
    fn new(
        id: MaterialId,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        diffuse: RgbaColor
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Face Color Material Buffer"),
            contents: bytes_of(&diffuse),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Face Color Material Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding()
                }
            ]
        });

        Self {
            id,
            diffuse,
            buffer,
            bind_group
        }
    }

    fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        pass.set_bind_group(2, &self.bind_group, &[]);
        Ok(())
    }
}


/// Textured material for rendering face primitives.
///
/// Uses a 2D texture and sampler for the diffuse channel. The texture is sampled
/// using the mesh's UV coordinates and modulated by diffuse lighting.
pub struct FaceTextureMaterial {
    /// Unique identifier for this material
    pub id: MaterialId,
    /// Diffuse texture
    pub diffuse: Texture,

    bind_group: wgpu::BindGroup
}

impl FaceTextureMaterial {
    fn new_from_path(
        id: MaterialId,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        path: &Path
    ) -> anyhow::Result<Self> {
        let image_bytes = std::fs::read(path)?;
        let diffuse = Texture::from_bytes(device, queue, &image_bytes, "Face Texture Material Diffuse Texture")?;

        Self::new_from_texture(id, device, bind_group_layout, diffuse)
    }

    fn new_from_texture(
        id: MaterialId,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        texture: Texture,
    ) -> anyhow::Result<Self> {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Face Texture Material Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler)
                }
            ]
        });

        Ok(Self {
            id,
            diffuse: texture,
            bind_group
        })
    }

    fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        pass.set_bind_group(2, &self.bind_group, &[]);
        Ok(())
    }
}


/// Solid color material for rendering line primitives.
///
/// Uses a uniform buffer to store the line color. Lines are rendered
/// without lighting calculations.
pub struct LineColorMaterial {
    /// Unique identifier for this material
    pub id: MaterialId,
    /// Line color (RGBA)
    pub color: RgbaColor,

    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup
}

impl LineColorMaterial {
    fn new(
        id: MaterialId,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color: RgbaColor
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Line Color Material Buffer"),
            contents: bytes_of(&color),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Line Color Material Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding()
                }
            ]
        });

        Self {
            id,
            color,
            buffer,
            bind_group
        }
    }

    fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        pass.set_bind_group(2, &self.bind_group, &[]);
        Ok(())
    }
}


/// Solid color material for rendering point primitives.
///
/// Uses a uniform buffer to store the point color. Points are rendered
/// without lighting calculations.
pub struct PointColorMaterial {
    /// Unique identifier for this material
    pub id: MaterialId,
    /// Point color (RGBA)
    pub color: RgbaColor,

    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup
}

impl PointColorMaterial {
    fn new(
        id: MaterialId,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color: RgbaColor
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Point Color Material Buffer"),
            contents: bytes_of(&color),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Point Color Material Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding()
                }
            ]
        });

        Self {
            id,
            color,
            buffer,
            bind_group
        }
    }

    fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        pass.set_bind_group(2, &self.bind_group, &[]);
        Ok(())
    }
}


/// Manages material creation, storage, and GPU resource allocation.
///
/// The MaterialManager maintains a registry of all materials and their associated
/// GPU resources (buffers, textures, bind groups). It provides bind group layouts
/// for both color and texture materials and ensures all materials have unique IDs.
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

    /// Bind group layout for color-based materials (binding 0: uniform buffer)
    pub color_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for texture materials (binding 0: texture, binding 1: sampler)
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl MaterialManager {
    /// Creates a new MaterialManager with default materials and bind group layouts.
    ///
    /// This initializes the material registry with three default materials (Face, Line, Point)
    /// and creates the required bind group layouts for color and texture materials.
    pub fn new(device: &wgpu::Device) -> Self {
        let mut materials = HashMap::new();
        let next_id = 3; // 0-2 are default materials

        let color_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                }
            ]
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let default_face_color_material = FaceColorMaterial::new(
            DefaultMaterial::Face as u32, device, &color_bind_group_layout, DEFAULT_FACE_COLOR
        );
        let default_line_color_material = LineColorMaterial::new(
            DefaultMaterial::Line as u32, device, &color_bind_group_layout, DEFAULT_LINE_COLOR
        );
        let default_point_color_material = PointColorMaterial::new(
            DefaultMaterial::Point as u32, device, &color_bind_group_layout, DEFAULT_POINT_COLOR
        );
        materials.insert(DefaultMaterial::Face as u32, Material::FaceColor(default_face_color_material));
        materials.insert(DefaultMaterial::Line as u32, Material::LineColor(default_line_color_material));
        materials.insert(DefaultMaterial::Point as u32, Material::PointColor(default_point_color_material));

        Self {
            materials,
            next_id,
            color_bind_group_layout,
            texture_bind_group_layout,
        }
    }

    /// Creates a new solid color material for face primitives.
    ///
    /// # Arguments
    /// * `device` - The WGPU device for creating GPU resources
    /// * `color` - The diffuse color for the material
    ///
    /// # Returns
    /// The unique MaterialId for the created material
    pub fn create_face_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = FaceColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::FaceColor(material));
        id
    }

    /// Creates a new solid color material for line primitives.
    ///
    /// # Arguments
    /// * `device` - The WGPU device for creating GPU resources
    /// * `color` - The line color
    ///
    /// # Returns
    /// The unique MaterialId for the created material
    pub fn create_line_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = LineColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::LineColor(material));
        id
    }

    /// Creates a new solid color material for point primitives.
    ///
    /// # Arguments
    /// * `device` - The WGPU device for creating GPU resources
    /// * `color` - The point color
    ///
    /// # Returns
    /// The unique MaterialId for the created material
    pub fn create_point_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = PointColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::PointColor(material));
        id
    }

    /// Creates a new textured material for face primitives from an image file.
    ///
    /// Loads an image from the filesystem and creates a texture material. Supports
    /// common image formats (PNG, JPEG, etc.) through the `image` crate.
    ///
    /// # Arguments
    /// * `device` - The WGPU device for creating GPU resources
    /// * `queue` - The WGPU queue for uploading texture data
    /// * `diffuse_path` - Path to the diffuse texture image file
    ///
    /// # Returns
    /// The unique MaterialId for the created material, or an error if loading fails
    pub fn create_face_texture_material_from_path<P: AsRef<Path>>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        diffuse_path: P
    ) -> anyhow::Result<MaterialId> {
        let id = self.next_id;
        self.next_id += 1;

        let material = FaceTextureMaterial::new_from_path(
            id,
            device,
            queue,
            &self.texture_bind_group_layout,
            diffuse_path.as_ref()
        )?;

        self.materials.insert(id, Material::FaceTexture(material));

        Ok(id)
    }

    /// Creates a new textured material for face primitives from an existing texture.
    ///
    /// # Arguments
    /// * `device` - The WGPU device for creating GPU resources
    /// * `texture` - Pre-loaded texture to use as the diffuse map
    ///
    /// # Returns
    /// The unique MaterialId for the created material, or an error if creation fails
    pub fn create_face_texture_material(
        &mut self,
        device: &wgpu::Device,
        texture: crate::texture::Texture,
    ) -> anyhow::Result<MaterialId> {
        let id = self.next_id;
        self.next_id += 1;

        let material = FaceTextureMaterial::new_from_texture(
            id,
            device,
            &self.texture_bind_group_layout,
            texture,
        )?;

        self.materials.insert(id, Material::FaceTexture(material));

        Ok(id)
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

    /// Binds a material's resources to a render pass.
    ///
    /// This sets the material's bind group at slot 2 in the render pass,
    /// making the material's uniforms and textures available to the shaders.
    ///
    /// # Arguments
    /// * `id` - The MaterialId to bind
    /// * `pass` - The render pass to bind to
    ///
    /// # Returns
    /// Ok(()) on success, or an error if the material doesn't exist
    pub fn bind(&self, id: MaterialId, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        let material = self.materials
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Attempt to bind non-existent material"))?;
        material.bind(pass)?;
        Ok(())
    }
}