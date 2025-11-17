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

const DEFAULT_FACE_COLOR: RgbaColor = RgbaColor {
    r: 1.0,
    g: 0.3,
    b: 1.0,
    a: 1.0
};

const DEFAULT_LINE_COLOR: RgbaColor = RgbaColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0
};

const DEFAULT_POINT_COLOR: RgbaColor = RgbaColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0
};

pub type MaterialId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialType {
    FaceColor,
    FaceTexture,
    LineColor,
    PointColor,
}

pub enum Material {
    FaceColor(FaceColorMaterial),
    FaceTexture(FaceTextureMaterial),
    LineColor(LineColorMaterial),
    PointColor(PointColorMaterial),
}

impl Material {
    pub fn id(&self) -> MaterialId {
        match self {
            Self::FaceColor(material) => material.id,
            Self::FaceTexture(material) => material.id,
            Self::LineColor(material) => material.id,
            Self::PointColor(material) => material.id,
        }
    }

    pub fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        match self {
            Self::FaceColor(material) => material.bind(pass),
            Self::FaceTexture(material) => material.bind(pass),
            Self::LineColor(material) => material.bind(pass),
            Self::PointColor(material) => material.bind(pass),
        }
    }

    pub fn material_type(&self) -> MaterialType {
        match self {
            Self::FaceColor(_) => MaterialType::FaceColor,
            Self::FaceTexture(_) => MaterialType::FaceTexture,
            Self::LineColor(_) => MaterialType::LineColor,
            Self::PointColor(_) => MaterialType::PointColor,
        }
    }
}


pub struct FaceColorMaterial {
    pub id: MaterialId,
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


pub struct FaceTextureMaterial {
    pub id: MaterialId,
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


pub struct LineColorMaterial {
    pub id: MaterialId,
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


pub struct PointColorMaterial {
    pub id: MaterialId,
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


pub struct MaterialManager {
    materials: HashMap<MaterialId, Material>,
    next_id: MaterialId,

    pub color_bind_group_layout: wgpu::BindGroupLayout,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl MaterialManager {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut materials = HashMap::new();
        let next_id = 1; // 0 is the default material

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

        let default_face_color_material = FaceColorMaterial::new(0, device, &color_bind_group_layout, DEFAULT_FACE_COLOR);
        materials.insert(0, Material::FaceColor(default_face_color_material));

        Self {
            materials,
            next_id,
            color_bind_group_layout,
            texture_bind_group_layout,
        }
    }

    pub fn create_face_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = FaceColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::FaceColor(material));
        id
    }

    pub fn create_line_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = LineColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::LineColor(material));
        id
    }

    pub fn create_point_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = PointColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::PointColor(material));
        id
    }

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

    pub fn get(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(&id)
    }

    pub fn bind(&self, id: MaterialId, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        let material = self.materials
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Attempt to bind non-existent material"))?;
        material.bind(pass)?;
        Ok(())
    }
}