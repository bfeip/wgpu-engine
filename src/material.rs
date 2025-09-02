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

const DEFAULT_COLOR: RgbaColor = RgbaColor {
    r: 1.0,
    g: 0.3,
    b: 1.0,
    a: 1.0
};

pub type MaterialId = u32;

pub enum MaterialType {
    Color,
    Texture
}

pub enum Material {
    Color(ColorMaterial),
    Texture(TextureMaterial)
}

impl Material {
    pub fn id(&self) -> MaterialId {
        match self {
            Self::Color(material) => material.id,
            Self::Texture(material) => material.id,
        }
    }

    pub fn bind(&self, pass: &mut wgpu::RenderPass) -> anyhow::Result<()> {
        match self {
            Self::Color(material) => material.bind(pass),
            Self::Texture(material) => material.bind(pass),
        }
    }

    pub fn material_type(&self) -> MaterialType {
        match self {
            Self::Color(_) => MaterialType::Color,
            Self::Texture(_) => MaterialType::Texture
        }
    }
}


pub struct ColorMaterial {
    pub id: MaterialId,
    pub diffuse: RgbaColor,

    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup
}

impl ColorMaterial {
    fn new(
        id: MaterialId,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        diffuse: RgbaColor
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Color Material Buffer"),
            contents: bytes_of(&diffuse),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Color Material Bind Group"),
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


pub struct TextureMaterial {
    pub id: MaterialId,
    pub diffuse: Texture,

    bind_group: wgpu::BindGroup
}

impl TextureMaterial {
    fn new(
        id: MaterialId,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        path: &Path
    ) -> anyhow::Result<Self> {
        let image_bytes = std::fs::read(path)?;
        let diffuse = Texture::from_bytes(device, queue, &image_bytes, "Texture Material Diffuse Texture")?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Material Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse.view)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse.sampler)
                }
            ]
        });

        let ret = Self {
            id,
            diffuse,
            bind_group
        };
        Ok(ret)
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

        let default_color_material = ColorMaterial::new(0, device, &color_bind_group_layout, DEFAULT_COLOR);
        materials.insert(0, Material::Color(default_color_material));

        Self {
            materials,
            next_id,
            color_bind_group_layout,
            texture_bind_group_layout,
        }
    }

    pub fn create_color_material(&mut self, device: &wgpu::Device, color: RgbaColor) -> MaterialId {
        let id = self.next_id;
        self.next_id += 1;
        let material = ColorMaterial::new(id, device, &self.color_bind_group_layout, color);
        self.materials.insert(id, Material::Color(material));

        id
    }

    pub fn create_texture_material_from_path<P: AsRef<Path>>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        diffuse_path: P
    ) -> anyhow::Result<MaterialId> {
        let id = self.next_id;
        self.next_id += 1;

        let material = TextureMaterial::new(
            id,
            device,
            queue,
            &self.texture_bind_group_layout,
            diffuse_path.as_ref()
        )?;

        self.materials.insert(id, Material::Texture(material));

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