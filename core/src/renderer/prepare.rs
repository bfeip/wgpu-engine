use anyhow::Result;
use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::scene::{
    GpuTexture, MaterialGpuResources, PrimitiveType, Scene,
};

use super::Renderer;

impl<'a> Renderer<'a> {
    /// Prepare all GPU resources for a scene before rendering.
    ///
    /// This method ensures all textures, materials, and meshes have their GPU resources
    /// created or updated as needed. It should be called before `render_scene_to_view()`.
    //
    // TODO: This iterates more or less everything in the scene. For performance in the future,
    // we should keep track of the need for these updates in the scene. I.e. mark things as
    // dirty if they need to be reified.
    pub fn prepare_scene(&mut self, scene: &mut Scene) -> Result<()> {
        // 0. Reify any unreified annotations (creates meshes/materials/nodes)
        scene.reify_annotations();

        // 1. Prepare all textures first (materials depend on them)
        for texture in scene.textures.values_mut() {
            if texture.needs_gpu_upload() {
                texture.ensure_gpu_resources(&self.device, &self.queue)?;
            }
        }

        // 2. Prepare all materials
        // We need to collect material IDs first to avoid borrow issues
        let material_ids: Vec<_> = scene.materials.keys().copied().collect();
        for mat_id in material_ids {
            // Check each primitive type
            for prim_type in [
                PrimitiveType::TriangleList,
                PrimitiveType::LineList,
                PrimitiveType::PointList,
            ] {
                let needs_update = scene
                    .materials
                    .get(&mat_id)
                    .map(|m| m.needs_gpu_resources(prim_type) && m.has_primitive_data(prim_type))
                    .unwrap_or(false);

                if needs_update {
                    self.prepare_material_primitive(scene, mat_id, prim_type)?;
                }
            }
        }

        // 3. Prepare all meshes
        for mesh in scene.meshes.values_mut() {
            if mesh.needs_gpu_upload() {
                mesh.ensure_gpu_resources(&self.device);
            }
        }

        // 4. Process environment maps for IBL
        if let Some(env_id) = scene.active_environment_map {
            if let Some(env_map) = scene.environment_maps.get_mut(&env_id) {
                if env_map.needs_generation() {
                    self.ibl_resources
                        .process_environment(&self.device, &self.queue, env_map)?;
                }
            }
        }

        Ok(())
    }

    /// Resolve a texture from the scene, falling back to a default texture if not found.
    fn resolve_texture_or_default<'b>(
        &'b self,
        scene: &'b Scene,
        texture_id: Option<u32>,
        default: &'b GpuTexture,
        name: &str,
    ) -> Result<(&'b wgpu::TextureView, &'b wgpu::Sampler)> {
        if let Some(tex_id) = texture_id {
            let tex = scene
                .textures
                .get(&tex_id)
                .ok_or_else(|| anyhow::anyhow!("{} texture {} not found", name, tex_id))?;
            let gpu = tex.gpu();
            Ok((&gpu.view, &gpu.sampler))
        } else {
            Ok((&default.view, &default.sampler))
        }
    }

    /// Create GPU resources for a color-based material (uniform buffer + bind group).
    fn create_color_material_resources(
        &self,
        color: &crate::common::RgbaColor,
        label: &str,
    ) -> MaterialGpuResources {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(color),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.material_layouts.color,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        MaterialGpuResources {
            bind_group,
            _buffer: Some(buffer),
        }
    }

    /// Prepare GPU resources for a specific material primitive type.
    fn prepare_material_primitive(
        &self,
        scene: &mut Scene,
        material_id: u32,
        primitive_type: PrimitiveType,
    ) -> Result<()> {
        match primitive_type {
            PrimitiveType::TriangleList => self.prepare_triangle_material(scene, material_id),
            PrimitiveType::LineList => self.prepare_line_material(scene, material_id),
            PrimitiveType::PointList => self.prepare_point_material(scene, material_id),
        }
    }

    /// Prepare GPU resources for triangle (face) rendering.
    fn prepare_triangle_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();
        let use_pbr =
            material.normal_texture().is_some() || material.metallic_roughness_texture().is_some();

        if use_pbr {
            self.prepare_pbr_material(scene, material_id)
        } else if material.base_color_texture().is_some() {
            self.prepare_textured_material(scene, material_id)
        } else {
            self.prepare_colored_material(scene, material_id)
        }
    }

    /// Prepare GPU resources for PBR material (normal/metallic-roughness textures).
    fn prepare_pbr_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        let pbr_uniform = material.build_pbr_uniform();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PBR Uniform Buffer"),
                contents: bytes_of(&pbr_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (base_color_view, base_color_sampler) = self.resolve_texture_or_default(
            scene,
            material.base_color_texture(),
            &self.default_textures.white,
            "Base color",
        )?;
        let (normal_view, normal_sampler) = self.resolve_texture_or_default(
            scene,
            material.normal_texture(),
            &self.default_textures.normal,
            "Normal",
        )?;
        let (mr_view, mr_sampler) = self.resolve_texture_or_default(
            scene,
            material.metallic_roughness_texture(),
            &self.default_textures.white,
            "Metallic-roughness",
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR Material Bind Group"),
            layout: &self.material_layouts.pbr,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(base_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(base_color_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(normal_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(mr_sampler),
                },
            ],
        });

        let material = scene.materials.get_mut(&material_id).unwrap();
        material.set_gpu(
            PrimitiveType::TriangleList,
            MaterialGpuResources {
                bind_group,
                _buffer: Some(buffer),
            },
        );
        material.mark_clean(PrimitiveType::TriangleList);
        Ok(())
    }

    /// Prepare GPU resources for texture-only material (base color texture, no PBR).
    fn prepare_textured_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();
        let texture_id = material.base_color_texture().unwrap();

        let texture = scene.textures.get(&texture_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Texture {} not found for material {}",
                texture_id,
                material_id
            )
        })?;
        let gpu_tex = texture.gpu();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Base Color Texture Bind Group"),
            layout: &self.material_layouts.texture,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&gpu_tex.sampler),
                },
            ],
        });

        let material = scene.materials.get_mut(&material_id).unwrap();
        material.set_gpu(
            PrimitiveType::TriangleList,
            MaterialGpuResources {
                bind_group,
                _buffer: None,
            },
        );
        material.mark_clean(PrimitiveType::TriangleList);
        Ok(())
    }

    /// Prepare GPU resources for color-only material (base_color_factor, no textures).
    fn prepare_colored_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();
        let gpu_resources =
            self.create_color_material_resources(&material.base_color_factor(), "Base Color Factor");

        let material = scene.materials.get_mut(&material_id).unwrap();
        material.set_gpu(PrimitiveType::TriangleList, gpu_resources);
        material.mark_clean(PrimitiveType::TriangleList);
        Ok(())
    }

    /// Prepare GPU resources for line rendering.
    fn prepare_line_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        if let Some(color) = material.line_color() {
            let gpu_resources = self.create_color_material_resources(&color, "Line Color");

            let material = scene.materials.get_mut(&material_id).unwrap();
            material.set_gpu(PrimitiveType::LineList, gpu_resources);
            material.mark_clean(PrimitiveType::LineList);
        }
        Ok(())
    }

    /// Prepare GPU resources for point rendering.
    fn prepare_point_material(&self, scene: &mut Scene, material_id: u32) -> Result<()> {
        let material = scene.materials.get(&material_id).unwrap();

        if let Some(color) = material.point_color() {
            let gpu_resources = self.create_color_material_resources(&color, "Point Color");

            let material = scene.materials.get_mut(&material_id).unwrap();
            material.set_gpu(PrimitiveType::PointList, gpu_resources);
            material.mark_clean(PrimitiveType::PointList);
        }
        Ok(())
    }
}
