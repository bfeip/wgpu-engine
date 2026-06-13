use anyhow::Result;
use bytemuck::bytes_of;
use wgpu::util::DeviceExt;

use crate::scene::{
    FaceMaterialId, LineMaterialId, PointMaterialId, PrimitiveType, Scene, TextureId,
};

use super::batching::collect_main_scene_data;
use super::gpu_resources::{LightsArrayUniform, MaterialGpuResources, PbrUniform};
use super::Renderer;

impl Renderer {
    /// Prepare all GPU resources for a scene before rendering.
    ///
    /// This method ensures all textures, materials, and meshes have their GPU resources
    /// created or updated as needed. It should be called before `render_scene_to_view()`.
    //
    // TODO: This iterates more or less everything in the scene. For performance in the future,
    // we should keep track of the need for these updates in the scene. I.e. mark things as
    // dirty if they need to be reified.
    pub fn prepare_scene(&mut self, scene: &mut Scene) -> Result<()> {
        // 1. Prepare all textures first (materials depend on them)
        for texture in scene.textures() {
            self.gpu_resources.ensure_texture(texture, &self.host.gpu().device, &self.host.gpu().queue)?;
        }

        // 2. Prepare all materials, one collection per primitive kind.
        // We collect ids first to avoid borrow issues (the prepare_* helpers take &mut Scene).
        let face_ids: Vec<FaceMaterialId> = scene
            .face_materials()
            .filter(|m| self.gpu_resources.material_needs_upload(m.id.erased(), PrimitiveType::TriangleList, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in face_ids {
            self.prepare_face_material(scene, id)?;
        }

        let line_ids: Vec<LineMaterialId> = scene
            .line_materials()
            .filter(|m| self.gpu_resources.material_needs_upload(m.id.erased(), PrimitiveType::LineList, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in line_ids {
            self.prepare_line_material(scene, id)?;
        }

        let point_ids: Vec<PointMaterialId> = scene
            .point_materials()
            .filter(|m| self.gpu_resources.material_needs_upload(m.id.erased(), PrimitiveType::PointList, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in point_ids {
            self.prepare_point_material(scene, id)?;
        }

        // 3. Prepare all meshes
        // TODO: For wireframe mode, generate line index buffers from triangle primitives
        // using MeshPrimitive::to_line_list(). This could be done here by checking a
        // wireframe flag and ensuring meshes have line primitives derived from their
        // triangle data before uploading to the GPU.
        for mesh in scene.meshes() {
            self.gpu_resources.ensure_mesh(mesh, &self.host.gpu().device);
        }

        // 4. Prepare lights
        let node_gen = scene.node_generation();
        if self.lights.synced_generation != node_gen {
            let frame_data = collect_main_scene_data(scene, &super::batching::sub_view_root_set(scene));
            let lights_uniform = LightsArrayUniform::from_resolved_lights(&frame_data.lights);
            self.host.gpu().queue
                .write_buffer(&self.lights.buffer, 0, bytes_of(&lights_uniform));
            self.lights.synced_generation = node_gen;
        }

        // 5. Process environment maps for IBL
        if let Some(env_id) = scene.active_environment_map()
            && let Some(env_map) = scene.get_environment_map(env_id)
        {
            self.ibl_resources
                .process_environment(&self.host.gpu().device, &self.host.gpu().queue, env_map)?;
        }

        Ok(())
    }

    /// Resolve a texture from the GPU resource manager, falling back to a default texture if not found.
    fn resolve_texture_or_default<'b>(
        &'b self,
        texture_id: Option<TextureId>,
        default: &'b super::gpu_resources::GpuTexture,
        name: &str,
    ) -> Result<(&'b wgpu::TextureView, &'b wgpu::Sampler)> {
        if let Some(tex_id) = texture_id {
            let gpu = self.gpu_resources
                .get_texture(tex_id)
                .ok_or_else(|| anyhow::anyhow!("{} texture {} not found in GPU resources", name, tex_id))?;
            Ok((&gpu.view, &gpu.sampler))
        } else {
            Ok((&default.view, &default.sampler))
        }
    }

    /// Create GPU resources for a color-based material (uniform buffer + bind group).
    fn create_color_material_resources(
        &self,
        color: &crate::scene::common::RgbaColor,
        label: &str,
    ) -> MaterialGpuResources {
        let buffer = self
            .host.gpu()
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_of(color),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.host.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.layouts.color,
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

    /// Prepare GPU resources for face (triangle) rendering.
    fn prepare_face_material(&mut self, scene: &mut Scene, material_id: FaceMaterialId) -> Result<()> {
        let material = scene.get_face_material(material_id).unwrap();
        let has_lighting = !material.flags().contains(crate::scene::MaterialFlags::DO_NOT_LIGHT);

        if has_lighting {
            self.prepare_pbr_material(scene, material_id)
        } else {
            self.prepare_colored_material(scene, material_id)
        }
    }

    /// Prepare GPU resources for PBR material (normal/metallic-roughness textures).
    fn prepare_pbr_material(&mut self, scene: &mut Scene, material_id: FaceMaterialId) -> Result<()> {
        let material = scene.get_face_material(material_id).unwrap();

        let pbr_uniform = PbrUniform::from_face_material(material);
        let buffer = self
            .host.gpu()
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PBR Uniform Buffer"),
                contents: bytes_of(&pbr_uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let (base_color_view, base_color_sampler) = self.resolve_texture_or_default(
            material.base_color_texture(),
            &self.fallback_textures.white,
            "Base color",
        )?;
        let (normal_view, normal_sampler) = self.resolve_texture_or_default(
            material.normal_texture(),
            &self.fallback_textures.default_normal,
            "Normal",
        )?;
        let (mr_view, mr_sampler) = self.resolve_texture_or_default(
            material.metallic_roughness_texture(),
            &self.fallback_textures.white,
            "Metallic-roughness",
        )?;

        let bind_group = self.host.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR Material Bind Group"),
            layout: &self.layouts.pbr,
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

        let generation = scene.get_face_material(material_id).unwrap().generation();
        self.gpu_resources.set_material(
            material_id.erased(),
            PrimitiveType::TriangleList,
            MaterialGpuResources {
                bind_group,
                _buffer: Some(buffer),
            },
            generation,
        );
        Ok(())
    }

    /// Prepare GPU resources for color-only material (base_color_factor, no textures).
    fn prepare_colored_material(&mut self, scene: &mut Scene, material_id: FaceMaterialId) -> Result<()> {
        let material = scene.get_face_material(material_id).unwrap();
        let gpu_resources =
            self.create_color_material_resources(&material.base_color_factor(), "Base Color Factor");
        let generation = material.generation();

        self.gpu_resources.set_material(material_id.erased(), PrimitiveType::TriangleList, gpu_resources, generation);
        Ok(())
    }

    /// Prepare GPU resources for line rendering.
    fn prepare_line_material(&mut self, scene: &mut Scene, material_id: LineMaterialId) -> Result<()> {
        let material = scene.get_line_material(material_id).unwrap();
        let gpu_resources = self.create_color_material_resources(&material.color(), "Line Color");
        let generation = material.generation();

        self.gpu_resources.set_material(material_id.erased(), PrimitiveType::LineList, gpu_resources, generation);
        Ok(())
    }

    /// Prepare GPU resources for point rendering.
    fn prepare_point_material(&mut self, scene: &mut Scene, material_id: PointMaterialId) -> Result<()> {
        let material = scene.get_point_material(material_id).unwrap();
        let gpu_resources = self.create_color_material_resources(&material.color(), "Point Color");
        let generation = material.generation();

        self.gpu_resources.set_material(material_id.erased(), PrimitiveType::PointList, gpu_resources, generation);
        Ok(())
    }
}
