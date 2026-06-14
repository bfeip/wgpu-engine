//! Material subsystem: everything needed to turn scene materials into GPU
//! pipelines and bind groups.
//!
//! This is the single owner of the material/shader machinery: the [`MaterialPipelineCache`]
//! (which owns the WESL [`ShaderGenerator`]), the per-material bind-group caches,
//! the material bind group layouts, and the fallback textures bound when a
//! material has no texture of its own. Passes reach it through
//! [`SceneFrame::materials`](super::pass_context::SceneFrame).

use anyhow::Result;
use bytemuck::bytes_of;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::render_core::GenCache;
use crate::scene::{
    FaceMaterialId, LineMaterialId, MaterialFlags, PointMaterialId, Scene, TextureId,
    common::RgbaColor,
};
use crate::shaders::ShaderGenerator;

use super::batching::BatchMaterial;
use super::gpu_resources::{
    BindGroupLayouts, FallbackTextures, GpuTexture, MaterialGpuResources, MaterialPipelineLayouts,
    PbrUniform, PipelineCacheKey,
};
use super::pipeline::MaterialPipelineCache;

/// Owns material → GPU translation: pipeline cache, shader generator, material
/// bind-group caches, material bind group layouts, and fallback textures.
///
/// Each material kind (face/line/point) has its own cache keyed by its typed id.
/// Callers reach the right cache through the typed accessors.
pub(crate) struct MaterialSystem {
    pipelines: MaterialPipelineCache,
    // TODO: These layouts are also owned by [`BindGroupLayouts`]. Probably nbd but
    // we should check to see if one or the other should drop its ownership. 
    color_layout: wgpu::BindGroupLayout,
    pbr_layout: wgpu::BindGroupLayout,
    // TODO: Fallback textures shouldn't exist. The shaders should handle no texture
    // more gracefully.
    fallback: FallbackTextures,
    face: GenCache<FaceMaterialId, MaterialGpuResources>,
    line: GenCache<LineMaterialId, MaterialGpuResources>,
    point: GenCache<PointMaterialId, MaterialGpuResources>,
}

impl MaterialSystem {
    pub fn new(
        device: &wgpu::Device,
        layouts: &BindGroupLayouts,
        fallback: FallbackTextures,
        shader_generator: ShaderGenerator,
        sample_count: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline_layouts = MaterialPipelineLayouts::new(device, layouts);
        let pipelines =
            MaterialPipelineCache::new(pipeline_layouts, shader_generator, sample_count, surface_format);
        Self {
            pipelines,
            color_layout: layouts.color.clone(),
            pbr_layout: layouts.pbr.clone(),
            fallback,
            face: GenCache::new(),
            line: GenCache::new(),
            point: GenCache::new(),
        }
    }

    /// Mutable access to the shader generator, for passes/workflows that compile
    /// technique-specific shaders (outline, silhouette, flat-color).
    // TODO: This feels like ownership confusion. Why does the material system own the
    // [`ShaderGenerator`] if other passes also need it.
    pub fn shader_generator_mut(&mut self) -> &mut ShaderGenerator {
        self.pipelines.shader_generator_mut()
    }

    /// Get (or create and cache) the material-variant render pipeline for `key`.
    pub fn pipeline(
        &mut self,
        device: &wgpu::Device,
        key: PipelineCacheKey,
    ) -> &wgpu::RenderPipeline {
        self.pipelines.get_or_create(device, key)
    }

    /// The cached bind group for a face material, if uploaded.
    pub fn face_material(&self, id: FaceMaterialId) -> Option<&MaterialGpuResources> {
        self.face.get(id)
    }

    /// The cached bind group for a line material, if uploaded.
    pub fn line_material(&self, id: LineMaterialId) -> Option<&MaterialGpuResources> {
        self.line.get(id)
    }

    /// The cached bind group for a point material, if uploaded.
    pub fn point_material(&self, id: PointMaterialId) -> Option<&MaterialGpuResources> {
        self.point.get(id)
    }

    /// The cached bind group for a batch's material, if uploaded.
    ///
    /// Dispatches on the typed [`BatchMaterial`] to the matching per-kind getter —
    /// the variant carries its own id kind, so there is no erased lookup.
    pub fn bind_group(&self, material: BatchMaterial) -> Option<&MaterialGpuResources> {
        match material {
            BatchMaterial::Face(id) => self.face_material(id),
            BatchMaterial::Line(id) => self.line_material(id),
            BatchMaterial::Point(id) => self.point_material(id),
        }
    }

    /// Drop all cached material bind groups (e.g. when the scene is replaced).
    pub fn clear(&mut self) {
        self.face.clear();
        self.line.clear();
        self.point.clear();
    }

    /// Upload any face/line/point materials whose generation has changed.
    ///
    /// `textures` supplies already-uploaded texture GPU resources for PBR
    /// materials; the renderer prepares textures before calling this.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        scene: &Scene,
        textures: &GenCache<TextureId, GpuTexture>,
    ) -> Result<()> {
        let face_ids: Vec<FaceMaterialId> = scene
            .face_materials()
            .filter(|m| self.face.needs_upload(m.id, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in face_ids {
            let material = scene.get_face_material(id).unwrap();
            let generation = material.generation();
            let resources = if material.flags().contains(MaterialFlags::DO_NOT_LIGHT) {
                self.create_color_material(device, &material.base_color_factor(), "Base Color Factor")
            } else {
                self.create_pbr_material(device, scene, id, textures)?
            };
            self.face.insert(id, resources, generation);
        }

        let line_ids: Vec<LineMaterialId> = scene
            .line_materials()
            .filter(|m| self.line.needs_upload(m.id, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in line_ids {
            let material = scene.get_line_material(id).unwrap();
            let resources = self.create_color_material(device, &material.color(), "Line Color");
            self.line.insert(id, resources, material.generation());
        }

        let point_ids: Vec<PointMaterialId> = scene
            .point_materials()
            .filter(|m| self.point.needs_upload(m.id, m.generation()))
            .map(|m| m.id)
            .collect();
        for id in point_ids {
            let material = scene.get_point_material(id).unwrap();
            let resources = self.create_color_material(device, &material.color(), "Point Color");
            self.point.insert(id, resources, material.generation());
        }

        Ok(())
    }

    /// Build the PBR material bind group (uniform + base color / normal /
    /// metallic-roughness textures, falling back to defaults where absent).
    fn create_pbr_material(
        &self,
        device: &wgpu::Device,
        scene: &Scene,
        material_id: FaceMaterialId,
        textures: &GenCache<TextureId, GpuTexture>,
    ) -> Result<MaterialGpuResources> {
        let material = scene.get_face_material(material_id).unwrap();

        let pbr_uniform = PbrUniform::from_face_material(material);
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("PBR Uniform Buffer"),
            contents: bytes_of(&pbr_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let (base_color_view, base_color_sampler) =
            self.resolve_texture(textures, material.base_color_texture(), &self.fallback.white, "Base color")?;
        let (normal_view, normal_sampler) =
            self.resolve_texture(textures, material.normal_texture(), &self.fallback.default_normal, "Normal")?;
        let (mr_view, mr_sampler) = self.resolve_texture(
            textures,
            material.metallic_roughness_texture(),
            &self.fallback.white,
            "Metallic-roughness",
        )?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR Material Bind Group"),
            layout: &self.pbr_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(base_color_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(base_color_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(normal_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(normal_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(mr_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(mr_sampler) },
            ],
        });

        Ok(MaterialGpuResources { bind_group, _buffer: Some(buffer) })
    }

    /// Build a flat-color material bind group (a single `vec4` uniform).
    fn create_color_material(
        &self,
        device: &wgpu::Device,
        color: &RgbaColor,
        label: &str,
    ) -> MaterialGpuResources {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytes_of(color),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.color_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        MaterialGpuResources { bind_group, _buffer: Some(buffer) }
    }

    /// Resolve an optional texture id to its GPU view+sampler, falling back to
    /// `default` when the material does not reference a texture.
    fn resolve_texture<'b>(
        &self,
        textures: &'b GenCache<TextureId, GpuTexture>,
        texture_id: Option<TextureId>,
        default: &'b GpuTexture,
        name: &str,
    ) -> Result<(&'b wgpu::TextureView, &'b wgpu::Sampler)> {
        if let Some(tex_id) = texture_id {
            let gpu = textures
                .get(tex_id)
                .ok_or_else(|| anyhow::anyhow!("{} texture {} not found in GPU resources", name, tex_id))?;
            Ok((&gpu.view, &gpu.sampler))
        } else {
            Ok((&default.view, &default.sampler))
        }
    }
}
