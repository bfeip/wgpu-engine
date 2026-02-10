//! GPU resource management for scene objects.
//!
//! This module centralizes GPU resource tracking for meshes, textures, and materials.
//! It tracks which resources have been uploaded and their sync state using generation numbers.

use std::collections::HashMap;

use anyhow::Result;
use image::GenericImageView;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::scene::{
    InstanceTransform, Light, LightType, Material, MaterialId, Mesh, MeshId,
    MeshIndex, PrimitiveType, Texture, TextureId, MAX_LIGHTS,
};

/// GPU-ready instance data for instanced rendering.
///
/// Contains a world transform matrix and normal matrix packed for the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstance {
    pub transform: [[f32; 4]; 4],
    pub normal_mat: [[f32; 3]; 3],
}

/// GPU-compatible representation of a single light for shader uniforms.
///
/// This struct is laid out to match WGSL uniform buffer alignment requirements.
/// vec3<f32> types require 16-byte alignment in WGSL, so scalar fields are
/// grouped at the start to pack efficiently.
///
/// # Memory Layout (64 bytes total)
///
/// | Offset | Size | Field          | Notes                              |
/// |--------|------|----------------|------------------------------------|
/// | 0      | 4    | light_type     | 0=Point, 1=Directional, 2=Spot     |
/// | 4      | 4    | range          | 0 = infinite range                 |
/// | 8      | 4    | inner_cone_cos | Spot: cos(inner angle)             |
/// | 12     | 4    | outer_cone_cos | Spot: cos(outer angle)             |
/// | 16     | 12   | position       | Point/Spot: world position         |
/// | 28     | 4    | intensity      | Light intensity multiplier         |
/// | 32     | 12   | direction      | Directional/Spot: light direction  |
/// | 44     | 4    | _padding1      | Alignment padding                  |
/// | 48     | 12   | color          | RGB color                          |
/// | 60     | 4    | _padding2      | Alignment padding                  |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    light_type: u32,
    range: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    position: [f32; 3],
    intensity: f32,
    direction: [f32; 3],
    _padding1: f32,
    color: [f32; 3],
    _padding2: f32,
}

impl LightUniform {
    /// Creates a `LightUniform` from a scene `Light`.
    pub fn from_light(light: &Light) -> Self {
        match light {
            Light::Point {
                position,
                color,
                intensity,
                range,
            } => LightUniform {
                light_type: LightType::Point as u32,
                range: *range,
                inner_cone_cos: 0.0,
                outer_cone_cos: 0.0,
                position: (*position).into(),
                intensity: *intensity,
                direction: [0.0, 0.0, 0.0],
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
            Light::Directional {
                direction,
                color,
                intensity,
            } => LightUniform {
                light_type: LightType::Directional as u32,
                range: 0.0,
                inner_cone_cos: 0.0,
                outer_cone_cos: 0.0,
                position: [0.0, 0.0, 0.0],
                intensity: *intensity,
                direction: (*direction).into(),
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
            Light::Spot {
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone_angle,
                outer_cone_angle,
            } => LightUniform {
                light_type: LightType::Spot as u32,
                range: *range,
                inner_cone_cos: inner_cone_angle.cos(),
                outer_cone_cos: outer_cone_angle.cos(),
                position: (*position).into(),
                intensity: *intensity,
                direction: (*direction).into(),
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
        }
    }
}

/// GPU-compatible array of lights with count.
///
/// # Memory Layout (528 bytes total)
///
/// | Offset | Size     | Field       |
/// |--------|----------|-------------|
/// | 0      | 4        | light_count |
/// | 4      | 12       | _padding    |
/// | 16     | 64 * 8   | lights      |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightsArrayUniform {
    pub light_count: u32,
    _padding: [u32; 3],
    pub lights: [LightUniform; MAX_LIGHTS],
}

impl LightsArrayUniform {
    /// Creates a lights array uniform from a slice of lights.
    ///
    /// Only the first `MAX_LIGHTS` lights will be used.
    pub fn from_lights(lights: &[Light]) -> Self {
        let mut uniform = Self {
            light_count: lights.len().min(MAX_LIGHTS) as u32,
            _padding: [0; 3],
            lights: [bytemuck::Zeroable::zeroed(); MAX_LIGHTS],
        };
        for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
            uniform.lights[i] = LightUniform::from_light(light);
        }
        uniform
    }
}

/// PBR material parameters for GPU uniform buffer.
///
/// This struct is sent to the shader and contains all scalar factors
/// plus flags indicating which textures are present.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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

    /// Creates a `PbrUniform` from a scene `Material`.
    pub fn from_material(material: &Material) -> Self {
        let mut texture_flags = 0u32;
        if material.base_color_texture().is_some() {
            texture_flags |= Self::FLAG_HAS_BASE_COLOR_TEXTURE;
        }
        if material.normal_texture().is_some() {
            texture_flags |= Self::FLAG_HAS_NORMAL_TEXTURE;
        }
        if material.metallic_roughness_texture().is_some() {
            texture_flags |= Self::FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE;
        }

        let base_color = material.base_color_factor();
        PbrUniform {
            base_color_factor: [base_color.r, base_color.g, base_color.b, base_color.a],
            metallic_factor: material.metallic_factor(),
            roughness_factor: material.roughness_factor(),
            normal_scale: material.normal_scale(),
            texture_flags,
        }
    }
}

/// GPU resources for a mesh (vertex and index buffers).
pub(crate) struct MeshGpuResources {
    pub vertex_buffer: wgpu::Buffer,
    pub triangle_index_buffer: wgpu::Buffer,
    pub line_index_buffer: wgpu::Buffer,
    pub point_index_buffer: wgpu::Buffer,
}

/// GPU state for a mesh, including resources and sync tracking.
pub(crate) struct MeshGpuState {
    pub resources: MeshGpuResources,
    pub synced_generation: u64,
}

/// GPU resources for a texture.
pub(crate) struct GpuTexture {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

/// GPU state for a texture, including resources and sync tracking.
pub(crate) struct TextureGpuState {
    pub gpu_texture: GpuTexture,
    pub synced_generation: u64,
}

/// GPU resources for a material primitive type (bind group).
pub(crate) struct MaterialGpuResources {
    pub bind_group: wgpu::BindGroup,
    pub _buffer: Option<wgpu::Buffer>,
}

/// GPU state for a material, with per-primitive-type tracking.
pub(crate) struct MaterialGpuState {
    pub face: Option<MaterialGpuResources>,
    pub line: Option<MaterialGpuResources>,
    pub point: Option<MaterialGpuResources>,
    pub face_synced_generation: u64,
    pub line_synced_generation: u64,
    pub point_synced_generation: u64,
}

impl MaterialGpuState {
    pub fn new() -> Self {
        Self {
            face: None,
            line: None,
            point: None,
            face_synced_generation: 0,
            line_synced_generation: 0,
            point_synced_generation: 0,
        }
    }

    pub fn get(&self, primitive_type: PrimitiveType) -> Option<&MaterialGpuResources> {
        match primitive_type {
            PrimitiveType::TriangleList => self.face.as_ref(),
            PrimitiveType::LineList => self.line.as_ref(),
            PrimitiveType::PointList => self.point.as_ref(),
        }
    }

    pub fn synced_generation(&self, primitive_type: PrimitiveType) -> u64 {
        match primitive_type {
            PrimitiveType::TriangleList => self.face_synced_generation,
            PrimitiveType::LineList => self.line_synced_generation,
            PrimitiveType::PointList => self.point_synced_generation,
        }
    }

    pub fn set_synced_generation(&mut self, primitive_type: PrimitiveType, generation: u64) {
        match primitive_type {
            PrimitiveType::TriangleList => self.face_synced_generation = generation,
            PrimitiveType::LineList => self.line_synced_generation = generation,
            PrimitiveType::PointList => self.point_synced_generation = generation,
        }
    }

    pub fn set(&mut self, primitive_type: PrimitiveType, resources: MaterialGpuResources) {
        match primitive_type {
            PrimitiveType::TriangleList => self.face = Some(resources),
            PrimitiveType::LineList => self.line = Some(resources),
            PrimitiveType::PointList => self.point = Some(resources),
        }
    }
}

/// Centralized manager for GPU resources associated with scene objects.
///
/// This tracks all GPU resources (buffers, textures, bind groups) and their
/// synchronization state using generation numbers. The renderer uses this
/// to know when to upload new data to the GPU.
pub(crate) struct GpuResourceManager {
    pub meshes: HashMap<MeshId, MeshGpuState>,
    pub textures: HashMap<TextureId, TextureGpuState>,
    pub materials: HashMap<MaterialId, MaterialGpuState>,
}

impl GpuResourceManager {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            textures: HashMap::new(),
            materials: HashMap::new(),
        }
    }

    /// Check if a mesh needs GPU resource upload.
    pub fn mesh_needs_upload(&self, mesh_id: MeshId, current_generation: u64) -> bool {
        match self.meshes.get(&mesh_id) {
            None => true,
            Some(state) => state.synced_generation != current_generation,
        }
    }

    /// Check if a texture needs GPU resource upload.
    pub fn texture_needs_upload(&self, texture_id: TextureId, current_generation: u64) -> bool {
        match self.textures.get(&texture_id) {
            None => true,
            Some(state) => state.synced_generation != current_generation,
        }
    }

    /// Check if a material primitive needs GPU resource upload.
    pub fn material_needs_upload(
        &self,
        material_id: MaterialId,
        primitive_type: PrimitiveType,
        current_generation: u64,
    ) -> bool {
        match self.materials.get(&material_id) {
            None => true,
            Some(state) => {
                state.get(primitive_type).is_none()
                    || state.synced_generation(primitive_type) != current_generation
            }
        }
    }

    /// Get mesh GPU resources.
    pub fn get_mesh(&self, mesh_id: MeshId) -> Option<&MeshGpuResources> {
        self.meshes.get(&mesh_id).map(|s| &s.resources)
    }

    /// Get texture GPU resources.
    pub fn get_texture(&self, texture_id: TextureId) -> Option<&GpuTexture> {
        self.textures.get(&texture_id).map(|s| &s.gpu_texture)
    }

    /// Get material GPU resources for a primitive type.
    pub fn get_material(
        &self,
        material_id: MaterialId,
        primitive_type: PrimitiveType,
    ) -> Option<&MaterialGpuResources> {
        self.materials
            .get(&material_id)
            .and_then(|s| s.get(primitive_type))
    }

    /// Store mesh GPU resources.
    pub fn set_mesh(&mut self, mesh_id: MeshId, resources: MeshGpuResources, generation: u64) {
        self.meshes.insert(
            mesh_id,
            MeshGpuState {
                resources,
                synced_generation: generation,
            },
        );
    }

    /// Store texture GPU resources.
    pub fn set_texture(
        &mut self,
        texture_id: TextureId,
        gpu_texture: GpuTexture,
        generation: u64,
    ) {
        self.textures.insert(
            texture_id,
            TextureGpuState {
                gpu_texture,
                synced_generation: generation,
            },
        );
    }

    /// Store material GPU resources for a primitive type.
    pub fn set_material(
        &mut self,
        material_id: MaterialId,
        primitive_type: PrimitiveType,
        resources: MaterialGpuResources,
        generation: u64,
    ) {
        let state = self
            .materials
            .entry(material_id)
            .or_insert_with(MaterialGpuState::new);
        state.set(primitive_type, resources);
        state.set_synced_generation(primitive_type, generation);
    }

    /// Ensure mesh GPU resources are created and up-to-date.
    ///
    /// If the mesh's generation doesn't match the stored synced_generation,
    /// new GPU resources are created.
    pub fn ensure_mesh(&mut self, mesh: &Mesh, device: &wgpu::Device) {
        let mesh_id = mesh.id;
        let generation = mesh.generation();

        if !self.mesh_needs_upload(mesh_id, generation) {
            return;
        }

        let resources = create_mesh_gpu_resources(mesh, device);
        self.set_mesh(mesh_id, resources, generation);
    }

    /// Ensure texture GPU resources are created and up-to-date.
    ///
    /// If the texture's generation doesn't match the stored synced_generation,
    /// new GPU resources are created.
    pub fn ensure_texture(
        &mut self,
        texture: &mut Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        let texture_id = texture.id();
        let generation = texture.generation();

        if !self.texture_needs_upload(texture_id, generation) {
            return Ok(());
        }

        let gpu_texture = create_texture_gpu_resources(texture, device, queue)?;
        self.set_texture(texture_id, gpu_texture, generation);
        Ok(())
    }
}

/// Create GPU resources for a mesh.
fn create_mesh_gpu_resources(mesh: &Mesh, device: &wgpu::Device) -> MeshGpuResources {
    let vertices = mesh.vertices();
    let primitives = mesh.primitives();

    // Create vertex buffer
    let vertex_buffer = if vertices.is_empty() {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Vertex Buffer"),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        })
    } else {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    };

    // Helper to create index buffer for a specific primitive type
    let create_index_buffer = |prim_type: PrimitiveType, label: &str| -> wgpu::Buffer {
        let indices: Vec<MeshIndex> = primitives
            .iter()
            .filter(|p| p.primitive_type == prim_type)
            .flat_map(|p| p.indices.iter().copied())
            .collect();

        if indices.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 0,
                usage: wgpu::BufferUsages::INDEX,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            })
        }
    };

    let triangle_index_buffer = create_index_buffer(PrimitiveType::TriangleList, "Triangle Index Buffer");
    let line_index_buffer = create_index_buffer(PrimitiveType::LineList, "Line Index Buffer");
    let point_index_buffer = create_index_buffer(PrimitiveType::PointList, "Point Index Buffer");

    MeshGpuResources {
        vertex_buffer,
        triangle_index_buffer,
        line_index_buffer,
        point_index_buffer,
    }
}

/// Create GPU resources for a texture.
fn create_texture_gpu_resources(
    texture: &mut Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<GpuTexture> {
    let img = texture.get_image()?;
    let rgba = img.to_rgba8();
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let wgpu_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture: &wgpu_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        size,
    );

    let view = wgpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    Ok(GpuTexture {
        _texture: wgpu_texture,
        view,
        sampler,
    })
}

/// Draw mesh instances using the provided GPU resources.
///
/// This function creates an instance buffer and issues a draw call.
pub(crate) fn draw_mesh_instances(
    device: &wgpu::Device,
    render_pass: &mut wgpu::RenderPass,
    gpu_resources: &MeshGpuResources,
    primitive_type: PrimitiveType,
    instance_transforms: &[InstanceTransform],
    index_count: u32,
) {
    // Convert InstanceTransforms to GpuInstance for the GPU
    let instance_raws: Vec<GpuInstance> = instance_transforms
        .iter()
        .map(|inst_transform| GpuInstance {
            transform: inst_transform.world_transform.into(),
            normal_mat: inst_transform.normal_matrix.into(),
        })
        .collect();

    // Create instance buffer
    // TODO: Consider caching/reusing instance buffers for better performance
    let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Instance Buffer"),
        contents: bytemuck::cast_slice(&instance_raws),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Select the appropriate index buffer based on primitive type
    let index_buffer = match primitive_type {
        PrimitiveType::TriangleList => &gpu_resources.triangle_index_buffer,
        PrimitiveType::LineList => &gpu_resources.line_index_buffer,
        PrimitiveType::PointList => &gpu_resources.point_index_buffer,
    };

    // Skip drawing if there are no indices for this primitive type
    if index_count == 0 {
        return;
    }

    // Draw
    let n_instances = instance_transforms.len() as u32;
    render_pass.set_vertex_buffer(0, gpu_resources.vertex_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    render_pass.draw_indexed(0..index_count, 0, 0..n_instances);
}

// ========== Helper functions for creating internal GPU textures ==========

/// Depth-stencil texture format used for depth and stencil buffers.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Single-channel format used for mask textures (selection masks, stencil masks, etc.).
pub const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Create a 1x1 solid color texture for use as a default texture.
///
/// This is used for materials when a specific texture is not provided.
pub(crate) fn create_solid_color_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color: [u8; 4],
    label: &str,
) -> GpuTexture {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            aspect: wgpu::TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &color,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    GpuTexture { _texture: texture, view, sampler }
}

/// Create a depth texture for use as a depth buffer.
pub(crate) fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    label: &str,
) -> GpuTexture {
    let size = wgpu::Extent3d {
        width: config.width.max(1),
        height: config.height.max(1),
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let texture = device.create_texture(&desc);

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });

    GpuTexture { _texture: texture, view, sampler }
}

/// Create a single-channel mask texture at the given dimensions.
pub(crate) fn create_mask_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MASK_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::common::{RgbaColor, EPSILON};
    use cgmath::Vector3;

    #[test]
    fn test_light_uniform_from_point_light() {
        let position = Vector3::new(1.0, 2.0, 3.0);
        let color = RgbaColor { r: 0.5, g: 0.6, b: 0.7, a: 1.0 };
        let light = Light::point(position, color, 2.0);

        let uniform = LightUniform::from_light(&light);

        assert!((uniform.position[0] - 1.0).abs() < EPSILON);
        assert!((uniform.position[1] - 2.0).abs() < EPSILON);
        assert!((uniform.position[2] - 3.0).abs() < EPSILON);
        assert_eq!(uniform.light_type, LightType::Point as u32);
        assert!((uniform.color[0] - 0.5).abs() < EPSILON);
        assert!((uniform.color[1] - 0.6).abs() < EPSILON);
        assert!((uniform.color[2] - 0.7).abs() < EPSILON);
        assert!((uniform.intensity - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_directional_light() {
        let direction = Vector3::new(0.0, -1.0, 0.0);
        let color = RgbaColor { r: 1.0, g: 1.0, b: 0.9, a: 1.0 };
        let light = Light::directional(direction, color, 1.5);

        let uniform = LightUniform::from_light(&light);

        assert_eq!(uniform.light_type, LightType::Directional as u32);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.intensity - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_spotlight() {
        let position = Vector3::new(0.0, 5.0, 0.0);
        let direction = Vector3::new(0.0, -1.0, 0.0);
        let color = RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
        let inner_angle = std::f32::consts::PI / 6.0;
        let outer_angle = std::f32::consts::PI / 4.0;
        let light = Light::spot(position, direction, color, 3.0, inner_angle, outer_angle);

        let uniform = LightUniform::from_light(&light);

        assert_eq!(uniform.light_type, LightType::Spot as u32);
        assert!((uniform.position[1] - 5.0).abs() < EPSILON);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.inner_cone_cos - inner_angle.cos()).abs() < EPSILON);
        assert!((uniform.outer_cone_cos - outer_angle.cos()).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_layout() {
        assert_eq!(std::mem::size_of::<LightUniform>(), 64);
        assert_eq!(std::mem::size_of::<LightsArrayUniform>(), 528);
    }

    #[test]
    fn test_lights_array_uniform_from_lights() {
        let lights = vec![
            Light::point(
                Vector3::new(1.0, 0.0, 0.0),
                RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 },
                1.0,
            ),
            Light::directional(
                Vector3::new(0.0, -1.0, 0.0),
                RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                1.0,
            ),
        ];

        let uniform = LightsArrayUniform::from_lights(&lights);

        assert_eq!(uniform.light_count, 2);
        assert_eq!(uniform.lights[0].light_type, LightType::Point as u32);
        assert_eq!(uniform.lights[1].light_type, LightType::Directional as u32);
    }
}
