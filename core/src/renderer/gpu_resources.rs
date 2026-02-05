//! GPU resource management for scene objects.
//!
//! This module centralizes GPU resource tracking for meshes, textures, and materials.
//! It tracks which resources have been uploaded and their sync state using generation numbers.

use std::collections::HashMap;

use anyhow::Result;
use image::GenericImageView;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::scene::{InstanceRaw, InstanceTransform, MaterialId, Mesh, MeshId, MeshIndex, PrimitiveType, Texture, TextureId};

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
    // Convert InstanceTransforms to InstanceRaw for the GPU
    let instance_raws: Vec<InstanceRaw> = instance_transforms
        .iter()
        .map(|inst_transform| InstanceRaw {
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
