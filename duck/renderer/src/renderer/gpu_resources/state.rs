use anyhow::Result;
use image::{imageops::FilterType, GenericImageView};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::render_core::GpuTexture;
use crate::renderer::batching::InstanceTransform;
use crate::scene::{Mesh, MeshIndex, PrimitiveType, Texture};

use super::uniforms::GpuInstance;

/// GPU resources for a mesh (vertex and index buffers).
pub(crate) struct MeshGpuResources {
    pub vertex_buffer: wgpu::Buffer,
    pub triangle_index_buffer: wgpu::Buffer,
    pub line_index_buffer: wgpu::Buffer,
    pub point_index_buffer: wgpu::Buffer,
}

/// GPU resources for a material primitive type (bind group).
pub(crate) struct MaterialGpuResources {
    pub bind_group: wgpu::BindGroup,
    pub _buffer: Option<wgpu::Buffer>,
}

/// A writable flat-color GPU resource: a `vec4<f32>` uniform buffer plus its
/// bind group against the shared color material layout (group 2 /
/// `material_color.wesl`).
///
/// Wraps the buffer+bind-group pair otherwise re-created inline wherever a flat
/// color is pushed to the GPU. The buffer is `COPY_DST` so [`write`](Self::write)
/// can update the color per frame.
pub(crate) struct ColorResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl ColorResources {
    pub fn new(
        device: &wgpu::Device,
        color_bgl: &wgpu::BindGroupLayout,
        color: crate::scene::common::RgbaColor,
        label: &str,
    ) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(&color),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: color_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        Self { buffer, bind_group }
    }

    pub fn write(&self, queue: &wgpu::Queue, color: crate::scene::common::RgbaColor) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&color));
    }
}


/// Create GPU resources for a mesh.
pub(crate) fn create_mesh_gpu_resources(mesh: &Mesh, device: &wgpu::Device) -> MeshGpuResources {
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
pub(crate) fn create_texture_gpu_resources(
    texture: &Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<GpuTexture> {
    let texture_id = texture.id();
    let img = texture.get_image()?;
    let dimensions = img.dimensions();

    // Resize if texture exceeds device limits (e.g. WebGL max 2048px)
    let max_dim = device.limits().max_texture_dimension_2d;
    let img = if dimensions.0 > max_dim || dimensions.1 > max_dim {
        log::info!(
            "Texture {} ({}x{}) exceeds max dimension {}; resizing",
            texture_id,
            dimensions.0,
            dimensions.1,
            max_dim,
        );
        std::borrow::Cow::Owned(img.resize(max_dim, max_dim, FilterType::Triangle))
    } else {
        std::borrow::Cow::Borrowed(img)
    };

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
        texture: wgpu_texture,
        view,
        sampler,
    })
}

/// Draw a single sub-range of a mesh's index buffer for one instance.
///
/// Used to render individual faces or edges (e.g. for highlight outlines).
/// `first_index` and `index_count` are raw index counts (not triangle/segment counts).
pub(crate) fn draw_mesh_subgeom(
    device: &wgpu::Device,
    render_pass: &mut wgpu::RenderPass,
    gpu_resources: &MeshGpuResources,
    primitive_type: PrimitiveType,
    instance_transform: &InstanceTransform,
    first_index: u32,
    index_count: u32,
) {
    if index_count == 0 {
        return;
    }

    let instance_raw = GpuInstance {
        transform: instance_transform.world_transform.into(),
        normal_mat: instance_transform.normal_matrix.into(),
    };

    let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("SubGeom Instance Buffer"),
        contents: bytemuck::cast_slice(&[instance_raw]),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = match primitive_type {
        PrimitiveType::TriangleList => &gpu_resources.triangle_index_buffer,
        PrimitiveType::LineList => &gpu_resources.line_index_buffer,
        PrimitiveType::PointList => &gpu_resources.point_index_buffer,
    };

    render_pass.set_vertex_buffer(0, gpu_resources.vertex_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(first_index..(first_index + index_count), 0, 0..1);
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
            // Upload the effective (screen-space-adjusted) transform. For
            // ordinary geometry this equals the world transform.
            transform: inst_transform.effective_transform.into(),
            normal_mat: inst_transform.effective_normal_matrix.into(),
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
    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    render_pass.draw_indexed(0..index_count, 0, 0..n_instances);
}
