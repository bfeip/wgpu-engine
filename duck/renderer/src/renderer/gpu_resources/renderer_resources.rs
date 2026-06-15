use std::collections::HashMap;

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::ibl::ibl_bind_group_layout;
use crate::renderer::surface_config::{SurfaceConfig, TexturePresence};
use crate::scene::PrimitiveType;

use super::uniforms::{CameraUniform, LightsArrayUniform};

/// GPU instance data for one camera: a uniform buffer plus its bind group.
pub(crate) struct CameraResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl CameraResources {
    /// Create a camera buffer and bind group against the shared camera layout.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> CameraResources {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        CameraResources { buffer, bind_group }
    }
}

/// GPU instance data for lighting uniforms: a buffer plus its bind group.
pub(crate) struct LightResources {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub synced_generation: u64,
}

impl LightResources {
    /// Create the light buffer and bind group against the shared light layout.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> LightResources {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<LightsArrayUniform>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("Light bind group"),
        });

        LightResources {
            buffer,
            bind_group,
            synced_generation: 0,
        }
    }
}

/// Canonical owner of every bind group layout used by the renderer.
///
/// A bind group layout is a *schema*: exactly one per kind, created once and
/// shared by every pipeline layout and every conforming bind group. Keeping them
/// all here (rather than bundled into the resource structs that happen to create
/// the first bind group of each kind) lets multiple instances — e.g. the main
/// camera plus per-sub-view camera slots — share a single layout, and gives
/// pipeline-layout construction one place to borrow from.
pub(crate) struct BindGroupLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub light: wgpu::BindGroupLayout,
    /// Color material layout for the standalone flat-color overlay shader
    /// (a single `vec4` uniform). Surface materials use the dynamically-derived
    /// layouts in [`MaterialLayoutCache`] instead.
    pub color: wgpu::BindGroupLayout,
    pub ibl: wgpu::BindGroupLayout,
}

impl BindGroupLayouts {
    /// Create all bind group layouts.
    pub fn new(device: &wgpu::Device) -> BindGroupLayouts {
        let camera = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // Visible in both VERTEX (for view_proj) and FRAGMENT (for eye_position in PBR)
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let light = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Light bind group layout"),
        });

        let color = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Color Material Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let ibl = ibl_bind_group_layout(device);

        BindGroupLayouts { camera, light, color, ibl }
    }
}

/// Lazily-built, cached layouts for surface materials.
///
/// Both the group-2 material bind-group layout and the full pipeline layout are
/// derived from a material's [`TexturePresence`] (and, for the pipeline layout,
/// whether IBL is active) — the same value that drives the shader's `@if`
/// bindings. Deriving layout and shader from one source is what guarantees they
/// always agree, so a variant binds exactly the textures it declares.
pub(crate) struct MaterialLayoutCache {
    camera: wgpu::BindGroupLayout,
    light: wgpu::BindGroupLayout,
    ibl: wgpu::BindGroupLayout,
    bind_group_layouts: HashMap<TexturePresence, wgpu::BindGroupLayout>,
    pipeline_layouts: HashMap<(TexturePresence, bool), wgpu::PipelineLayout>,
}

impl MaterialLayoutCache {
    pub fn new(layouts: &BindGroupLayouts) -> MaterialLayoutCache {
        MaterialLayoutCache {
            camera: layouts.camera.clone(),
            light: layouts.light.clone(),
            ibl: layouts.ibl.clone(),
            bind_group_layouts: HashMap::new(),
            pipeline_layouts: HashMap::new(),
        }
    }

    /// The group-2 bind-group layout for a material with the given textures:
    /// the params uniform at binding 0 plus a texture+sampler pair at the fixed
    /// slot of each present texture (absent slots are simply omitted).
    pub fn bind_group_layout(
        &mut self,
        device: &wgpu::Device,
        presence: TexturePresence,
    ) -> &wgpu::BindGroupLayout {
        self.bind_group_layouts
            .entry(presence)
            .or_insert_with(|| build_material_bind_group_layout(device, presence))
    }

    /// The pipeline layout for a variant: `[camera, lights, material]`, plus the
    /// IBL group when `has_ibl`.
    pub fn pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        presence: TexturePresence,
        has_ibl: bool,
    ) -> &wgpu::PipelineLayout {
        // Materialize the bind-group layout first so the borrow ends before we
        // touch `pipeline_layouts`; the clone is a cheap Arc bump.
        let material = self.bind_group_layout(device, presence).clone();
        let camera = &self.camera;
        let light = &self.light;
        let ibl = &self.ibl;
        self.pipeline_layouts
            .entry((presence, has_ibl))
            .or_insert_with(|| {
                let mut bgls: Vec<&wgpu::BindGroupLayout> = vec![camera, light, &material];
                if has_ibl {
                    bgls.push(ibl);
                }
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Surface Material Pipeline Layout"),
                    bind_group_layouts: &bgls,
                    push_constant_ranges: &[],
                })
            })
    }
}

/// Build the group-2 bind-group layout for the given texture presence.
///
/// Binding 0 is always the material params uniform; each present channel adds a
/// texture+sampler pair at its fixed slot (from [`MaterialTextureSlot`]). Binding
/// numbers are explicit, so gaps left by absent textures are legal and keep every
/// texture at a stable slot.
fn build_material_bind_group_layout(
    device: &wgpu::Device,
    presence: TexturePresence,
) -> wgpu::BindGroupLayout {
    fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
            },
            count: None,
        }
    }
    fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        }
    }

    // Binding 0: material params uniform (always present).
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for slot in presence.slots() {
        entries.push(texture_entry(slot.texture_binding()));
        entries.push(sampler_entry(slot.sampler_binding()));
    }

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Surface Material Bind Group Layout"),
        entries: &entries,
    })
}

/// Cache key for render pipelines: the surface variant plus its primitive topology.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineCacheKey {
    pub surface: SurfaceConfig,
    pub primitive_type: PrimitiveType,
}
