//! BRDF integration lookup table generation.
//!
//! Generates a 2D LUT used in the split-sum approximation for specular IBL.
//! The LUT stores pre-integrated BRDF terms that can be looked up at runtime
//! using N·V and roughness as coordinates.

use crate::ibl::BRDF_LUT_SIZE;

/// Shader source for BRDF LUT generation.
const BRDF_LUT_SHADER: &str = include_str!("../shaders/ibl/brdf_lut.wgsl");

/// GPU resources for the BRDF lookup table.
#[derive(Debug)]
pub struct BrdfLut {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

/// Pipeline for generating the BRDF lookup table.
pub struct BrdfLutPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BrdfLutPipeline {
    /// Create a new BRDF LUT generation pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BRDF LUT Shader"),
            source: wgpu::ShaderSource::Wgsl(BRDF_LUT_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BRDF LUT Bind Group Layout"),
            entries: &[
                // Output LUT storage texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BRDF LUT Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BRDF LUT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Generate the BRDF lookup table.
    ///
    /// This only needs to be done once - the LUT is independent of the environment map.
    pub fn generate(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> BrdfLut {
        // Create output texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BRDF LUT Texture"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("BRDF LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BRDF LUT Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BRDF LUT Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BRDF LUT Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = (BRDF_LUT_SIZE + 15) / 16;
            let workgroups_y = (BRDF_LUT_SIZE + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        BrdfLut {
            texture,
            view,
            sampler,
        }
    }
}
