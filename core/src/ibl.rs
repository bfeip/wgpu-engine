//! Image Based Lighting (IBL) support for PBR rendering.
//!
//! This module provides environment map loading, processing, and GPU resource management
//! for image-based lighting. Environment maps are loaded from equirectangular HDR images
//! and processed into the required formats (irradiance map, pre-filtered environment map,
//! and BRDF LUT) using GPU compute shaders.

mod brdf_lut;
mod cubemap;
mod equirect;
mod hdr_loader;
mod irradiance;
mod prefilter;

pub use hdr_loader::{load_hdr_from_path, HdrImage};

pub(crate) use brdf_lut::{BrdfLut, BrdfLutPipeline};
pub(crate) use cubemap::GpuCubemap;
pub(crate) use equirect::EquirectToCubePipeline;
pub(crate) use irradiance::IrradiancePipeline;
pub(crate) use prefilter::PrefilterPipeline;

// Re-export EnvironmentMap types from scene crate
pub use wgpu_engine_scene::scene::environment::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};

/// Size of the environment cubemap (per face).
pub const ENVIRONMENT_CUBEMAP_SIZE: u32 = 512;

/// Size of the irradiance cubemap (per face). Low resolution since irradiance is low-frequency.
pub const IRRADIANCE_CUBEMAP_SIZE: u32 = 32;

/// Size of the pre-filtered environment cubemap base level (per face).
pub const PREFILTERED_CUBEMAP_SIZE: u32 = 128;

/// Number of mip levels for the pre-filtered cubemap (roughness levels).
pub const PREFILTERED_MIP_LEVELS: u32 = 5;

/// Size of the BRDF integration LUT.
pub const BRDF_LUT_SIZE: u32 = 512;

/// Processed GPU resources for an environment map.
#[derive(Debug)]
pub(crate) struct ProcessedEnvironment {
    /// Original environment as a cubemap.
    pub _environment: GpuCubemap,
    /// Diffuse irradiance cubemap.
    pub _irradiance: GpuCubemap,
    /// Pre-filtered specular cubemap.
    pub _prefiltered: GpuCubemap,
    /// Bind group for sampling in fragment shader.
    pub bind_group: wgpu::BindGroup,
}

/// GPU resources and pipelines for Image Based Lighting.
///
/// This struct manages all IBL-related GPU resources including compute pipelines
/// for processing environment maps and the shared BRDF lookup table.
pub(crate) struct IblResources {
    /// Pipeline for converting equirectangular HDR to cubemap.
    equirect_pipeline: EquirectToCubePipeline,
    /// Pipeline for generating irradiance maps.
    irradiance_pipeline: IrradiancePipeline,
    /// Pipeline for generating pre-filtered environment maps.
    prefilter_pipeline: PrefilterPipeline,
    /// Pipeline for generating the BRDF LUT.
    _brdf_lut_pipeline: BrdfLutPipeline,
    /// Shared BRDF lookup table (generated once).
    pub brdf_lut: BrdfLut,
    /// Bind group layout for IBL sampling in fragment shaders.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Processed environment maps by ID.
    processed_environments: std::collections::HashMap<EnvironmentMapId, ProcessedEnvironment>,
}

impl IblResources {
    /// Create new IBL resources.
    ///
    /// This initializes all compute pipelines and generates the BRDF LUT.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let equirect_pipeline = EquirectToCubePipeline::new(device);
        let irradiance_pipeline = IrradiancePipeline::new(device);
        let prefilter_pipeline = PrefilterPipeline::new(device);
        let brdf_lut_pipeline = BrdfLutPipeline::new(device);

        // Generate BRDF LUT once at startup
        let brdf_lut = brdf_lut_pipeline.generate(device, queue);

        // Create bind group layout for IBL sampling
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Bind Group Layout"),
            entries: &[
                // Irradiance cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // Irradiance sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Pre-filtered cubemap
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // Pre-filtered sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // BRDF LUT
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // BRDF LUT sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self {
            equirect_pipeline,
            irradiance_pipeline,
            prefilter_pipeline,
            _brdf_lut_pipeline: brdf_lut_pipeline,
            brdf_lut,
            bind_group_layout,
            processed_environments: std::collections::HashMap::new(),
        }
    }

    /// Process an environment map, generating all required GPU resources.
    pub fn process_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        env_map: &mut EnvironmentMap,
    ) -> anyhow::Result<()> {
        if !env_map.needs_generation() {
            return Ok(());
        }

        // Load HDR image from source
        let hdr_image = match &env_map.source {
            EnvironmentSource::EquirectangularPath(path) => load_hdr_from_path(path)?,
        };

        // Convert equirectangular to cubemap
        let environment = self.equirect_pipeline.convert(device, queue, &hdr_image);

        // Generate irradiance map
        let irradiance = self.irradiance_pipeline.generate(device, queue, &environment);

        // Generate pre-filtered map
        let prefiltered = self.prefilter_pipeline.generate(device, queue, &environment);

        // Create bind group for fragment shader sampling
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("IBL Bind Group Env {}", env_map.id)),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&irradiance.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&irradiance.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&prefiltered.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&prefiltered.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.brdf_lut.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.brdf_lut.sampler),
                },
            ],
        });

        // Store processed environment
        self.processed_environments.insert(
            env_map.id,
            ProcessedEnvironment {
                _environment: environment,
                _irradiance: irradiance,
                _prefiltered: prefiltered,
                bind_group,
            },
        );

        env_map.dirty = false;
        Ok(())
    }

    /// Get the processed environment for an ID, if available.
    pub fn get_processed(&self, id: EnvironmentMapId) -> Option<&ProcessedEnvironment> {
        self.processed_environments.get(&id)
    }

    /// Remove a processed environment.
    pub fn _remove_processed(&mut self, id: EnvironmentMapId) {
        self.processed_environments.remove(&id);
    }
}
