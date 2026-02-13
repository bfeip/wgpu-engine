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

pub use hdr_loader::{load_hdr_from_bytes, load_hdr_from_path, HdrImage};

pub(crate) use brdf_lut::{BrdfLut, BrdfLutPipeline};
pub(crate) use cubemap::GpuCubemap;
pub(crate) use equirect::EquirectToCubePipeline;
pub(crate) use irradiance::IrradiancePipeline;
pub(crate) use prefilter::PrefilterPipeline;

// Re-export EnvironmentMap types from scene crate
pub use wgpu_engine_scene::environment::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};

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
/// When compute shaders are unavailable (e.g. WebGL backend), only the bind group
/// layout is created and IBL processing is skipped.
pub(crate) struct IblResources {
    /// Pipeline for converting equirectangular HDR to cubemap (None without compute).
    equirect_pipeline: Option<EquirectToCubePipeline>,
    /// Pipeline for generating irradiance maps (None without compute).
    irradiance_pipeline: Option<IrradiancePipeline>,
    /// Pipeline for generating pre-filtered environment maps (None without compute).
    prefilter_pipeline: Option<PrefilterPipeline>,
    /// Pipeline for generating the BRDF LUT (None without compute).
    _brdf_lut_pipeline: Option<BrdfLutPipeline>,
    /// Shared BRDF lookup table (None without compute).
    pub brdf_lut: Option<BrdfLut>,
    /// Bind group layout for IBL sampling in fragment shaders.
    /// Always present — needed for pipeline layout creation even when IBL is unavailable.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Whether compute shaders are available.
    has_compute: bool,
    /// Processed environment maps by ID.
    processed_environments: std::collections::HashMap<EnvironmentMapId, ProcessedEnvironment>,
}

impl IblResources {
    /// Create new IBL resources.
    ///
    /// When `has_compute` is true, initializes all compute pipelines and generates the
    /// BRDF LUT. When false (e.g. WebGL backend), only creates the bind group layout
    /// needed for pipeline layout compatibility.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, has_compute: bool) -> Self {
        // Create bind group layout for IBL sampling — always needed for pipeline layouts
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

        if has_compute {
            let equirect_pipeline = EquirectToCubePipeline::new(device);
            let irradiance_pipeline = IrradiancePipeline::new(device);
            let prefilter_pipeline = PrefilterPipeline::new(device);
            let brdf_lut_pipeline = BrdfLutPipeline::new(device);
            let brdf_lut = brdf_lut_pipeline.generate(device, queue);

            Self {
                equirect_pipeline: Some(equirect_pipeline),
                irradiance_pipeline: Some(irradiance_pipeline),
                prefilter_pipeline: Some(prefilter_pipeline),
                _brdf_lut_pipeline: Some(brdf_lut_pipeline),
                brdf_lut: Some(brdf_lut),
                bind_group_layout,
                has_compute,
                processed_environments: std::collections::HashMap::new(),
            }
        } else {
            log::warn!("Compute shaders unavailable — IBL environment maps will not be processed.");
            Self {
                equirect_pipeline: None,
                irradiance_pipeline: None,
                prefilter_pipeline: None,
                _brdf_lut_pipeline: None,
                brdf_lut: None,
                bind_group_layout,
                has_compute,
                processed_environments: std::collections::HashMap::new(),
            }
        }
    }

    /// Process an environment map, generating all required GPU resources.
    ///
    /// When compute shaders are unavailable, marks the environment as processed
    /// (to avoid repeated warnings) and returns without generating IBL resources.
    pub fn process_environment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        env_map: &mut EnvironmentMap,
    ) -> anyhow::Result<()> {
        if !env_map.needs_generation() {
            return Ok(());
        }

        if !self.has_compute {
            log::warn!("Skipping environment map processing (no compute shaders). Scene will render without IBL.");
            env_map.dirty = false;
            return Ok(());
        }

        // Load HDR image from source
        let hdr_image = match &env_map.source {
            EnvironmentSource::EquirectangularPath(path) => load_hdr_from_path(path)?,
            EnvironmentSource::EquirectangularHdr(bytes) => load_hdr_from_bytes(bytes)?,
        };

        // The unwraps below are safe: has_compute is true so pipelines were initialized
        let equirect = self.equirect_pipeline.as_ref().unwrap();
        let irradiance_pipeline = self.irradiance_pipeline.as_ref().unwrap();
        let prefilter_pipeline = self.prefilter_pipeline.as_ref().unwrap();
        let brdf_lut = self.brdf_lut.as_ref().unwrap();

        // Convert equirectangular to cubemap
        let environment = equirect.convert(device, queue, &hdr_image);

        // Generate irradiance map
        let irradiance = irradiance_pipeline.generate(device, queue, &environment);

        // Generate pre-filtered map
        let prefiltered = prefilter_pipeline.generate(device, queue, &environment);

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
                    resource: wgpu::BindingResource::TextureView(&brdf_lut.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&brdf_lut.sampler),
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
