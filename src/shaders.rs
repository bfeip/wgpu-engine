use std::collections::HashMap;
use std::path::PathBuf;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, StandardResolver};

use crate::material::{MaterialType, MaterialProperties};

/// Shader generator using WESL compiler to create modular shaders
pub(crate) struct ShaderGenerator {
    /// WESL compiler instance
    compiler: Wesl<StandardResolver>,
    /// Cache of compiled WGSL strings (MaterialProperties → WGSL)
    shader_cache: HashMap<MaterialProperties, String>,
}

impl ShaderGenerator {
    /// Create a new ShaderGenerator
    ///
    /// This initializes the WESL compiler pointing to the shaders directory
    pub fn new() -> Self {
        // Get the shader directory path relative to the source root
        let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/shaders");

        // Initialize WESL compiler with StandardResolver
        let compiler = Wesl::new(&shader_dir);

        Self {
            compiler,
            shader_cache: HashMap::new(),
        }
    }

    /// Generate a shader module for the given material type (Phase 1 compatibility)
    ///
    /// In Phase 2+, this will be replaced with a method that takes MaterialProperties directly
    pub fn generate_shader(&mut self, device: &wgpu::Device, material_type: MaterialType) -> anyhow::Result<wgpu::ShaderModule> {
        // Convert MaterialType to properties (this translation layer will be removed in Phase 2)
        let properties = MaterialProperties::from_material_type(material_type);

        let wgsl = self.get_or_compile_wgsl(&properties)?;

        let shader_label = match material_type {
            MaterialType::FaceColor => "Face Color Material Shader",
            MaterialType::FaceTexture => "Face Texture Material Shader",
            MaterialType::LineColor => "Line Color Material Shader",
            MaterialType::PointColor => "Point Color Material Shader",
        };

        Ok(device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into())
        }))
    }

    /// Get or compile WGSL for shader properties (with caching)
    fn get_or_compile_wgsl(&mut self, properties: &MaterialProperties) -> anyhow::Result<String> {
        // Check cache first
        if let Some(cached) = self.shader_cache.get(properties) {
            return Ok(cached.clone());
        }

        // Build feature map for WESL conditional compilation
        let features = [
            ("has_texture", properties.has_texture),
            ("has_lighting", properties.has_lighting),
        ];

        // Set features and compile the main module
        let path: ModulePath = "package::main".parse()?;
        self.compiler.set_features(features);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        // Cache and return
        self.shader_cache.insert(properties.clone(), wgsl.clone());
        Ok(wgsl)
    }
}

/// Generates bind group layouts based on material properties
///
/// This ensures that Rust bind group layouts stay synchronized with WESL shader expectations.
/// Both are derived from the same MaterialProperties, guaranteeing consistency.
pub(crate) struct BindGroupGenerator {
    /// Cache of bind group layouts (MaterialProperties → BindGroupLayout)
    layout_cache: HashMap<MaterialProperties, wgpu::BindGroupLayout>,
}

impl BindGroupGenerator {
    /// Create a new BindGroupGenerator
    pub fn new() -> Self {
        Self {
            layout_cache: HashMap::new(),
        }
    }

    /// Get or generate a bind group layout for the given material properties
    ///
    /// The layout generation follows the same rules as the WESL shader bindings:
    /// - Color materials: binding 0 = uniform buffer (vec4<f32>)
    /// - Texture materials: binding 0 = texture, binding 1 = sampler
    pub fn get_or_generate_layout(
        &mut self,
        device: &wgpu::Device,
        properties: &MaterialProperties,
    ) -> &wgpu::BindGroupLayout {
        // Check if we need to generate
        if !self.layout_cache.contains_key(properties) {
            let layout = Self::generate_layout(device, properties);
            self.layout_cache.insert(properties.clone(), layout);
        }

        self.layout_cache.get(properties).unwrap()
    }

    /// Generate a bind group layout from material properties
    fn generate_layout(
        device: &wgpu::Device,
        properties: &MaterialProperties,
    ) -> wgpu::BindGroupLayout {
        let mut entries = Vec::new();

        if !properties.has_texture {
            // Color uniform at binding 0 (matches material_color.wesl)
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        } else {
            // Texture at binding 0 (matches material_texture.wesl)
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            });

            // Sampler at binding 1 (matches material_texture.wesl)
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &entries,
        })
    }
}
