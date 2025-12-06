use std::collections::HashMap;
use std::path::PathBuf;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, StandardResolver};

use crate::material::MaterialProperties;

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

    /// Generate a shader module for the given material properties
    pub fn generate_shader(&mut self, device: &wgpu::Device, properties: &MaterialProperties) -> anyhow::Result<wgpu::ShaderModule> {
        let wgsl = self.get_or_compile_wgsl(properties)?;

        let shader_label = match (properties.has_texture, properties.has_lighting) {
            (true, true) => "Lit Texture Material Shader",
            (true, false) => "Unlit Texture Material Shader",
            (false, true) => "Lit Color Material Shader",
            (false, false) => "Unlit Color Material Shader",
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
