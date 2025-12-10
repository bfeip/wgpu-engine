use std::collections::HashMap;
use std::path::PathBuf;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, StandardResolver};

use crate::scene::MaterialProperties;

/// Shader generator using WESL compiler to create modular shaders
pub(crate) struct ShaderGenerator {
    /// WESL compiler instance
    compiler: Wesl<StandardResolver>,
    /// Cache of compiled shader modules (MaterialProperties → ShaderModule)
    module_cache: HashMap<MaterialProperties, wgpu::ShaderModule>,
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
            module_cache: HashMap::new(),
        }
    }

    /// Generate a shader module for the given material properties
    pub fn generate_shader(&mut self, device: &wgpu::Device, properties: &MaterialProperties) -> anyhow::Result<wgpu::ShaderModule> {
        // Check cache first
        if let Some(cached) = self.module_cache.get(properties) {
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

        let shader_label = match (properties.has_texture, properties.has_lighting) {
            (true, true) => "Lit Texture Material Shader",
            (true, false) => "Unlit Texture Material Shader",
            (false, true) => "Lit Color Material Shader",
            (false, false) => "Unlit Color Material Shader",
        };

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into())
        });

        // Cache and return
        self.module_cache.insert(properties.clone(), module.clone());
        Ok(module)
    }
}
