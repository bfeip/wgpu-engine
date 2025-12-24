use std::collections::HashMap;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, VirtualResolver};

use crate::scene::MaterialProperties;

// Embed shader sources at compile time for WASM compatibility
const SHADER_MAIN: &str = include_str!("shaders/main.wesl");
const SHADER_COMMON: &str = include_str!("shaders/common.wesl");
const SHADER_CAMERA: &str = include_str!("shaders/camera.wesl");
const SHADER_CONSTANTS: &str = include_str!("shaders/constants.wesl");
const SHADER_LIGHTING: &str = include_str!("shaders/lighting.wesl");
const SHADER_VERTEX: &str = include_str!("shaders/vertex.wesl");
const SHADER_MATERIAL_COLOR: &str = include_str!("shaders/material_color.wesl");
const SHADER_MATERIAL_TEXTURE: &str = include_str!("shaders/material_texture.wesl");
const SHADER_FRAGMENT_COLOR_LIT: &str = include_str!("shaders/fragment_color_lit.wesl");
const SHADER_FRAGMENT_COLOR_UNLIT: &str = include_str!("shaders/fragment_color_unlit.wesl");
const SHADER_FRAGMENT_TEXTURE_LIT: &str = include_str!("shaders/fragment_texture_lit.wesl");

/// Shader generator using WESL compiler to create modular shaders
pub(crate) struct ShaderGenerator {
    /// WESL compiler instance with embedded shader sources
    compiler: Wesl<VirtualResolver<'static>>,
    /// Cache of compiled shader modules (MaterialProperties → ShaderModule)
    module_cache: HashMap<MaterialProperties, wgpu::ShaderModule>,
}

impl ShaderGenerator {
    /// Create a new ShaderGenerator
    ///
    /// This initializes the WESL compiler with embedded shader sources,
    /// enabling compatibility with WASM and other environments without filesystem access.
    pub fn new() -> Self {
        let mut resolver = VirtualResolver::default();

        // Add all shader modules with their package paths
        resolver.add_module("package::main".parse().unwrap(), SHADER_MAIN.into());
        resolver.add_module("package::common".parse().unwrap(), SHADER_COMMON.into());
        resolver.add_module("package::camera".parse().unwrap(), SHADER_CAMERA.into());
        resolver.add_module("package::constants".parse().unwrap(), SHADER_CONSTANTS.into());
        resolver.add_module("package::lighting".parse().unwrap(), SHADER_LIGHTING.into());
        resolver.add_module("package::vertex".parse().unwrap(), SHADER_VERTEX.into());
        resolver.add_module("package::material_color".parse().unwrap(), SHADER_MATERIAL_COLOR.into());
        resolver.add_module("package::material_texture".parse().unwrap(), SHADER_MATERIAL_TEXTURE.into());
        resolver.add_module("package::fragment_color_lit".parse().unwrap(), SHADER_FRAGMENT_COLOR_LIT.into());
        resolver.add_module("package::fragment_color_unlit".parse().unwrap(), SHADER_FRAGMENT_COLOR_UNLIT.into());
        resolver.add_module("package::fragment_texture_lit".parse().unwrap(), SHADER_FRAGMENT_TEXTURE_LIT.into());

        // Create compiler with standard extensions enabled, then swap in the virtual resolver
        let compiler = Wesl::new(".").set_custom_resolver(resolver);

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
