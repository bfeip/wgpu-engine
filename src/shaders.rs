use std::collections::HashMap;
use std::path::PathBuf;
use std::hash::Hash;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, StandardResolver};

use crate::material::MaterialType;

/// Material shader properties that determine how the shader is generated
/// This will eventually replace MaterialType entirely
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderProperties {
    /// Whether this material uses a texture (vs solid color)
    has_texture: bool,
    /// Whether lighting calculations should be applied
    has_lighting: bool,
}

impl ShaderProperties {
    /// Convert from legacy MaterialType (temporary for Phase 1)
    fn from_material_type(material_type: MaterialType) -> Self {
        match material_type {
            MaterialType::FaceColor => ShaderProperties {
                has_texture: false,
                has_lighting: true,
            },
            MaterialType::FaceTexture => ShaderProperties {
                has_texture: true,
                has_lighting: true,
            },
            MaterialType::LineColor | MaterialType::PointColor => ShaderProperties {
                has_texture: false,
                has_lighting: false,
            },
        }
    }

    /// Determine which WESL fragment module to use
    fn get_fragment_module(&self) -> &'static str {
        match (self.has_texture, self.has_lighting) {
            (false, true) => "package::fragment_color_lit",
            (false, false) => "package::fragment_color_unlit",
            (true, true) => "package::fragment_texture_lit",
            (true, false) => {
                // Unlit texture materials not currently supported, but could be added
                // For now, fall back to lit version
                "package::fragment_texture_lit"
            }
        }
    }
}

/// Shader generator using WESL compiler to create modular shaders
pub(crate) struct ShaderGenerator {
    /// WESL compiler instance
    compiler: Wesl<StandardResolver>,
    /// Cache of compiled WGSL strings (ShaderProperties → WGSL)
    shader_cache: HashMap<ShaderProperties, String>,
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
        let properties = ShaderProperties::from_material_type(material_type);

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
    fn get_or_compile_wgsl(&mut self, properties: &ShaderProperties) -> anyhow::Result<String> {
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
