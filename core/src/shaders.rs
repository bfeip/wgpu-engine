use std::collections::HashMap;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, VirtualResolver};

use crate::scene::{MaterialProperties, SceneProperties};

/// Combined key for shader cache (material + scene properties)
type ShaderCacheKey = (MaterialProperties, SceneProperties);

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
// PBR shader modules
const SHADER_PBR: &str = include_str!("shaders/pbr.wesl");
const SHADER_MATERIAL_PBR: &str = include_str!("shaders/material_pbr.wesl");
const SHADER_NORMAL_MAPPING: &str = include_str!("shaders/normal_mapping.wesl");
const SHADER_FRAGMENT_PBR_LIT: &str = include_str!("shaders/fragment_pbr_lit.wesl");
// IBL shader module
const SHADER_IBL: &str = include_str!("shaders/ibl.wesl");
// Screen-space outline shader modules
const SHADER_OUTLINE_MASK: &str = include_str!("shaders/outline_mask.wesl");
const SHADER_OUTLINE_SCREENSPACE: &str = include_str!("shaders/outline_screenspace.wesl");

/// Shader generator using WESL compiler to create modular shaders
pub(crate) struct ShaderGenerator {
    /// WESL compiler instance with embedded shader sources
    compiler: Wesl<VirtualResolver<'static>>,
    /// Cache of compiled shader modules keyed by (MaterialProperties, SceneProperties)
    module_cache: HashMap<ShaderCacheKey, wgpu::ShaderModule>,
    /// Cached outline mask shader module
    outline_mask_cache: Option<wgpu::ShaderModule>,
    /// Cached outline screenspace shader module
    outline_screenspace_cache: Option<wgpu::ShaderModule>,
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
        // PBR modules
        resolver.add_module("package::pbr".parse().unwrap(), SHADER_PBR.into());
        resolver.add_module("package::material_pbr".parse().unwrap(), SHADER_MATERIAL_PBR.into());
        resolver.add_module("package::normal_mapping".parse().unwrap(), SHADER_NORMAL_MAPPING.into());
        resolver.add_module("package::fragment_pbr_lit".parse().unwrap(), SHADER_FRAGMENT_PBR_LIT.into());
        // IBL module
        resolver.add_module("package::ibl".parse().unwrap(), SHADER_IBL.into());
        // Screen-space outline modules
        resolver.add_module("package::outline_mask".parse().unwrap(), SHADER_OUTLINE_MASK.into());
        resolver.add_module("package::outline_screenspace".parse().unwrap(), SHADER_OUTLINE_SCREENSPACE.into());

        // Create compiler with standard extensions enabled, then swap in the virtual resolver
        let compiler = Wesl::new(".").set_custom_resolver(resolver);

        Self {
            compiler,
            module_cache: HashMap::new(),
            outline_mask_cache: None,
            outline_screenspace_cache: None,
        }
    }

    /// Generate a shader module for the given material and scene properties
    pub fn generate_shader(
        &mut self,
        device: &wgpu::Device,
        material_props: &MaterialProperties,
        scene_props: &SceneProperties,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        // Check cache first
        let cache_key = (material_props.clone(), scene_props.clone());
        if let Some(cached) = self.module_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Build feature map for WESL conditional compilation
        // use_pbr enables the PBR shader path (requires PBR bind group layout)
        // For now, enable PBR when normal or metallic-roughness textures are present
        let use_pbr = material_props.has_normal_map || material_props.has_metallic_roughness_texture;
        let features = [
            ("has_texture", material_props.has_base_color_texture),
            ("has_lighting", material_props.has_lighting),
            ("use_pbr", use_pbr),
            ("has_ibl", scene_props.has_ibl && use_pbr && material_props.has_lighting),
        ];

        // Set features and compile the main module
        let path: ModulePath = "package::main".parse()?;
        self.compiler.set_features(features);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        let shader_label = if use_pbr && material_props.has_lighting {
            if scene_props.has_ibl {
                "PBR Lit IBL Material Shader"
            } else {
                "PBR Lit Material Shader"
            }
        } else {
            match (material_props.has_base_color_texture, material_props.has_lighting) {
                (true, true) => "Lit Texture Material Shader",
                (true, false) => "Unlit Texture Material Shader",
                (false, true) => "Lit Color Material Shader",
                (false, false) => "Unlit Color Material Shader",
            }
        };

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into())
        });

        // Cache and return
        self.module_cache.insert(cache_key, module.clone());
        Ok(module)
    }

    /// Generate the outline mask shader module.
    /// Renders selected objects to a mask texture.
    pub fn generate_outline_mask_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        if let Some(cached) = &self.outline_mask_cache {
            return Ok(cached.clone());
        }

        let path: ModulePath = "package::outline_mask".parse()?;
        let empty_features: [(&str, bool); 0] = [];
        self.compiler.set_features(empty_features);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Outline Mask Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        self.outline_mask_cache = Some(module.clone());
        Ok(module)
    }

    /// Generate the screen-space outline shader module.
    /// Fullscreen post-process that samples the mask texture to draw outlines.
    pub fn generate_outline_screenspace_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        if let Some(cached) = &self.outline_screenspace_cache {
            return Ok(cached.clone());
        }

        let path: ModulePath = "package::outline_screenspace".parse()?;
        let empty_features: [(&str, bool); 0] = [];
        self.compiler.set_features(empty_features);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Outline Screenspace Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        self.outline_screenspace_cache = Some(module.clone());
        Ok(module)
    }
}
