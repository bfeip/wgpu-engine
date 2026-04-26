use std::borrow::Cow;
use std::collections::HashMap;

use wgpu::ShaderModuleDescriptor;
use wesl::{Wesl, ModulePath, VirtualResolver};

use crate::scene::{AlphaMode, MaterialProperties, SceneProperties};

/// Combined key for shader cache (material + scene properties + depth prepass)
type ShaderCacheKey = (MaterialProperties, SceneProperties, bool);

// Embed shader sources at compile time for WASM compatibility
const SHADER_MAIN: &str = include_str!("shaders/main.wesl");
const SHADER_COMMON: &str = include_str!("shaders/common.wesl");
const SHADER_CAMERA: &str = include_str!("shaders/camera.wesl");
const SHADER_CONSTANTS: &str = include_str!("shaders/constants.wesl");
const SHADER_LIGHTING: &str = include_str!("shaders/lighting.wesl");
const SHADER_VERTEX: &str = include_str!("shaders/vertex.wesl");
const SHADER_MATERIAL_COLOR: &str = include_str!("shaders/material_color.wesl");
const SHADER_FRAGMENT_COLOR_UNLIT: &str = include_str!("shaders/fragment_color_unlit.wesl");
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
// Hidden-line workflow shader
const SHADER_HIDDEN_LINE_SOLID: &str = include_str!("shaders/hidden_line_solid.wesl");

/// Build a VirtualResolver pre-loaded with all engine shader modules.
///
/// All module sources are `&'static str` embedded at compile time, so
/// registration is zero-copy (HashMap entries hold borrowed pointers).
fn build_engine_resolver() -> VirtualResolver<'static> {
    let mut resolver = VirtualResolver::default();

    resolver.add_module("package::main".parse().unwrap(), SHADER_MAIN.into());
    resolver.add_module("package::common".parse().unwrap(), SHADER_COMMON.into());
    resolver.add_module("package::camera".parse().unwrap(), SHADER_CAMERA.into());
    resolver.add_module("package::constants".parse().unwrap(), SHADER_CONSTANTS.into());
    resolver.add_module("package::lighting".parse().unwrap(), SHADER_LIGHTING.into());
    resolver.add_module("package::vertex".parse().unwrap(), SHADER_VERTEX.into());
    resolver.add_module("package::material_color".parse().unwrap(), SHADER_MATERIAL_COLOR.into());
    resolver.add_module("package::fragment_color_unlit".parse().unwrap(), SHADER_FRAGMENT_COLOR_UNLIT.into());
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
    resolver.add_module("package::hidden_line_solid".parse().unwrap(), SHADER_HIDDEN_LINE_SOLID.into());

    resolver
}

/// Compile a user-supplied WESL shader with access to all engine shader modules.
///
/// The user's shader source is compiled with engine modules available for
/// import (`package::common`, `package::camera`, `package::lighting`, etc.).
///
/// A fresh resolver is built each call to avoid contaminating the engine's
/// cached compiler state. `VirtualResolver` does not implement `Clone`, and the
/// engine's feature flags must not bleed into user compilation. The build cost
/// is negligible: 15 HashMap inserts of `&'static str` pointers.
///
/// The user module is registered as `package::user` so that `package::` imports
/// in the user's WESL resolve correctly. In WESL, `package::` means "the current
/// package", so user code must live in the same namespace as the engine modules.
/// Use specific item imports, e.g.:
/// ```wesl
/// import package::common::{VertexInput, InstanceInput, LIGHT_TYPE_DIRECTIONAL};
/// import package::camera::camera;    // the camera uniform global
/// import package::lighting::lights;  // the lights uniform global
/// ```
pub(crate) fn compile_user_wesl(device: &wgpu::Device, source: &str) -> anyhow::Result<wgpu::ShaderModule> {
    let mut resolver = build_engine_resolver();
    resolver.add_module("package::user".parse()?, Cow::Owned(source.to_owned()));

    let mut compiler = Wesl::new(".").set_custom_resolver(resolver);
    let path: ModulePath = "package::user".parse()?;
    let empty_features: [(&str, bool); 0] = [];
    compiler.set_features(empty_features);
    let result = compiler.compile(&path)?;
    let wgsl = result.to_string();

    Ok(device.create_shader_module(ShaderModuleDescriptor {
        label: Some("User WESL Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    }))
}

/// Shader generator using WESL compiler to create modular shaders
pub(crate) struct ShaderGenerator {
    /// WESL compiler instance with embedded shader sources
    compiler: Wesl<VirtualResolver<'static>>,
    /// Cache of compiled shader modules keyed by (MaterialProperties, SceneProperties)
    module_cache: HashMap<ShaderCacheKey, wgpu::ShaderModule>,
    /// Cached outline mask shader module
    outline_mask_cache: Option<wgpu::ShaderModule>,
    /// Cached outline screenspace shader module (non-MSAA, multisampled=false)
    outline_screenspace_cache: Option<wgpu::ShaderModule>,
    /// Cached outline screenspace shader module (MSAA, multisampled=true)
    outline_screenspace_ms_cache: Option<wgpu::ShaderModule>,
    /// Cached hidden-line solid shader module
    hidden_line_solid_cache: Option<wgpu::ShaderModule>,
}

impl ShaderGenerator {
    /// Create a new ShaderGenerator
    ///
    /// This initializes the WESL compiler with embedded shader sources,
    /// enabling compatibility with WASM and other environments without filesystem access.
    pub fn new() -> Self {
        let compiler = Wesl::new(".").set_custom_resolver(build_engine_resolver());

        Self {
            compiler,
            module_cache: HashMap::new(),
            outline_mask_cache: None,
            outline_screenspace_cache: None,
            outline_screenspace_ms_cache: None,
            hidden_line_solid_cache: None,
        }
    }

    /// Generate a shader module for the given material and scene properties
    pub fn generate_shader(
        &mut self,
        device: &wgpu::Device,
        material_props: &MaterialProperties,
        scene_props: &SceneProperties,
        depth_prepass: bool,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        // Check cache first
        let cache_key = (material_props.clone(), scene_props.clone(), depth_prepass);
        if let Some(cached) = self.module_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Build feature map for WESL conditional compilation
        let features = [
            ("has_lighting", material_props.has_lighting),
            ("has_ibl", scene_props.has_ibl && material_props.has_lighting),
            ("double_sided", material_props.double_sided),
            ("alpha_mask", material_props.alpha_mode == AlphaMode::Mask),
            ("depth_prepass", depth_prepass),
        ];

        // Set features and compile the main module
        let path: ModulePath = "package::main".parse()?;
        self.compiler.set_features(features);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        let shader_label = if material_props.has_lighting {
            if scene_props.has_ibl {
                "PBR Lit IBL Shader"
            } else {
                "PBR Lit Shader"
            }
        } else {
            "Unlit Color Shader"
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
    /// `multisampled = true` produces a variant that reads from a `texture_multisampled_2d`
    /// and averages MSAA samples for smooth edge coverage.
    /// `multisampled = false` produces the standard single-sample variant.
    pub fn generate_outline_screenspace_shader(
        &mut self,
        device: &wgpu::Device,
        multisampled: bool,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        let cache = if multisampled {
            &mut self.outline_screenspace_ms_cache
        } else {
            &mut self.outline_screenspace_cache
        };
        if let Some(cached) = cache {
            return Ok(cached.clone());
        }

        let path: ModulePath = "package::outline_screenspace".parse()?;
        self.compiler.set_features([("multisampled", multisampled)]);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        let label = if multisampled {
            "Outline Screenspace Shader (MSAA)"
        } else {
            "Outline Screenspace Shader"
        };
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let cache = if multisampled {
            &mut self.outline_screenspace_ms_cache
        } else {
            &mut self.outline_screenspace_cache
        };
        *cache = Some(module.clone());
        Ok(module)
    }

    /// Generate the hidden-line solid shader module.
    /// Renders geometry flat-white with no lighting for the hidden-line workflow.
    pub fn generate_hidden_line_solid_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        if let Some(cached) = &self.hidden_line_solid_cache {
            return Ok(cached.clone());
        }

        let path: ModulePath = "package::hidden_line_solid".parse()?;
        let empty_features: [(&str, bool); 0] = [];
        self.compiler.set_features(empty_features);
        let result = self.compiler.compile(&path)?;
        let wgsl = result.to_string();

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Hidden Line Solid Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        self.hidden_line_solid_cache = Some(module.clone());
        Ok(module)
    }
}
