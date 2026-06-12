use crate::render_core::ShaderLibrary;
use crate::scene::{AlphaMode, MaterialProperties, SceneProperties};

// Embed shader sources at compile time for WASM compatibility
const ENGINE_MODULES: [(&str, &str); 17] = [
    ("package::main", include_str!("shaders/main.wesl")),
    ("package::common", include_str!("shaders/common.wesl")),
    ("package::camera", include_str!("shaders/camera.wesl")),
    ("package::constants", include_str!("shaders/constants.wesl")),
    ("package::lighting", include_str!("shaders/lighting.wesl")),
    ("package::vertex", include_str!("shaders/vertex.wesl")),
    ("package::material_color", include_str!("shaders/material_color.wesl")),
    ("package::fragment_color_unlit", include_str!("shaders/fragment_color_unlit.wesl")),
    ("package::flat_color", include_str!("shaders/flat_color.wesl")),
    // PBR modules
    ("package::pbr", include_str!("shaders/pbr.wesl")),
    ("package::material_pbr", include_str!("shaders/material_pbr.wesl")),
    ("package::normal_mapping", include_str!("shaders/normal_mapping.wesl")),
    ("package::fragment_pbr_lit", include_str!("shaders/fragment_pbr_lit.wesl")),
    // IBL module
    ("package::ibl", include_str!("shaders/ibl.wesl")),
    // Screen-space outline modules
    ("package::outline_mask", include_str!("shaders/outline_mask.wesl")),
    ("package::outline_screenspace", include_str!("shaders/outline_screenspace.wesl")),
    // Silhouette edge detection shader
    ("package::silhouette_edges", include_str!("shaders/silhouette_edges.wesl")),
];

/// Build a [`ShaderLibrary`] pre-loaded with all engine shader modules.
fn engine_library() -> ShaderLibrary {
    ShaderLibrary::new(ENGINE_MODULES)
}

/// Compile a user-supplied WESL shader with access to all engine shader modules.
///
/// The user's shader source is compiled with engine modules available for
/// import (`package::common`, `package::camera`, `package::lighting`, etc.).
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
    engine_library().compile_adhoc(device, source)
}

/// Typed entry points for the engine's shader variants.
///
/// Thin wrapper over a [`ShaderLibrary`] holding the engine WESL modules: each
/// method maps typed renderer state (material/scene properties, MSAA flags) to
/// the module path + feature flags the library compiles. Variant caching lives
/// in the library.
pub(crate) struct ShaderGenerator {
    library: ShaderLibrary,
}

impl ShaderGenerator {
    /// Create a new ShaderGenerator
    ///
    /// All engine shader sources are embedded at compile time, enabling
    /// compatibility with WASM and other environments without filesystem access.
    pub fn new() -> Self {
        Self { library: engine_library() }
    }

    /// Generate a shader module for the given material and scene properties
    pub fn generate_shader(
        &mut self,
        device: &wgpu::Device,
        material_props: &MaterialProperties,
        scene_props: &SceneProperties,
        depth_prepass: bool,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        // Build feature map for WESL conditional compilation
        let features = [
            ("has_lighting", material_props.has_lighting),
            ("has_ibl", scene_props.has_ibl && material_props.has_lighting),
            ("double_sided", material_props.double_sided),
            ("alpha_mask", material_props.alpha_mode == AlphaMode::Mask),
            ("depth_prepass", depth_prepass),
        ];

        let label = if material_props.has_lighting {
            if scene_props.has_ibl {
                "PBR Lit IBL Shader"
            } else {
                "PBR Lit Shader"
            }
        } else {
            "Unlit Color Shader"
        };

        self.library.compile(device, "package::main", &features, label)
    }

    /// Generate the outline mask shader module.
    /// Renders selected objects to a mask texture.
    pub fn generate_outline_mask_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        self.library
            .compile(device, "package::outline_mask", &[], "Outline Mask Shader")
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
        let label = if multisampled {
            "Outline Screenspace Shader (MSAA)"
        } else {
            "Outline Screenspace Shader"
        };
        self.library.compile(
            device,
            "package::outline_screenspace",
            &[("multisampled", multisampled)],
            label,
        )
    }

    /// Generate the flat-color shader module.
    /// Renders geometry using a color from the material uniform (group 2) with no lighting.
    pub fn generate_flat_color_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        self.library
            .compile(device, "package::flat_color", &[], "Flat Color Shader")
    }

    /// Generate the silhouette edge detection shader module.
    /// `depth_multisampled = true` produces a variant that reads from a
    /// `texture_depth_multisampled_2d`; `false` reads a `texture_depth_2d`.
    pub fn generate_silhouette_shader(
        &mut self,
        device: &wgpu::Device,
        depth_multisampled: bool,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        let label = if depth_multisampled {
            "Silhouette Edges Shader (MSAA)"
        } else {
            "Silhouette Edges Shader"
        };
        self.library.compile(
            device,
            "package::silhouette_edges",
            &[("depth_multisampled", depth_multisampled)],
            label,
        )
    }
}
