// constants.wesl
// Shader constants for bind groups and bindings

// Mathematical constants
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// Bind group indices
const GROUP_CAMERA: u32 = 0u;
const GROUP_LIGHTS: u32 = 1u;
const GROUP_MATERIAL: u32 = 2u;
const GROUP_ENVIRONMENT: u32 = 3u;

// Camera bind group bindings
const BINDING_CAMERA_UNIFORM: u32 = 0u;

// Lights bind group bindings
const BINDING_LIGHTS_UNIFORM: u32 = 0u;

// Lighting constants
const AMBIENT_FACTOR: f32 = 0.001;

// Material bind group bindings.
//
// Every surface material shares one binding scheme. Binding 0 is always the
// material params uniform. Each texture occupies a FIXED slot (base color 1/2,
// normal 3/4, metallic-roughness 5/6) and is present only in the variants whose
// `has_*_texture` feature requests it. Absent textures leave their slots empty
// rather than renumbering the others — binding numbers are explicit indices, so
// gaps are legal (cf. `outline_screenspace.wesl`). Fixed slots mean the WESL
// `@binding` and the Rust layout builder never have to agree on a variable
// numbering: a texture is always at the same binding regardless of its siblings.
const BINDING_MATERIAL_PARAMS: u32 = 0u;
const BINDING_MATERIAL_BASE_COLOR_TEXTURE: u32 = 1u;
const BINDING_MATERIAL_BASE_COLOR_SAMPLER: u32 = 2u;
const BINDING_MATERIAL_NORMAL_TEXTURE: u32 = 3u;
const BINDING_MATERIAL_NORMAL_SAMPLER: u32 = 4u;
const BINDING_MATERIAL_METALLIC_ROUGHNESS_TEXTURE: u32 = 5u;
const BINDING_MATERIAL_METALLIC_ROUGHNESS_SAMPLER: u32 = 6u;

// Color material binding for the standalone flat-color overlay shader.
const BINDING_MATERIAL_COLOR: u32 = 0u;

// IBL (Image Based Lighting) bind group bindings
const BINDING_IBL_IRRADIANCE_TEXTURE: u32 = 0u;
const BINDING_IBL_IRRADIANCE_SAMPLER: u32 = 1u;
const BINDING_IBL_PREFILTERED_TEXTURE: u32 = 2u;
const BINDING_IBL_PREFILTERED_SAMPLER: u32 = 3u;
const BINDING_IBL_BRDF_LUT_TEXTURE: u32 = 4u;
const BINDING_IBL_BRDF_LUT_SAMPLER: u32 = 5u;
const BINDING_IBL_ENV_PARAMS: u32 = 6u;

// IBL constants
const MAX_REFLECTION_LOD: f32 = 4.0;
