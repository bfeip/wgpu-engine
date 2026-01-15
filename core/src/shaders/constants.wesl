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

// Material bind group bindings (simple material)
const BINDING_MATERIAL_COLOR: u32 = 0u;
const BINDING_MATERIAL_TEXTURE: u32 = 0u;
const BINDING_MATERIAL_SAMPLER: u32 = 1u;

// PBR lighting constants
const AMBIENT_FACTOR: f32 = 0.001;

// PBR material bind group bindings
const BINDING_PBR_PARAMS: u32 = 0u;
const BINDING_PBR_BASE_COLOR_TEXTURE: u32 = 1u;
const BINDING_PBR_BASE_COLOR_SAMPLER: u32 = 2u;
const BINDING_PBR_NORMAL_TEXTURE: u32 = 3u;
const BINDING_PBR_NORMAL_SAMPLER: u32 = 4u;
const BINDING_PBR_METALLIC_ROUGHNESS_TEXTURE: u32 = 5u;
const BINDING_PBR_METALLIC_ROUGHNESS_SAMPLER: u32 = 6u;

// IBL (Image Based Lighting) bind group bindings
const BINDING_IBL_IRRADIANCE_TEXTURE: u32 = 0u;
const BINDING_IBL_IRRADIANCE_SAMPLER: u32 = 1u;
const BINDING_IBL_PREFILTERED_TEXTURE: u32 = 2u;
const BINDING_IBL_PREFILTERED_SAMPLER: u32 = 3u;
const BINDING_IBL_BRDF_LUT_TEXTURE: u32 = 4u;
const BINDING_IBL_BRDF_LUT_SAMPLER: u32 = 5u;

// IBL constants
const MAX_REFLECTION_LOD: f32 = 4.0;
