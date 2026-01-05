// constants.wesl
// Shader constants for bind groups and bindings

// Bind group indices
const GROUP_CAMERA: u32 = 0u;
const GROUP_LIGHTS: u32 = 1u;
const GROUP_MATERIAL: u32 = 2u;

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
