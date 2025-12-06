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

// Material bind group bindings
const BINDING_MATERIAL_COLOR: u32 = 0u;
const BINDING_MATERIAL_TEXTURE: u32 = 0u;
const BINDING_MATERIAL_SAMPLER: u32 = 1u;
