// material_texture.wesl
// Texture material bindings at group 2

use package::constants::{GROUP_MATERIAL, BINDING_MATERIAL_TEXTURE, BINDING_MATERIAL_SAMPLER};

@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_TEXTURE)
pub var t_diffuse: texture_2d<f32>;

@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_SAMPLER)
pub var s_diffuse: sampler;
