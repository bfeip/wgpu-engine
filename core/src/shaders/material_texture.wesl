// material_texture.wesl
// Texture material bindings at group 2

import package::constants::{GROUP_MATERIAL, BINDING_MATERIAL_TEXTURE, BINDING_MATERIAL_SAMPLER};

@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_TEXTURE)
var t_diffuse: texture_2d<f32>;

@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_SAMPLER)
var s_diffuse: sampler;
