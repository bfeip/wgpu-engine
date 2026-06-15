// material.wesl
// Unified surface material: params uniform plus optional textures at group 2.
//
// Texture bindings are compiled in only when the matching feature is set
// (`has_base_color_texture`, `has_normal_texture`, `has_metallic_roughness_texture`),
// so a variant declares exactly the bindings its bind-group layout provides —
// no fallback textures, no runtime texture flags.

import package::constants::{
    GROUP_MATERIAL,
    BINDING_MATERIAL_PARAMS,
    BINDING_MATERIAL_BASE_COLOR_TEXTURE,
    BINDING_MATERIAL_BASE_COLOR_SAMPLER,
    BINDING_MATERIAL_NORMAL_TEXTURE,
    BINDING_MATERIAL_NORMAL_SAMPLER,
    BINDING_MATERIAL_METALLIC_ROUGHNESS_TEXTURE,
    BINDING_MATERIAL_METALLIC_ROUGHNESS_SAMPLER,
};

struct MaterialParams {
    base_color_factor: vec4<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    normal_scale: f32,
    alpha_cutoff: f32,
}

@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_PARAMS)
var<uniform> material: MaterialParams;

@if(has_base_color_texture)
@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_BASE_COLOR_TEXTURE)
var t_base_color: texture_2d<f32>;
@if(has_base_color_texture)
@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_BASE_COLOR_SAMPLER)
var s_base_color: sampler;

@if(has_normal_texture)
@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_NORMAL_TEXTURE)
var t_normal: texture_2d<f32>;
@if(has_normal_texture)
@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_NORMAL_SAMPLER)
var s_normal: sampler;

@if(has_metallic_roughness_texture)
@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_METALLIC_ROUGHNESS_TEXTURE)
var t_metallic_roughness: texture_2d<f32>;
@if(has_metallic_roughness_texture)
@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_METALLIC_ROUGHNESS_SAMPLER)
var s_metallic_roughness: sampler;

// Base color: texture * factor when a base-color texture is bound, else the factor.
fn get_base_color(tex_coords: vec2<f32>) -> vec4<f32> {
    var color = material.base_color_factor;
    @if(has_base_color_texture) {
        color = textureSample(t_base_color, s_base_color, tex_coords) * material.base_color_factor;
    }
    return color;
}

// Metallic (x) and roughness (y). glTF packs roughness in G, metallic in B.
fn get_metallic_roughness(tex_coords: vec2<f32>) -> vec2<f32> {
    var mr = vec2<f32>(material.metallic_factor, material.roughness_factor);
    @if(has_metallic_roughness_texture) {
        let sample = textureSample(t_metallic_roughness, s_metallic_roughness, tex_coords);
        mr = vec2<f32>(sample.b * material.metallic_factor, sample.g * material.roughness_factor);
    }
    return mr;
}
