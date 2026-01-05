// material_pbr.wesl
// PBR material uniform and texture bindings at group 2

import package::constants::{
    GROUP_MATERIAL,
    BINDING_PBR_PARAMS,
    BINDING_PBR_BASE_COLOR_TEXTURE,
    BINDING_PBR_BASE_COLOR_SAMPLER,
    BINDING_PBR_NORMAL_TEXTURE,
    BINDING_PBR_NORMAL_SAMPLER,
    BINDING_PBR_METALLIC_ROUGHNESS_TEXTURE,
    BINDING_PBR_METALLIC_ROUGHNESS_SAMPLER,
};

struct PbrParams {
    base_color_factor: vec4<f32>,
    metallic_factor: f32,
    roughness_factor: f32,
    normal_scale: f32,
    texture_flags: u32,
}

// Texture flag constants
const FLAG_HAS_BASE_COLOR_TEXTURE: u32 = 1u;
const FLAG_HAS_NORMAL_TEXTURE: u32 = 2u;
const FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE: u32 = 4u;

@group(GROUP_MATERIAL) @binding(BINDING_PBR_PARAMS)
var<uniform> pbr_params: PbrParams;

@group(GROUP_MATERIAL) @binding(BINDING_PBR_BASE_COLOR_TEXTURE)
var t_base_color: texture_2d<f32>;
@group(GROUP_MATERIAL) @binding(BINDING_PBR_BASE_COLOR_SAMPLER)
var s_base_color: sampler;

@group(GROUP_MATERIAL) @binding(BINDING_PBR_NORMAL_TEXTURE)
var t_normal: texture_2d<f32>;
@group(GROUP_MATERIAL) @binding(BINDING_PBR_NORMAL_SAMPLER)
var s_normal: sampler;

@group(GROUP_MATERIAL) @binding(BINDING_PBR_METALLIC_ROUGHNESS_TEXTURE)
var t_metallic_roughness: texture_2d<f32>;
@group(GROUP_MATERIAL) @binding(BINDING_PBR_METALLIC_ROUGHNESS_SAMPLER)
var s_metallic_roughness: sampler;

fn has_base_color_texture() -> bool {
    return (pbr_params.texture_flags & FLAG_HAS_BASE_COLOR_TEXTURE) != 0u;
}

fn has_normal_texture() -> bool {
    return (pbr_params.texture_flags & FLAG_HAS_NORMAL_TEXTURE) != 0u;
}

fn has_metallic_roughness_texture() -> bool {
    return (pbr_params.texture_flags & FLAG_HAS_METALLIC_ROUGHNESS_TEXTURE) != 0u;
}

// Get base color (samples texture if present, otherwise uses factor)
fn get_base_color(tex_coords: vec2<f32>) -> vec4<f32> {
    if has_base_color_texture() {
        return textureSample(t_base_color, s_base_color, tex_coords) * pbr_params.base_color_factor;
    } else {
        return pbr_params.base_color_factor;
    }
}

// Get metallic and roughness values
fn get_metallic_roughness(tex_coords: vec2<f32>) -> vec2<f32> {
    if has_metallic_roughness_texture() {
        // glTF spec: roughness in G channel, metallic in B channel
        let sample = textureSample(t_metallic_roughness, s_metallic_roughness, tex_coords);
        return vec2<f32>(
            sample.b * pbr_params.metallic_factor,
            sample.g * pbr_params.roughness_factor
        );
    } else {
        return vec2<f32>(pbr_params.metallic_factor, pbr_params.roughness_factor);
    }
}
