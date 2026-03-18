// ibl.wesl
// Image Based Lighting functions for PBR ambient lighting

import package::constants::{
    GROUP_ENVIRONMENT,
    BINDING_IBL_IRRADIANCE_TEXTURE,
    BINDING_IBL_IRRADIANCE_SAMPLER,
    BINDING_IBL_PREFILTERED_TEXTURE,
    BINDING_IBL_PREFILTERED_SAMPLER,
    BINDING_IBL_BRDF_LUT_TEXTURE,
    BINDING_IBL_BRDF_LUT_SAMPLER,
    MAX_REFLECTION_LOD
};
import package::pbr::fresnel_schlick_roughness;

// IBL textures and samplers
@group(GROUP_ENVIRONMENT) @binding(BINDING_IBL_IRRADIANCE_TEXTURE)
var ibl_irradiance: texture_cube<f32>;
@group(GROUP_ENVIRONMENT) @binding(BINDING_IBL_IRRADIANCE_SAMPLER)
var ibl_irradiance_sampler: sampler;
@group(GROUP_ENVIRONMENT) @binding(BINDING_IBL_PREFILTERED_TEXTURE)
var ibl_prefiltered: texture_cube<f32>;
@group(GROUP_ENVIRONMENT) @binding(BINDING_IBL_PREFILTERED_SAMPLER)
var ibl_prefiltered_sampler: sampler;
@group(GROUP_ENVIRONMENT) @binding(BINDING_IBL_BRDF_LUT_TEXTURE)
var ibl_brdf_lut: texture_2d<f32>;
@group(GROUP_ENVIRONMENT) @binding(BINDING_IBL_BRDF_LUT_SAMPLER)
var ibl_brdf_lut_sampler: sampler;

// Calculate IBL ambient lighting contribution
fn calculate_ibl(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_v = max(dot(normal, view_dir), 0.0);

    // F0 = reflectance at normal incidence
    // Dielectrics use 0.04, metals use albedo
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Fresnel term for IBL
    let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);

    // Energy conservation
    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    // Diffuse IBL: sample irradiance map with normal
    let irradiance = textureSample(ibl_irradiance, ibl_irradiance_sampler, normal).rgb;
    let diffuse = irradiance * albedo;

    // Specular IBL: sample pre-filtered environment at roughness-based mip level
    let reflect_dir = reflect(-view_dir, normal);
    let mip_level = roughness * MAX_REFLECTION_LOD;
    let prefiltered_color = textureSampleLevel(
        ibl_prefiltered,
        ibl_prefiltered_sampler,
        reflect_dir,
        mip_level
    ).rgb;

    // Sample BRDF LUT (x = N.V, y = roughness)
    let brdf = textureSample(ibl_brdf_lut, ibl_brdf_lut_sampler, vec2<f32>(n_dot_v, roughness)).rg;

    // Split-sum approximation for specular
    let specular = prefiltered_color * (f * brdf.x + brdf.y);

    return k_d * diffuse + specular;
}
