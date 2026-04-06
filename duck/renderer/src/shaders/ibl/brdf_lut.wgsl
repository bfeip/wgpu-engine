// Compute shader to generate a BRDF integration lookup table.
//
// This 2D LUT is used in the split-sum approximation for specular IBL.
// X-axis: N·V (view angle cosine)
// Y-axis: roughness
// Output: (scale, bias) for Fresnel term: F0 * scale + bias

const PI: f32 = 3.14159265359;
const SAMPLE_COUNT: u32 = 1024u;

// Rg16Float doesn't support storage binding on all platforms, use Rgba16Float
@group(0) @binding(0) var output_lut: texture_storage_2d<rgba16float, write>;

// Radical inverse using Van der Corput sequence
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

// Hammersley point set
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// GGX importance sampling
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;

    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}

// Schlick-GGX geometry function
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    // Note: for IBL we use k = (roughness^2) / 2 instead of ((roughness+1)^2) / 8
    let a = roughness;
    let k = (a * a) / 2.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith geometry function
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx_v * ggx_l;
}

// Integrate BRDF for given N·V and roughness
fn integrate_brdf(n_dot_v: f32, roughness: f32) -> vec2<f32> {
    // View direction (in tangent space where N = (0, 0, 1))
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0; // Scale factor for F0
    var b = 0.0; // Bias factor

    for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let xi = hammersley(i, SAMPLE_COUNT);
        let h = importance_sample_ggx(xi, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);

        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v);
            let fc = pow(1.0 - v_dot_h, 5.0);

            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    return vec2<f32>(a, b) / f32(SAMPLE_COUNT);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_size = textureDimensions(output_lut);

    if gid.x >= output_size.x || gid.y >= output_size.y {
        return;
    }

    // Map pixel coordinates to N·V and roughness
    let n_dot_v = (f32(gid.x) + 0.5) / f32(output_size.x);
    let roughness = (f32(gid.y) + 0.5) / f32(output_size.y);

    // Clamp n_dot_v away from 0 to avoid division issues
    let clamped_n_dot_v = max(n_dot_v, 0.001);

    let result = integrate_brdf(clamped_n_dot_v, roughness);

    textureStore(output_lut, vec2<i32>(gid.xy), vec4<f32>(result, 0.0, 1.0));
}
