// main.wesl
// Main shader entry points with conditional compilation
// This is the single module that gets compiled with different feature flags

import package::common::{VertexInput, InstanceInput, VertexOutput};
import package::vertex::vertex_standard;

// Conditionally import fragment shaders based on lighting mode
// Lit materials use PBR (Cook-Torrance BRDF), unlit use flat color
@if(has_lighting)
    import package::fragment_pbr_lit::fragment_pbr_lit;
@if(!has_lighting)
    import package::fragment_color_unlit::fragment_color_unlit;

// Vertex shader entry point
@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    // All materials currently use the same vertex shader
    return vertex_standard(model, instance);
}

// Fragment shader entry point with conditional compilation
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    @if(has_lighting) {
        return fragment_pbr_lit(in);
    }

    @if(!has_lighting) {
        return fragment_color_unlit(in);
    }

    // Fallback (should never reach here)
    return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for error
}
