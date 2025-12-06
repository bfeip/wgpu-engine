// main.wesl
// Main shader entry points with conditional compilation
// This is the single module that gets compiled with different feature flags

import package::common::{VertexInput, InstanceInput, VertexOutput};
import package::vertex::vertex_standard;
import package::fragment_color_lit::fragment_color_lit;
import package::fragment_color_unlit::fragment_color_unlit;
import package::fragment_texture_lit::fragment_texture_lit;

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
    // Conditional compilation based on material properties
    @if(has_texture && has_lighting) {
        return fragment_texture_lit(in);
    }

    @if(!has_texture && has_lighting) {
        return fragment_color_lit(in);
    }

    @if(!has_texture && !has_lighting) {
        return fragment_color_unlit(in);
    }

    // Fallback (should never reach here)
    return vec4<f32>(1.0, 0.0, 1.0, 1.0); // Magenta for error
}
