// surface.wesl
// Single configurable surface shader entry point.
//
// One module compiled with feature flags drives every material variant:
//   lit, has_base_color_texture, has_normal_texture,
//   has_metallic_roughness_texture, has_ibl, double_sided, alpha_mask, depth_prepass.
// The unlit path returns the base color directly; the lit path runs PBR shading.

import package::common::{VertexInput, InstanceInput, VertexOutput};
import package::vertex::vertex_standard;
import package::material::{material, get_base_color};

// Lit shading lives in its own module so the PBR machinery and its group-3 IBL
// bindings are only pulled in (and only declared) for lit variants.
@if(lit)
    import package::lit_surface::shade_lit;

// Vertex shader entry point
@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    return vertex_standard(model, instance);
}

// Fragment shader entry point
@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let base_color = get_base_color(in.tex_coords);

    // Alpha mask: discard below the cutoff. The depth prepass uses a high fixed
    // cutoff so only nearly-opaque fragments of blend materials write depth;
    // regular mask rendering uses the material's own cutoff.
    @if(alpha_mask) {
        @if(depth_prepass) {
            if (base_color.a < 0.95) {
                discard;
            }
        }
        @if(!depth_prepass) {
            if (base_color.a < material.alpha_cutoff) {
                discard;
            }
        }
    }

    @if(!lit) {
        return base_color;
    }
    @if(lit) {
        return shade_lit(in, front_facing, base_color);
    }

    // Unreachable: exactly one of lit / !lit is compiled in.
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
