// material_color.wesl
// Color material binding at group 2

import package::constants::{GROUP_MATERIAL, BINDING_MATERIAL_COLOR};

@group(GROUP_MATERIAL) @binding(BINDING_MATERIAL_COLOR)
var<uniform> material_color: vec4<f32>;
