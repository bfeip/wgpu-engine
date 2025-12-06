// camera.wesl
// Camera uniform binding at group 0

import package::constants::{GROUP_CAMERA, BINDING_CAMERA_UNIFORM};
import package::common::CameraUniform;

@group(GROUP_CAMERA) @binding(BINDING_CAMERA_UNIFORM)
var<uniform> camera: CameraUniform;
