// camera.wesl
// Camera uniform binding at group 0

use package::constants::{GROUP_CAMERA, BINDING_CAMERA_UNIFORM};
use package::common::CameraUniform;

@group(GROUP_CAMERA) @binding(BINDING_CAMERA_UNIFORM)
pub var<uniform> camera: CameraUniform;
