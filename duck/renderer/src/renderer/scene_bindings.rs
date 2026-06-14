use bytemuck::bytes_of;

use crate::scene::PositionedCamera;
use crate::scene::Scene;

use super::gpu_resources::{
    BindGroupLayouts, CameraResources, CameraUniform, LightResources, LightsArrayUniform,
};

/// The persistent scene-level bind groups owned by the renderer: the camera
/// (group 0) and the lights array (group 1).
///
/// These are uniform buffers the renderer fills every frame, so it owns them
/// outright. IBL (group 3) is deliberately *not* here: that bind group lives in
/// [`IblResources`](crate::ibl), keyed per environment map and rebuilt by GPU
/// preprocessing, so it can only be borrowed per frame — it joins camera/lights
/// in [`SceneBindingRefs`], the per-frame view assembled in
/// [`refs`](Self::refs).
pub(crate) struct SceneBindings {
    camera: CameraResources,
    lights: LightResources,
}

impl SceneBindings {
    pub fn new(device: &wgpu::Device, layouts: &BindGroupLayouts) -> Self {
        Self {
            camera: CameraResources::new(device, &layouts.camera),
            lights: LightResources::new(device, &layouts.light),
        }
    }

    /// Write `camera` into the shared camera uniform buffer.
    pub fn write_camera(&self, queue: &wgpu::Queue, camera: &PositionedCamera) {
        let uniform = [CameraUniform::from_positioned_camera(camera)];
        queue.write_buffer(&self.camera.buffer, 0, bytemuck::cast_slice(&uniform));
    }

    /// Re-upload the lights uniform if the scene's node generation has changed
    /// since the last sync. `resolve` supplies the resolved lights for the
    /// current frame (the renderer gathers them from the scene graph), and is
    /// only called when an upload is actually needed.
    pub fn sync_lights(
        &mut self,
        queue: &wgpu::Queue,
        scene: &Scene,
        resolve: impl FnOnce() -> LightsArrayUniform,
    ) {
        let node_gen = scene.node_generation();
        if self.lights.synced_generation != node_gen {
            queue.write_buffer(&self.lights.buffer, 0, bytes_of(&resolve()));
            self.lights.synced_generation = node_gen;
        }
    }

    /// Force the next [`sync_lights`](Self::sync_lights) to re-upload, e.g. after
    /// the scene's GPU resources are cleared.
    pub fn invalidate_lights(&mut self) {
        self.lights.synced_generation = 0;
    }

    /// Assemble the per-frame view of all scene-level bind groups: the owned
    /// camera/lights plus the borrowed, optional IBL group for this frame.
    pub fn refs<'a>(&'a self, ibl: Option<&'a wgpu::BindGroup>) -> SceneBindingRefs<'a> {
        SceneBindingRefs {
            camera: &self.camera.bind_group,
            lights: &self.lights.bind_group,
            ibl,
        }
    }
}

/// Per-frame references to the scene-level bind groups (camera, lights, IBL).
///
/// Bundles the three standard scene groups so passes can bind them together.
/// Group indices follow the standard shader ABI ([`crate::abi`]); a pass chooses
/// whether and where to bind each.
#[derive(Clone, Copy)]
pub struct SceneBindingRefs<'a> {
    pub camera: &'a wgpu::BindGroup,
    pub lights: &'a wgpu::BindGroup,
    /// `Some` if there is an active, fully-processed environment map for IBL.
    pub ibl: Option<&'a wgpu::BindGroup>,
}
