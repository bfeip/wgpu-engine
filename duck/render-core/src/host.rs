use crate::{
    FrameFamily, FrameTargets, Gpu, ReadbackTarget, RenderWorkflow, TargetConfig, TargetFeatures,
};

/// Tightly-packed RGBA8 pixel data read back from the GPU.
pub struct RgbaPixels {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Owns the frame-agnostic per-frame machinery: the GPU handles, the shared
/// frame targets, the active workflow, and an optional readback target for
/// headless rendering. Knows nothing about what a frame contains.
///
/// The caller builds an `F::Frame<'_>` from its own state each frame and hands
/// it to [`render`](Self::render). Because the frame is built from the
/// *caller's* fields and the host borrows only its own, `&mut host` and the
/// frame coexist without conflict — the borrow-split this design preserves.
pub struct RenderHost<F: FrameFamily> {
    gpu: Gpu,
    targets: FrameTargets,
    workflow: Box<dyn RenderWorkflow<F>>,
    /// Cached for headless rendering, reused across frames at the same size.
    readback: Option<ReadbackTarget>,
}

impl<F: FrameFamily> RenderHost<F> {
    pub fn new(
        gpu: Gpu,
        config: TargetConfig,
        features: TargetFeatures,
        workflow: Box<dyn RenderWorkflow<F>>,
    ) -> Self {
        let targets = FrameTargets::new(&gpu, config, features);
        Self { gpu, targets, workflow, readback: None }
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn targets(&self) -> &FrameTargets {
        &self.targets
    }

    pub fn config(&self) -> TargetConfig {
        self.targets.config()
    }

    /// Replace the active rendering workflow.
    ///
    /// The new workflow takes effect immediately on the next frame. The
    /// previous workflow and all its GPU resources are dropped.
    pub fn set_workflow(&mut self, workflow: Box<dyn RenderWorkflow<F>>) {
        self.workflow = workflow;
    }

    /// Recreate size-dependent attachments and forward to the workflow.
    /// Ignores zero dimensions.
    pub fn resize(&mut self, size: (u32, u32)) {
        if size.0 == 0 || size.1 == 0 {
            return;
        }
        self.targets.resize(&self.gpu, size);
        self.workflow.resize(&self.gpu, &self.targets);
        self.readback = None;
    }

    /// Execute the active workflow for one frame. The encoder is not
    /// submitted — the caller is responsible for that.
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut F::Frame<'_>,
    ) {
        self.workflow
            .execute(&self.gpu, &self.targets, encoder, view, frame);
    }

    /// Render one frame into an owned offscreen target and read it back.
    ///
    /// Creates and submits its own encoder, then blocks until the GPU work
    /// completes. The readback target is cached and reused across calls at
    /// the same size. Assumes a 4-byte-per-pixel target format.
    pub fn render_to_rgba(&mut self, frame: &mut F::Frame<'_>) -> anyhow::Result<RgbaPixels> {
        let (width, height) = self.targets.size();

        if self
            .readback
            .as_ref()
            .is_none_or(|r| r.size() != (width, height))
        {
            self.readback = Some(ReadbackTarget::new(
                &self.gpu.device,
                width,
                height,
                self.targets.format(),
            ));
        }
        let target = self.readback.take().unwrap();

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Readback Render Encoder"),
            });
        self.render(&mut encoder, target.view(), frame);
        target.encode_copy(&mut encoder);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let data = target.read(&self.gpu.device)?;
        self.readback = Some(target);

        Ok(RgbaPixels { width, height, data })
    }
}
