use crate::{FrameTargets, Gpu};

/// A *frame family*: a type-level tag naming the per-frame data type for one
/// rendering stack.
///
/// # Why this trait exists
///
/// A [`RenderWorkflow`] reads per-frame data that borrows (the scene, bind
/// groups, caches), so that data type carries a lifetime — conceptually
/// `Frame<'a>`. We *also* want workflows to be runtime-swappable and
/// user-extensible, i.e. held as `Box<dyn RenderWorkflow<…>>`.
///
/// Those requirements collide. The natural form,
///
/// ```ignore
/// trait RenderWorkflow { type Frame<'a>; fn execute(&mut self, f: &mut Self::Frame<'_>); }
/// ```
///
/// is not `dyn`-compatible: a trait with a *generic associated type* cannot be
/// made into a trait object, so it could never be boxed. `FrameFamily` lifts the
/// lifetime-parameterized type out into its own trait, leaving `RenderWorkflow`
/// with a plain type parameter `F` — and `dyn RenderWorkflow<F>` *is*
/// `dyn`-compatible. This is the standard stable-Rust workaround for "I need a
/// trait object whose associated type has a lifetime."
///
/// # Implementing
///
/// An implementer is a pure type-level token: it is **never constructed** and
/// carries no data, so an uninhabited enum is the natural choice.
/// [`RenderHost<F>`](crate::RenderHost) uses `F` only to name `F::Frame<'_>`;
/// no value of `F` ever exists at runtime.
///
/// ```
/// use duck_engine_render_core::FrameFamily;
///
/// struct MyFrameData<'a> { scene: &'a str }
///
/// enum MyFrames {}
/// impl FrameFamily for MyFrames {
///     type Frame<'a> = MyFrameData<'a>;
/// }
/// // Workflows for this stack then implement `RenderWorkflow<MyFrames>`.
/// ```
pub trait FrameFamily: 'static {
    /// The per-frame data type, parameterized by the frame's borrow lifetime.
    ///
    /// The core never inspects it — its contents are defined entirely by the
    /// rendering stack built on top. A custom stack can put anything here,
    /// including `()` for a frame that borrows nothing.
    type Frame<'a>;
}

/// A named rendering strategy that owns all passes and GPU resources for one
/// frame style.
///
/// Workflows are the top-level unit of render customization. A
/// [`RenderHost`](crate::RenderHost) runs exactly one workflow per frame, and
/// workflows can be swapped at runtime via
/// [`RenderHost::set_workflow`](crate::RenderHost::set_workflow).
///
/// The type parameter `F` selects the [`FrameFamily`] — i.e. which per-frame
/// data type `execute` receives. Because `F` is a plain type parameter (the
/// lifetime lives on `F::Frame<'a>`, not on this trait), `dyn RenderWorkflow<F>`
/// is `dyn`-compatible: workflows sharing a frame family are runtime-swappable
/// in a `Box<dyn RenderWorkflow<F>>`.
pub trait RenderWorkflow<F: FrameFamily>: 'static {
    fn name(&self) -> &'static str;

    /// Called after a viewport resize. Implementations must forward to any
    /// passes that own size-dependent GPU resources.
    fn resize(&mut self, _gpu: &Gpu, _targets: &FrameTargets) {}

    /// Execute all passes for this frame.
    ///
    /// `targets` is passed separately rather than carried inside the frame so
    /// that the host can lend its own attachments while the caller retains an
    /// independently-built frame — the borrow split that lets `&mut host`
    /// coexist with frame data borrowed from the caller's other fields.
    fn execute(
        &mut self,
        gpu: &Gpu,
        targets: &FrameTargets,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        frame: &mut F::Frame<'_>,
    );
}
