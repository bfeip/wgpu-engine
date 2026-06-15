/// The GPU handle pair: device + queue.
///
/// Owned by whoever drives rendering; both `wgpu::Device` and `wgpu::Queue`
/// are internally reference-counted, so `Gpu` is cheaply cloneable.
#[derive(Clone)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Capabilities discovered while creating a headless [`Gpu`].
pub struct GpuCapabilities {
    pub has_compute: bool,
    pub backend: wgpu::Backend,
}

impl Gpu {
    /// Wrap pre-created device and queue (e.g. created alongside a surface).
    #[must_use] 
    pub const fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
    }

    /// Create a GPU without a surface, for offscreen/headless rendering.
    ///
    /// Creates its own wgpu instance and adapter. Useful for generating still
    /// images, thumbnails, or server-side rendering.
    /// 
    /// # Errors
    /// 
    /// Will return `Err` if initialization of WGPU fails.
    pub async fn headless() -> anyhow::Result<(Self, GpuCapabilities)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow::anyhow!("No suitable GPU adapter for headless rendering: {e}"))?;

        let backend = adapter.get_info().backend;
        let has_compute = backend != wgpu::Backend::Gl;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("Headless Renderer"),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await?;

        Ok((Self { device, queue }, GpuCapabilities { has_compute, backend }))
    }
}
