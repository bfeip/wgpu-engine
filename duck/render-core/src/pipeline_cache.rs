use std::collections::HashMap;
use std::hash::Hash;

/// A render pipeline cache keyed by an arbitrary caller-defined key.
///
/// The key type encodes whatever properties require a distinct compiled
/// pipeline (material variants, primitive topology, target formats, ...).
/// The core makes no assumptions about it.
pub struct PipelineCache<K: Eq + Hash> {
    map: HashMap<K, wgpu::RenderPipeline>,
}

impl<K: Eq + Hash> PipelineCache<K> {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Get the pipeline for `key`, building it via `build` on first use.
    pub fn get_or_create(
        &mut self,
        key: K,
        build: impl FnOnce() -> wgpu::RenderPipeline,
    ) -> &wgpu::RenderPipeline {
        self.map.entry(key).or_insert_with(build)
    }

    /// Discard all cached pipelines.
    ///
    /// Call when a parameter baked into the pipelines (sample count, target
    /// format, ...) changes, so they are recreated on next use.
    pub fn invalidate(&mut self) {
        self.map.clear();
    }
}

impl<K: Eq + Hash> Default for PipelineCache<K> {
    fn default() -> Self {
        Self::new()
    }
}
