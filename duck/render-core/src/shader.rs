use std::borrow::Cow;
use std::collections::HashMap;

use wesl::{ModulePath, VirtualResolver, Wesl};
use wgpu::ShaderModuleDescriptor;

/// Cache key: (root module path, feature flags sorted by name).
type ShaderKey = (String, Vec<(String, bool)>);

/// A WESL module library with feature-flagged compilation and module caching.
///
/// Holds a set of named WESL modules (typically `&'static str` sources embedded
/// at compile time) and compiles any of them to a [`wgpu::ShaderModule`] with a
/// given set of feature flags. Compiled modules are cached by
/// (root module, feature set), so repeated requests for the same variant are
/// free.
///
/// The library is agnostic to what the modules contain — shader semantics
/// (materials, lighting, ...) belong to the crate that supplies the sources.
///
/// # Example
///
/// ```
/// use duck_engine_render_core::ShaderLibrary;
///
/// let mut library = ShaderLibrary::new([(
///     "package::tint",
///     "@if(red) const TINT: f32 = 7.0;
///      @if(!red) const TINT: f32 = 0.0;
///      @fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(TINT); }",
/// )]);
///
/// let wgsl = library.compile_to_wgsl("package::tint", &[("red", true)]).unwrap();
/// assert!(wgsl.contains("7.0"));
/// ```
pub struct ShaderLibrary {
    /// (module path, source) pairs, kept for building fresh ad-hoc resolvers.
    modules: Vec<(&'static str, &'static str)>,
    compiler: Wesl<VirtualResolver<'static>>,
    cache: HashMap<ShaderKey, wgpu::ShaderModule>,
}

impl ShaderLibrary {
    /// Create a library from (module path, WESL source) pairs,
    /// e.g. `("package::camera", include_str!("shaders/camera.wesl"))`.
    pub fn new(modules: impl IntoIterator<Item = (&'static str, &'static str)>) -> Self {
        let modules: Vec<_> = modules.into_iter().collect();
        let compiler = Wesl::new(".").set_custom_resolver(build_resolver(&modules));
        Self {
            modules,
            compiler,
            cache: HashMap::new(),
        }
    }

    /// Compile `root` with the given feature flags into a shader module,
    /// returning the cached module if this variant was compiled before.
    pub fn compile(
        &mut self,
        device: &wgpu::Device,
        root: &str,
        features: &[(&str, bool)],
        label: &str,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        let key = cache_key(root, features);
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        let wgsl = self.compile_to_wgsl(root, features)?;
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        self.cache.insert(key, module.clone());
        Ok(module)
    }

    /// Compile `root` with the given feature flags to WGSL source text.
    ///
    /// Used internally by [`compile`](Self::compile); exposed so callers (and
    /// tests) can inspect generated WGSL without a GPU device. Not cached.
    pub fn compile_to_wgsl(
        &mut self,
        root: &str,
        features: &[(&str, bool)],
    ) -> anyhow::Result<String> {
        let path: ModulePath = root.parse()?;
        self.compiler.set_features(features.iter().copied());
        let result = self.compiler.compile(&path)?;
        Ok(result.to_string())
    }

    /// Compile a caller-supplied WESL source with access to all library modules.
    ///
    /// The source is registered as `package::user`, so `package::` imports in it
    /// resolve against this library's modules. A fresh resolver is built each
    /// call so the persistent compiler's cached state and feature flags do not
    /// bleed into user compilation; the cost is negligible (a handful of
    /// HashMap inserts of `&'static str` pointers).
    pub fn compile_adhoc(
        &self,
        device: &wgpu::Device,
        source: &str,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        let mut resolver = build_resolver(&self.modules);
        resolver.add_module("package::user".parse()?, Cow::Owned(source.to_owned()));

        let mut compiler = Wesl::new(".").set_custom_resolver(resolver);
        let path: ModulePath = "package::user".parse()?;
        let empty_features: [(&str, bool); 0] = [];
        compiler.set_features(empty_features);
        let result = compiler.compile(&path)?;
        let wgsl = result.to_string();

        Ok(device.create_shader_module(ShaderModuleDescriptor {
            label: Some("User WESL Shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        }))
    }
}

fn build_resolver(modules: &[(&'static str, &'static str)]) -> VirtualResolver<'static> {
    let mut resolver = VirtualResolver::default();
    for (path, source) in modules {
        resolver.add_module(path.parse().expect("invalid WESL module path"), (*source).into());
    }
    resolver
}

fn cache_key(root: &str, features: &[(&str, bool)]) -> ShaderKey {
    let mut features: Vec<(String, bool)> = features
        .iter()
        .map(|(name, on)| (name.to_string(), *on))
        .collect();
    features.sort();
    (root.to_string(), features)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIB_MODULE: &str = "
        const SHARED: f32 = 42.0;
    ";

    const ROOT_MODULE: &str = "
        import package::lib::SHARED;

        @if(extra)
        fn factor() -> f32 { return SHARED; }

        @if(!extra)
        fn factor() -> f32 { return 0.5; }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(factor());
        }
    ";

    fn library() -> ShaderLibrary {
        ShaderLibrary::new([("package::lib", LIB_MODULE), ("package::root", ROOT_MODULE)])
    }

    #[test]
    fn features_gate_compiled_output() {
        let mut lib = library();
        let with = lib.compile_to_wgsl("package::root", &[("extra", true)]).unwrap();
        let without = lib.compile_to_wgsl("package::root", &[("extra", false)]).unwrap();
        assert!(with.contains("42"));
        assert!(!without.contains("42"));
        assert!(without.contains("0.5"));
    }

    #[test]
    fn imports_resolve_across_modules() {
        // SHARED comes from package::lib; reaching the output proves the import resolved.
        let mut lib = library();
        let wgsl = lib.compile_to_wgsl("package::root", &[("extra", true)]).unwrap();
        assert!(wgsl.contains("42"));
    }

    #[test]
    fn cache_key_is_order_insensitive() {
        let a = cache_key("package::root", &[("x", true), ("y", false)]);
        let b = cache_key("package::root", &[("y", false), ("x", true)]);
        assert_eq!(a, b);
    }
}
