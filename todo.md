# TODO

## Jots
- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Integration tests
  - Materials (including faces, lines, and points)
  - Lights
- Instance flags and visibility
- Import / export
- File format
- Camera interpolation
- Merge event dispatcher and operator manager into something like "interaction manager"
- egui canvas frame
- Higher level faces and lines
- Mesh wireframe
- Optimize drawing
- Antialiasing
- NURBS
- 3D Overlays
- Make operators optional feature

## Tech debt
- Renderer refactor needed. We should gather up all the Uniforms and GPU resources
and make them owned by the renderer module.
- load_gltf functions rename + docs update
- wgsl parser compilation issue

## Crate split

Extract the `scene` module into a GPU-free crate (`wgpu-engine-scene`) that can be
used independently for format conversion and other tools without requiring the renderer.

### New crate: `wgpu-engine-scene`

Contents:
- `scene/` - Mesh, Material, Texture, Node, Light, Annotation, etc. (GPU code removed)
- `common/` - Ray, Aabb, Plane, ConvexPolyhedron, transform utilities
- `geom_query/` - Picking queries (already GPU-free)
- `gltf.rs` - Asset loading (already GPU-free)

Dependencies: cgmath, bytemuck, image, gltf, anyhow, bitflags, obj-rs (NO wgpu)

### GPU code migration

Move all GPU resource management from scene types to the renderer:

1. Create `renderer/gpu_resources.rs` with:
   ```rust
   pub struct SceneGpuResources {
       mesh_buffers: HashMap<MeshId, MeshGpuBuffers>,
       textures: HashMap<TextureId, GpuTexture>,
       material_bind_groups: HashMap<(MaterialId, PrimitiveType), wgpu::BindGroup>,
       // Track what we've synced
       mesh_generations: HashMap<MeshId, u64>,
       texture_generations: HashMap<TextureId, u64>,
   }
   ```

2. Move from scene to renderer:
   - `mesh.rs`: `MeshGpuResources`, `ensure_gpu_resources()`, `draw_instances()`
   - `texture.rs`: `GpuTexture`, `ensure_gpu_resources()`, static GPU creation methods
   - `material.rs`: `MaterialGpuResources`, `bind()` method
   - `instance.rs`: `InstanceRaw::desc()`, `Vertex::desc()` → `renderer/pipeline.rs`

### Change tracking with generation numbers

Instead of GPU-specific dirty flags, use semantically clean generation counters:

```rust
// In scene crate - no GPU concepts
pub struct Mesh {
    vertices: Vec<Vertex>,
    generation: u64,  // Increments on any mutation
}

impl Mesh {
    pub fn generation(&self) -> u64 { self.generation }

    pub fn set_vertices(&mut self, vertices: Vec<Vertex>) {
        self.vertices = vertices;
        self.generation += 1;
    }
}
```

The renderer tracks last-synced generations and uploads when they differ.

### Implementation phases

**Phase 1: Refactor in-place**
1. Create `renderer/gpu_resources.rs` with SceneGpuResources
2. Move GPU code from mesh.rs, texture.rs, material.rs to renderer
3. Replace dirty flags with generation numbers
4. Update renderer to use new GPU resource manager
5. Verify everything still works

**Phase 2: Extract crate**
1. Create `scene/` directory with Cargo.toml
2. Move common/, scene/, geom_query/, gltf.rs to new crate
3. Update workspace Cargo.toml
4. Add wgpu-engine-scene as dependency to core

**Phase 3: Cleanup**
1. Update imports throughout core
2. Re-export from core for API compatibility
3. Delete old files from core
4. Verify: `cargo tree -p wgpu-engine-scene | grep wgpu` should be empty