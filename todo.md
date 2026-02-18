# TODO

## Jots
- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Integration tests
  - Materials (including faces, lines, and points)
  - Lights
- Camera interpolation
- Scene cloning or multi-thread scene operations
- Scene merging, better high level operations
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
- load_gltf functions rename + docs update
- wgsl parser compilation issue

## March quality push
- Materials cleanup and fixes
- Rendering debug tools (wireframe, normal viewing, light visualization, etc.)
- format.rs and import export cleanup
- IBL weirdness
- High-level docs
- Better WASM API
- Anti-aliasing?
- Transparency?
- More benchmarking and tests
- Performance
- API and docs review
- Merge walk and orbit operators under navigation operator
- Shader refactor / modular shaders