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
- Shader refactor / modular shaders
- Rendering debug tools (wireframe, normal viewing, light visualization, etc.)
- format.rs and import export cleanup
- Everything optional behind features + good default features
- IBL weirdness
- High-level docs
- Better WASM API
- Anti-aliasing?
- Transparency?
- More benchmarking and tests
- Performance
- API and docs review
- Merge walk and orbit operators under navigation operator
- Everything optional behind features + good default features

Stretch:
- Scene graph expansion plan
- Extract egui example to replace glTF viewer
- Performance
- High-level docs
- Better WASM API (a redesign of the react demo might be better)