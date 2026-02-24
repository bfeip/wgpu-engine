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
### Rendering re-do (2 weeks)
Done Together, 4 days:
- Materials cleanup and fixes
- Shader refactor / modular shaders

One day each, 4 total:
- Rendering debug tools (wireframe, normal viewing, light visualization, etc.)
- Anti-aliasing
- Transparency
- Core API + docs review

With the remaining time:
- Draw order optimization
- IBL weirdness

### Crate improvements (2 weeks)
Three days together:
- format.rs and import export cleanup
- Scene API and docs review

One day each:
- format.rs and import export cleanup
- API and docs review
- Benchmarking and tests
- Camera space lights

One day together:
- Merge walk and orbit operators under navigation operator
- Everything optional behind features + good default features

Stretch:
- Scene graph expansion plan
- Performance
- High-level docs
- Better WASM API (a redesign of the react demo might be better)