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
