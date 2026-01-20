# TODO

## Jots
- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Integration tests
  - Materials (including faces, lines, and points)
  - Lights
- Instance flags and visibility
- Camera interpolation
- Highlighting and proper selection
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
- The annotation manager should be made a part of the scene, but not modify nodes
directly. They'll be reified when the scene is drawn or something.
- The scene should have better dirty state management. I think currently the states
are all set at once or something like that.
- load_gltf functions rename + docs update
- wgsl parser compilation issue
