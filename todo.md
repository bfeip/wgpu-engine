# TODO

## Jots
- Fix navigation operator to use bounding for sensitivity

- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Example glTF loading
- Instance flags and visibility
- Camera interpolation
- Highlighting and proper selection
- Merge event dispatcher and operator manager into something like "interaction manager"
- egui canvas frame
- Higher level faces and lines
- Mesh wireframe
- Multiple lights
- Optimize drawing
- PBR rendering
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
- trunk debug WASM
- Update readme

## Impl notes:
- Ensure that the dirty flag is handled properly. Children of nodes that are marked dirty
will need to be updated as well.
- DrawBatch currently requires a matching material and mesh for instances to be batched
together. However, we could do another type of batching for instances that share a
material but not a mesh, where the vertices and indices of the mesh are buffered
together and drawn together.
- We currently create an instance buffer with the instance transforms every time we
draw. We should keep these buffers attached to the draw items so they can be reused.
- I've hardcoded new camera values in the draw state as well as changing the minimum
zoom in radius. This should probably be undone as soon as possible.