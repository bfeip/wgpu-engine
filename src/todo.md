# TODO

## Jots
- Fix navigation operator to use bounding for sensitivity

- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Example glTF loading
- Highlighting and proper selection
- Merge event dispatcher and operator manager into something like "interaction manager"
- egui canvas frame
- Instance flags and visibility
- Higher level faces and lines
- Mesh wireframe
- Multiple lights
- Optimize drawing
- PBR rendering
- Antialiasing
- NURBS
- 3D Overlays
- Make operators optional feature

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