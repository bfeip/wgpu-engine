# TODO

## Jots
- Geometry Query
    - Boundings
    - Ray casting
    - Volumes?
- Fix navigation operator to use bounding for sensitivity
- Add fit camera to bounding
- NURBS
- Selection
- Multiple lights
- Optimize drawing
- Proper API
- Normal maps
- Antialiasing
- Slang support

### Impl notes:
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