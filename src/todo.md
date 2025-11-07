# TODO

## Jots
- Geometry Query
    - Boundings
    - Ray casting
    - Volumes?
- NURBS
- Selection
- Multiple lights
- Optimize drawing
- Proper API
- Normal maps
- Antialiasing
- Slang support

## glTF support and scene layout
I want to work towards being able to lead scenes from GLTF. GLTF has a tree-like structure
so probably the first step for use to implement this is to do our scene tree. I've been
putting this off for as long as possible because I wanted to have a clear idea of what I'm
trying to accomplish before I write something so important. Bearing that in mind, I think
it's still a good idea to leave the scene tree as simple as possible. I've considered
approaches that involve octrees, but those are only really relevant if I have many
instances. With the simple scenes I'm working with right now I think I can keep it
simple and simply have a scene made of nodes, each node has a transform, and it can
contain instances. Instances reference the mesh they're an instance of. When drawing
the scene we walk the tree to compute the full transforms of the instances, determine if
they're in the view frustum (we can probably leave this for later), and gather the instances
according to material so their batches can be merged into the same draw call OR if there's
more than a certain number of instances (3 for example), instead of merging those by material
we draw them all together.

### Step 1:
Implement a scene tree with nodes. Each node will have a transform and contain either child
nodes or instances.

### Step 2:
Implement a scene walker that will walk the scene tree, gathering the transforms on nodes
so that we can compute the final transform on instances. The walker will sort the instances
into draw items, buffers to be sent to the GPU optimized to minimize the number of draw calls.

### Impl notes:
Ensure that the dirty flag is handled properly. Children of nodes that are marked dirty
will need to be updated as well.
DrawBatch currently requires a matching material and mesh for instances to be batched
together. However, we could do another type of batching for instances that share a
material but not a mesh, where the vertices and indices of the mesh are buffered
together and drawn together.
We currently create an instance buffer with the instance transforms every time we
draw. We should keep these buffers attached to the draw items so they can be reused.
I've hardcoded new camera values in the draw state as well as changing the minimum
zoom in radius. This should probably be undone as soon as possible.

## Don't forget
- Initialize light buffer with light data