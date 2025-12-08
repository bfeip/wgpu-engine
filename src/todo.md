# TODO

## Jots
- Re-work ray picking walker to use TreeWalker trait
- Ensure scene walkers are optimal
    - Ray selection uses transform and bounding walks. Combine.
- Fix navigation operator to use bounding for sensitivity

- Split mesh into DrawMesh and Mesh for API reasons
- Split materials into DrawMaterials and Materials for API reasons
- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Drag and drop GLTF loading operator
- Orthographic camera
- Highlighting and proper selection
- Instance flags and visibility
- Mesh wireframe
- Multiple lights
- Optimize drawing
- Normal maps
- Antialiasing
- NURBS

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

## Geom Query
I want to implement a geom query module to handle operations like picking instances, or
computing boundings.

The picking operations will be a series of functions where I can supply the scene and
an object to pick with, such as a ray, a point on the screen from which to cast a ray,
a convex hull, A frustum selected from a rectangle on the screen, etc. This will
involve a lot of rays, so I probably need to create a `Ray` class in `common.rs`.

For Boundings, I'll want to create a `TreeVisitor` that will walk the tree and
calculate the bounding for any given node.

### Step 1
Step one involves setting up simple ray picking. Given a ray and the scene, we should
be able to see if the ray intersects any instance geometry. This likely involves first
doing a bounding check, to see what instances are candidates for selection, then
checking the triangles to see if any of them are intersected by the ray.

### Step 2
Step two is where we do the ability to compute rays from a point on the screen. Then
we can write a simple selection operator, where we click on the screen and print
what instance was selected in the console, just to test.

### Step 3
Step 3 is selection by a convex hull. We will have to implement a hull struct.
Then we can implement picking via a hull. This likely involves seeing if a instance
bounding is fully contained within the hull, if it's fully outside the hull, or
if it's partially inside. Instances that have their boundings partially inside will
have to be checked to determine if any faces are intersecting

We could also do non-convex hull selection but that might be too complicated for
a first pass.

### Step 4
In step 4 we implement an operator for area selection on the screen. The user can
click and drag to make a rectangle that we will cast into the scene as a frustum
to determine the selection. Similar to the point on the screen component in step 2,
this will just print what was selected for now, later we can do more complex selection
behavior.

### Step 5
I want to finish off with bounding computations. We will compute boundings with a tree
walker and by examining the geometry, probably.