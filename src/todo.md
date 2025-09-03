# TODO

## Jots
- Proper API
- Normal maps
- Multiple lights
- Scene from GLTF
- Slang support

## Material classes
There are several material classes for different kinds of materials. E.g. There
is a ColorMaterial struct for materials with a simple diffuse color,
TexturedMaterial for materials with diffuse and possibly other textures.

These should be managed by a MaterialManager. Each material is assigned an ID
with the MaterialManager.

The bind group layouts are a bit tricky because they need to be shared between
the bind groups themselves and also the render pipeline layout. I think maybe
the MaterialManager or maybe the DrawState should own them.

The pipeline layouts need to be owned by the DrawState since they combine bind group
layouts both from materials and parts of the scene (Camera, Lights)

We probably only need one render pipeline at a time and do not need to store a pipeline
for each material. All of the material specific information for a pipeline is in the
layout and the bind groups of the materials.

## glTF support and scene layout
I want to implement a layout for scenes and I'm looking at glTF as a reference.
I think most objects will have IDs, e.g. MeshIds, MaterialIds, and InstanceIds.
One thing I'm worried about right now is that the concept of an instance is
contained as part of a mesh. Mesh objects own their instances, I think when we have a scene
That's going to change. I did that for a good reason, because we want to draw instances
of the same mesh with the same material together. But I think what I'll need to do
instead is have the concepts separated and then walk the scene tree looking for instances
that we want to draw, and _then_ grouping them by what's best to draw together.

### Scene Refactor for glTF Support - Priority Order

#### 1. Separate Instance Management from Mesh (High Priority)
- Create a new `InstanceManager` or similar struct to own all instances
- Remove `instances` HashMap from `Mesh` struct  
- Create separate `MeshId`, `MaterialId`, and `InstanceId` types for proper referencing
- Update `Instance` struct to reference mesh by ID instead of raw pointer

#### 2. Implement Scene Tree Structure (High Priority) 
- Create a proper scene graph/tree structure to represent hierarchical transforms
- Add `Node` struct with parent/child relationships and local transforms
- Update `Scene` to contain a tree of nodes rather than flat mesh list
- Each node can reference meshes, materials, lights, or be empty transform nodes

#### 3. Create Scene Walker/Collector (Medium Priority)
- Implement function to traverse scene tree and collect all instances to render
- Group instances by mesh + material combination for efficient batched rendering
- Calculate world transforms by walking up the scene hierarchy
- Replace current direct mesh iteration in render loop

#### 4. Update Rendering Pipeline (Medium Priority)
- Modify `DrawState::render()` to use the scene walker
- Update instance buffer management to handle dynamically collected instances
- Ensure proper transform calculations for hierarchical scenes

#### 5. Implement Basic glTF Loading (Medium Priority)
- Extend `GltfParser` to actually load scenes (not just dump)
- Map glTF nodes to your scene tree structure
- Load meshes, materials, and transform hierarchies from glTF files
- Test with simple glTF files first

#### 6. Optimize Instance Rendering (Low Priority)
- Pre-compute and cache instance buffers instead of recreating each frame
- Implement dirty flagging for when scene tree changes
- Add culling and LOD support for large scenes

## Don't forget
- Initialize light buffer with light data