# TODO

## Jots
- Events and operators
- Screen trees and octrees
- Scene from GLTF
- Selection
- Multiple lights
- Optimize drawing
- Proper API
- Normal maps
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

## Don't forget
- Initialize light buffer with light data