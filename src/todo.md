# TODO

## Jots
- Proper API
- Normal maps
- Multiple lights
- Scene from GLTF
- Slang support

## Material Classes
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

## Don't forget
- Initialize light buffer with light data