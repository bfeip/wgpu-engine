# TODO

## Jots
- Events and operators
- NURBS
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

## Operators
I'm going to be adding Operators which will facilitate interaction with the 3d scene.
To begin with, I'll add the base Operator trait and the OperatorManager which will hold
a priority queue of objects that implement the Operator trait. These operators will
interact with the recently developed Event system. When an operator is added onto the
OperatorManager's priority queue, the operators methods will have to link up to the
EventDispatcher via a callback created by the operator.

This will require some changes to Events and the EventDispatcher, firstly since the
operators exist in a priority queue that can be rearranged at runtime depending on what
operators the user is using, the callback order that the EventDispatcher dispatches to
will have to be similarly re-worked so that callbacks can be recognized, and removed
or re-ordered during runtime. Likely every callback will be assigned an ID when it's
registered with the dispatcher, then that ID can later be used to un-register the
callback.

Another thing we will have to change in the event system is that event callbacks will
likely want to modify data in their operators. For example, an operator that tracks 
mouse drags for camera movement will first have to listen for mouse down events, then
after a key down happens it will listen to mouse movement to move the camera. The 
dragging ends when a mouse button up event occurs. In order to do this we may wand to
find a way to attach additional context to an EventCallback. I'm not sure exactly how
to do that in Rust though.

The only operator I might do in this first pass might be the navigation operator.
Selection is the other operator I really want but I think I should have a way to do
ray casting before I start work on that. That kind of opens a whole door into deeper
geometry query questions.

## Don't forget
- Initialize light buffer with light data