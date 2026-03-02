# TODO

## Jots
- Tree walker feature to skip subtrees on condition
- Ability to walk the scene with multiple walkers at once

- Integration tests
  - Materials (including faces, lines, and points)
  - Lights
- Camera interpolation
- Scene cloning or multi-thread scene operations
- Scene merging, better high level operations
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
- load_gltf functions rename + docs update
- wgsl parser compilation issue

## March quality push
### Rendering re-do (2 weeks)
Done Together, 4 days:
- Materials cleanup and fixes
- Shader refactor / modular shaders

One day each, 4 total:
- Rendering debug tools (wireframe, normal viewing, light visualization, etc.)
- Anti-aliasing
- Transparency
- Core API + docs review

With the remaining time:
- Draw order optimization
- IBL weirdness

### Crate improvements (2 weeks)
Three days together:
- format.rs and import export cleanup
- Scene API and docs review

One day each:
- format.rs and import export cleanup
  - Split import export
  - trait-ification of importers + exporters
- API and docs review
- Benchmarking and tests
- Camera space lights

One day together:
- Merge walk and orbit operators under navigation operator
- Everything optional behind features + good default features

Stretch:
- Scene graph expansion plan
- Extract egui example to replace glTF viewer
- Performance
- High-level docs
- Better WASM API (a redesign of the react demo might be better)

### Material shader re-design
We need to plan for a refactor or re-design of our material and shader systems. The problem is that currently the system is rather inflexible: e.g. PBR materials must have textures associated with them in order to render correctly, despite base color + metalness roughness factors being a valid workflow.

From what I've seen, the materials themselves in `material.rs` look fine, it's more or less a POD struct that has all the fields we need. My problem is moreso I think with the shaders and how shaders are generated from materials. in `generate_shader` in `shaders.rs` and in `main.wesl` is where I think most of the problems are. These two combined force the shader down one of 4 constrictive pathways (which really just boil down to 2, PBR lit, and everything else).

I'd especially like to do away with the `use_pbr` shader feature. I think it implies a false dichotomy between how our materials work. This was introduced because we were going to also support blinn-phong materials and I wanted the workflows separate. However, the reality of the situation is that we should render the material as best as possible using the Cook-Torrance workflow we've developed. We can introduce other rendering models later.

Please do an analysis on how to make the material and shader functionality less restrictive
and more cleanly implemented