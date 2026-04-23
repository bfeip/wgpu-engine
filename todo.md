# TODO

## Jots
- Integration tests
  - Materials (including faces, lines, and points)
  - Lights
- Camera interpolation
- Scene cloning or multi-thread scene operations
- Scene merging, better high level operations
- Merge event dispatcher and operator manager into something like "interaction manager"
  - Input map so that all keybinds etc. are in one place.
- egui canvas frame
- Optimize drawing
- 3D Overlays
- Make operators optional feature
- Viewer trait(s)
  - Any number of viewers can implement the trait and do rendering
  - This will make writing a web viewer and keeping it consistent much easier
- geom query tests
- Better highlighting behavior

## glTF material extensions
Trivial (~1 day total):
- KHR_materials_unlit (just read the flag, shader path already exists)
- Emissive factor/texture (core glTF, not even an extension)
- Occlusion texture (core glTF, multiplies ambient only)
- KHR_materials_emissive_strength (scalar multiplier on emissive)

Harder but important:
- KHR_materials_ior (Fresnel F0 from IOR instead of hardcoded 0.04)
- KHR_materials_volume (colored glass absorption — requires transmission first)
- KHR_materials_clearcoat (second specular lobe — needs gltf crate upgrade)

## Tech debt
- load_gltf functions rename + docs update
- wgsl parser compilation issue