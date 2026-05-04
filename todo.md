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
- Further separate Viewer from scene and renderer
  - Renderer and scene should be completely application agnostic
  - Viewer is CAD oriented.

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

## May Scene Platform Overhaul

**Goal:** Transform scene into an extensible platform for advanced applications — streaming viewers,
CAD integration, and network collaboration. This is a significant breaking restructuring; the work
below supersedes the existing node, light, and camera APIs.

### Phase 1 — Polymorphic Node Types - DONE

Replace the single monolithic `Node` (which carries `Option<InstanceId>`) with a typed payload model.

**NodePayload variants:**
- `Branch` — structural container, no content (current "empty" nodes)
- `Instance(InstanceId)` — current behavior, references a mesh+material pair
- `Camera(CameraData)` — lens and projection parameters, no transform data (that's the node's transform)
- `PointLight(...)`, `DirectionalLight(...)`, `SpotLight(...)` — replaces the flat `Vec<Light>` in Scene
- `Custom(Box<dyn CustomNodePayload>)` — external crates plug in here

**Scene changes:**
- Remove `lights: Vec<Light>` — replaced by light nodes in the tree
- Remove `views: HashMap<ViewId, View>` — replaced by camera nodes in the tree
- Add `active_camera: Option<NodeId>` — which camera node drives rendering
- Generation tracking must cover payload mutations, not only transform/visibility changes
- `Light::CoordinateSpace` eliminated — camera-relative lights achieved by parenting a light node under a camera node

**Renderer changes:**
- Traverse tree to collect light nodes instead of reading `scene.lights()`
- `render_scene_to_view()` accepts an optional explicit `&Camera` override; falls back to the active camera node's transform + parameters
- Viewer's navigation operators act on either the freeform interactive camera or a selected camera node's transform

---

### Phase 2 — Custom Node Type Registry

Allow external crates to register node types that the scene can hold, serialize, and stream.

- `CustomNodePayload` trait: `type_tag() -> &'static str`, `to_bytes() -> Vec<u8>`, optional bounds hook
- `NodeTypeRegistry`: maps type tag strings → `fn(&[u8]) -> Result<Box<dyn CustomNodePayload>>`
- Registry populated at app init; CAD crate exposes `register_cad_node_types(&mut NodeTypeRegistry)`
- Scene or app context holds the registry; passed as context during deserialization and streaming
- **Unknown type tags must round-trip safely:** preserve raw bytes so a node the local app doesn't understand isn't destroyed on save/forward

---

### Phase 3 — Stable Resource Identity - DONE

Replace all auto-increment `u32` ID types (`NodeId`, `MeshId`, `MaterialId`, `TextureId`, `InstanceId`) with UUIDs.

- Current counters are local to a single `Scene` instance and fall apart on scene merge, streaming, and file round-trips; `format.rs` already works around this with an ID remapping step on load
- Use UUID v7 (time-ordered): globally unique, no server/client coordination needed, natural creation-order sort without implying structural ordering
- The `uuid` crate supports v7; IDs become `u128` under the hood — fast to compare and hash, negligible cost at scene sizes
- `format.rs` ID remapping step is eliminated; saved IDs are stable across load/save cycles
- `SceneEvent` references in Phase 4 streaming are now unambiguous across sessions and between peers

---

### Phase 4 — DUCK Format v0.3

Evolve the binary scene format to handle typed payloads.

- `Nodes` section: each node carries `payload_type_tag: String` + `payload_bytes: Vec<u8>`
- Built-in types (Instance, Camera, lights) use their own compact bincode layouts under well-known tags
- Remove `Lights` and `Views` top-level sections — these are now just nodes
- Add `ActiveCamera` section: `Option<NodeId>`
- v0.2 backward compatibility: on read, old light entries → light nodes, old views → camera nodes
- Custom payload serialization is opaque to the format — scene format carries bytes, type owner provides meaning

---

### Phase 5 — Network Streaming

Enable servers to stream scene subtrees to connected viewers.

- `SceneEvent` enum: NodeAdded, NodeRemoved, NodeTransformChanged, NodePayloadChanged,
  NodeVisibilityChanged, ResourceAdded, ResourceMutated, ResourceRemoved
- Scene maintains an append log of events stamped with generation numbers
- `Scene::events_since(generation) -> &[SceneEvent]` for delta sync
- Initial snapshot = full subtree serialized as DUCK v0.3 bytes
- Client applies event stream; custom node types with unregistered tags stored as opaque bytes
- Server API: `stream_subtree(root_node_id)` — client subscribes, receives snapshot + ongoing events
- Transport layer TBD and kept out of the scene crate (WebSocket, HTTP/2 SSE, WebTransport for WASM)

---

### Phase 6 — CAD Node Types

Apply Phase 2 (Custom Node Type Registry) extensibility to store CAD-specific data directly in the scene tree.

- `CadSurfaceNode`: links a scene mesh node to its source B-Rep face (OCCT shape handle, tessellation params)
- `CadMeasurementNode`: dimension value, tolerance, referenced topology IDs (face/edge), display geometry
- `CadAssemblyNode`: product name, part number, physical properties (mass, density, material spec)
- CAD importer updated: creates typed CAD nodes as children of mesh nodes; PMI becomes `CadMeasurementNode` subtrees instead of raw line geometry
- All types registered via `register_cad_node_types()` called at app startup