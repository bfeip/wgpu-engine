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

### Phase 4 — DUCK Format Redesign

Redesign the binary format and serialization infrastructure for per-resource access and reduced complexity.

#### 4a — Manual serde for `Texture` and `EnvironmentMap` - DONE

Eliminate `SerializedTexture`, `SerializedEnvironmentMap`, and `SerializedPreprocessedIbl` from `format.rs` by implementing `Serialize`/`Deserialize` directly on the scene types.

- **`Texture` serialize**: embed the actual image bytes as `Vec<u8>` — use `original_bytes` if in memory (e.g., from a glTF), read from `source_path` if known (getting already-compressed PNG/JPEG bytes from disk), or re-encode the decoded `DynamicImage` as PNG as a last resort. Never write the path itself.
- **`Texture` deserialize**: reconstruct a `Texture` from the embedded bytes and stored metadata (`format`, `width`, `height`); the source path is not restored.
- **`EnvironmentMap` serialize**: serialize `intensity`, `rotation`, and `hdr_data: Option<Vec<u8>>` — read HDR bytes from `source_path` on serialize if not already in memory.
- **`EnvironmentMap` deserialize**: reconstruct with the HDR bytes as the source.
- `PreprocessedIbl` already derives Serialize/Deserialize; wire it to `EnvironmentMap` by ID as today.

#### 4b — Flat per-resource format (sections eliminated) - DONE

Replace the section-based format with a flat sequence of independently-encoded resources. Each resource manages its own compression and encoding. The TOC maps each resource ID to its file offset and size.

New file structure:
```
[Header: magic "DUCK", version u16, flags u16, toc_offset u64]
[Resource bytes, one per resource, independently encoded]
[TOC: zstd-compressed bincode Vec<TocEntry>]
```

`TocEntry`:
```rust
pub struct TocEntry {
    pub resource_type: ResourceType,  // Mesh, Texture, Material, Node, Instance, EnvironmentMap, Metadata
    pub resource_id: Uuid,
    pub offset: u64,
    pub size: u32,
}
```

`ResourceType` also carries the encoding in use, so the decoder knows what to do:
- **Texture**: PNG or JPEG bytes written directly (already compressed by the image format; no outer wrapper)
- **EnvironmentMap**: zstd-compressed bincode (includes embedded HDR bytes)
- **Mesh**: zstd-compressed bincode (Draco geometry compression reserved as a future `ResourceType` variant)
- **Material, Node, Instance, Metadata**: zstd-compressed bincode individually

The decoder reads the TOC first (seek to `toc_offset`, decompress, deserialize). To load a resource: look it up by ID in the TOC, seek to its offset, read `size` bytes, decode according to `resource_type`.

Remove the `SectionType` enum and all section-oriented logic from `format.rs`.

#### 4c — Remove annotation filtering - DONE

Delete `AnnotationContentFilter` and its call site in `to_bytes_with_options`. Annotations and their reified geometry are included in the serialized file as normal nodes/instances/meshes. This removes ~80 lines of recursive collection logic.

#### 4d — Break format.rs into modules - DONE

Split `duck/import-export/src/format.rs` into a `format/` directory:

- `format.rs` — public API: `to_bytes`, `from_bytes`, `save_to_file`, `load_from_file`, `to_bytes_with_options`, `FormatError`, `SaveOptions`
- `format/header.rs` — `FileHeader`, `TocEntry`, `ResourceType`, version constants
- `format/compress.rs` — `CompressionLevel`, `compress_with_level`, `decompress`, zstd helpers
- `format/encode.rs` — scene → per-resource bytes → file
- `format/decode.rs` — file → TOC → per-resource decode → scene assembly

---

### Phase 5 — Network Streaming

Enable servers to stream scene subtrees to connected viewers. Both initial sync and live mutations
use a single `EventBatch` mechanism — no separate snapshot format. Resources are streamed in
priority order (most screen-prominent first) so clients can start interacting before the full
model arrives. The server supports live mutation and pushes ongoing delta events to all connected
clients. Transport is abstracted behind a `ByteChannel` trait so TCP, WebSocket, HTTP SSE, or
WebTransport can be swapped in without changing the protocol.

#### New crates

- `duck-engine-streaming` (`duck/streaming/`) — codec, `ByteChannel` trait, `StreamingServer`,
  `StreamingClient`, `apply_event_to_scene`, priority queue
- `scene-server` (`duck/scene-server/`) — CLI binary: loads a model file, serves it on a TCP port

#### Scene crate additions (`duck/scene/src/event.rs`)

- `SceneEvent` enum (serde-gated): `MeshAdded`, `MeshRemoved`, `MaterialAdded`, `MaterialRemoved`,
  `TextureAdded`, `InstanceAdded`, `InstanceRemoved`, `NodeAdded(Node)`, `NodeRemoved`,
  `NodeTransformSet`, `NodePayloadSet`, `NodeVisibilitySet`, `EnvironmentMapAdded`,
  `ActiveEnvironmentMapSet`, `ActiveCameraSet`
- `SceneEventLog` — ring buffer of `(generation, SceneEvent)` pairs, owned by the server
  (not by `Scene` — zero overhead for non-streaming paths)
- Instrumented mutation wrappers (free functions): call raw `Scene` methods and record the event
- `Scene::insert_node` updated to automatically manage `root_nodes` when `parent == None`,
  simplifying both the DUCK deserializer and streaming

#### Wire protocol

Framing: `[4-byte u32 LE length][zstd-compressed bincode Message]`

Client → Server:
- `Subscribe { client_gen, root_node, camera }` — `client_gen = 0` for fresh connect;
  `root_node = Some(id)` for partial-scene subscription; `camera` for priority sorting
- `CameraUpdate(CameraHint)` — re-sort priority queue mid-stream
- `Pause` / `Resume` — flow control for memory-constrained clients

Server → Client:
- `EventBatch { events: Vec<(u64, SceneEvent)> }` — used for both initial sync and live updates
- `SyncComplete { server_gen }` — initial resource streaming done; client is in live mode
- `Ping` / `Goodbye`

Reconnect (`client_gen > 0`): server sends only events since that generation if log is warm;
otherwise re-runs full priority sync.

#### Initial sync order (priority streaming)

1. All `NodeAdded` events (tree skeleton — tiny, arrives first)
2. `MaterialAdded`, `InstanceAdded`, `MeshAdded` in priority order (screen coverage estimate
   from camera hint; larger bounding sphere = higher priority)
3. `TextureAdded` events (lowest priority — potentially large)
4. `SyncComplete`

#### Viewer integration

- `streaming` feature flag on `duck-engine-viewer`
- `Viewer::connect_stream(addr)`, `disconnect_stream()`, `stream_sync_complete()`
- `poll_stream()` called from `Viewer::update()` — non-blocking `try_recv` each frame
- Camera hint sent to server each frame so priority queue stays current as user navigates

#### egui-demo UI

- `streaming` feature flag on `egui-demo`
- New "Network" tab in left panel: URL input (default `127.0.0.1:7878`), Connect/Disconnect
  button, progress indicator while streaming, status label

#### Instance-first priority streaming (follow-up)

The current `build_priority_queue` in `duck/streaming/src/priority.rs` streams resources in
type-grouped buckets: all nodes → all instances → all materials → all meshes → all textures.
This means a client receives every material and every mesh before it can render anything — even if
most of them are off-screen or tiny.

**Goal:** stream the most visually prominent instances first, bundled with the resources they
depend on, so the client can render partial scenes progressively.

**New ordering:**
1. All `NodeAdded` events (still first — tiny tree skeleton)
2. Instances sorted by world-space screen coverage, each preceded by its dependencies (if not
   already queued):
   - Mesh for the instance
   - Material for the instance
   - Textures referenced by that material
3. Remaining textures not yet queued (e.g., from low-priority instances)
4. Environment maps

**Changes required:**

- `duck/streaming/src/priority.rs` — `build_priority_queue`:
  - Replace the type-grouped loops with a single instance-driven pass
  - Sort instances by world-space screen coverage (see below)
  - For each instance in priority order, push its mesh, material, and material's textures
    (guarded by a `HashSet<Id>` to skip already-queued resources), then push the instance
  - After the instance pass, sweep any textures not yet covered; then environment maps

- `duck/streaming/src/priority.rs` — replace `mesh_screen_priority` with
  `instance_screen_priority(scene, instance_id, camera)`:
  - The existing TODO already notes that object-space bounds without transform are wrong
  - Look up the node carrying this instance to get its world transform
  - Apply world transform to mesh AABB min/max corners; compute world-space bounding sphere
  - Compute solid-angle coverage from that sphere and the camera position

- `duck/scene/src/lib.rs` (or scene graph module):
  - Add `Scene::node_for_instance(InstanceId) -> Option<NodeId>` (or equivalent) so
    `instance_screen_priority` can look up world transforms without a linear scan
  - `Scene::world_transform(NodeId)` already exists (cached in `Cell<Option<T>>`); confirm
    it's accessible or expose it if private

- `duck/scene/src/lib.rs` — `Material`:
  - Add `Material::texture_ids() -> impl Iterator<Item = TextureId>` enumerating all textures
    the material references (base color, normal, metallic-roughness, emissive, occlusion) so
    the dependency bundling step can queue them without hard-coding field names in priority.rs

---

### Phase 6 — CAD Node Types

Apply Phase 2 (Custom Node Type Registry) extensibility to store CAD-specific data directly in the scene tree.

- `CadSurfaceNode`: links a scene mesh node to its source B-Rep face (OCCT shape handle, tessellation params)
- `CadMeasurementNode`: dimension value, tolerance, referenced topology IDs (face/edge), display geometry
- `CadAssemblyNode`: product name, part number, physical properties (mass, density, material spec)
- CAD importer updated: creates typed CAD nodes as children of mesh nodes; PMI becomes `CadMeasurementNode` subtrees instead of raw line geometry
- All types registered via `register_cad_node_types()` called at app startup