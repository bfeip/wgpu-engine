# CAD Crate — Known Limitations & Future Work

This document tracks known limitations, roadblocks, and future work items for the `cad` crate. Update it as new issues are discovered during development.

---

## Active Limitations

### 1. `MeshIndex = u16` vertex cap
**Status:** Worked around via chunking  
**File:** `scene/src/mesh.rs` — `pub type MeshIndex = u16`

Each `scene::Mesh` is capped at 65,535 vertices. Complex CAD models can easily exceed this. The importer chunks large meshes into multiple child nodes using `split_mesh` / `split_line_mesh` from `wgpu-engine-scene`.

**Long-term fix:** Change `MeshIndex` to `u32`. Requires updating the renderer's wgpu index format from `Uint16` to `Uint32` and auditing all GPU pipeline code.

---

### 2. No color / layer data (XDE not exposed)
**Status:** Unresolved  

STEP and IGES files encode per-solid and per-face color via the XDE (Extended Data Framework) subsystem of OCCT (`XCAFDoc_ColorTool`, `XCAFDoc_LayerTool`). The `opencascade-rs` bindings do not currently expose XDE.

All imported geometry receives a single configurable color from `CadImportOptions`.

**Future work:** Add XDE bindings to `opencascade-rs` to extract per-shape/per-face color and map to distinct `scene::Material` entries.

---

### 3. Assembly hierarchy lost (flat import)
**Status:** Unresolved  

`Shape::read_step` / `Shape::read_iges` returns a single root `Shape` (possibly a `TopoDS_Compound`). `opencascade-rs` only exposes flat face/edge/vertex iterators — there is no API to traverse the compound tree (sub-shapes, part names, instance transforms).

All geometry is flattened under a single root node with no name or hierarchy preservation.

**Future work:** Expose `TopoDS_Compound` sub-shape traversal in `opencascade-rs`. Use XDE's `XCAFDoc_DocumentTool` to recover assembly structure with part names and transforms.

---

### 4. Edge deduplication (wireframe doubling)
**Status:** Not yet implemented  

`shape.edges()` traverses all topological edges. Shared edges between adjacent faces appear once per face adjacency, producing doubled line segments in the wireframe.

**Future work:** Deduplicate edges by hashing (start, end) point pairs (rounded to a fixed epsilon) before building the `LineList` mesh.

---

### 5. No unit metadata / coordinate system normalization
**Status:** Worked around via `scale_factor`  

STEP files typically use millimetres; the engine has no concept of units. `CadImportOptions::scale_factor` (default `1.0`) allows the caller to convert units (e.g. `0.001` for mm → m), but this is manual.

**Future work:** Read the unit system from the STEP header (`LENGTH_UNIT`) and auto-apply a scale factor, or expose unit information via the import result.

---

### 6. No BREP / parametric data preserved
**Status:** By design for PoC  

The importer only extracts tessellated triangle/line geometry. The underlying B-Rep topology (surfaces, curves, topology graph) is discarded after tessellation.

**Future work:** If parametric re-tessellation or CAD editing is needed, retain the OCCT shape tree alongside the scene representation.
