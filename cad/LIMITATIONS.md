# CAD Crate — Known Limitations & Future Work

This document tracks known limitations, roadblocks, and future work items for the `cad` crate. Update it as new issues are discovered during development.

---

### No color / layer data (XDE not exposed)
**Status:** Unresolved  

STEP and IGES files encode per-solid and per-face color via the XDE (Extended Data Framework) subsystem of OCCT (`XCAFDoc_ColorTool`, `XCAFDoc_LayerTool`). The `opencascade-rs` bindings do not currently expose XDE.

All imported geometry receives a single configurable color from `CadImportOptions`.

**Future work:** Add XDE bindings to `opencascade-rs` to extract per-shape/per-face color and map to distinct `scene::Material` entries.

---

### Assembly hierarchy lost (flat import)
**Status:** Unresolved  

`Shape::read_step` / `Shape::read_iges` returns a single root `Shape` (possibly a `TopoDS_Compound`). `opencascade-rs` only exposes flat face/edge/vertex iterators — there is no API to traverse the compound tree (sub-shapes, part names, instance transforms).

All geometry is flattened under a single root node with no name or hierarchy preservation.

**Future work:** Expose `TopoDS_Compound` sub-shape traversal in `opencascade-rs`. Use XDE's `XCAFDoc_DocumentTool` to recover assembly structure with part names and transforms.

---

### Edge deduplication (wireframe doubling)
**Status:** Not yet implemented  

`shape.edges()` traverses all topological edges. Shared edges between adjacent faces appear once per face adjacency, producing doubled line segments in the wireframe.

**Future work:** Deduplicate edges by hashing (start, end) point pairs (rounded to a fixed epsilon) before building the `LineList` mesh.

---

### No unit metadata / coordinate system normalization
**Status:** Worked around via `scale_factor`  

STEP files typically use millimetres; the engine has no concept of units. `CadImportOptions::scale_factor` (default `1.0`) allows the caller to convert units (e.g. `0.001` for mm → m), but this is manual.

**Future work:** Read the unit system from the STEP header (`LENGTH_UNIT`) and auto-apply a scale factor, or expose unit information via the import result.

---

### No BREP / parametric data preserved
**Status:** By design for PoC  

The importer only extracts tessellated triangle/line geometry. The underlying B-Rep topology (surfaces, curves, topology graph) is discarded after tessellation.

**Future work:** If parametric re-tessellation or CAD editing is needed, retain the OCCT shape tree alongside the scene representation.

---

### No in-memory (bytes) loading
**Status:** Unresolved

`opencascade-rs` only exposes `ReadFile(filename)` — the underlying OCCT C++ bindings have no stream or memory-based read path for STEP or IGES. Loading CAD files from `SceneSource::Bytes` (with no filesystem path) is therefore unsupported. The `import-export` `CadImporter` returns `LoadError::UnsupportedPlatform` in this case.

**Future work:** Expose OCCT's in-memory stream reading in `opencascade-rs` (e.g. via `OSD_MemStream`) and update `cad::load_step` / `load_iges` to accept `&[u8]` directly.
