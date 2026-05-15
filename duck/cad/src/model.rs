use std::collections::HashMap;

use anyhow::Result;
use duck_engine_scene::{NodeId, Scene};

use crate::{CadImportOptions, CadImportResult, CadShape};

/// A stable identifier for a shape within a [`CadModel`].
pub type CadShapeId = u32;

struct Entry {
    shape: CadShape,
    name: Option<String>,
}

/// A named, ID-tracked collection of [`CadShape`]s.
///
/// The modeler owns a `CadModel` to keep BRep shapes alive between edits. Call
/// [`tessellate`][CadModel::tessellate] to produce a [`Scene`] for rendering, or
/// [`tessellate_shape_into`][CadModel::tessellate_shape_into] to incrementally
/// refresh a single shape.
pub struct CadModel {
    shapes: HashMap<CadShapeId, Entry>,
    next_id: CadShapeId,
}

impl Default for CadModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CadModel {
    pub fn new() -> Self {
        Self { shapes: HashMap::new(), next_id: 0 }
    }

    /// Add a shape and return its stable ID.
    pub fn add(&mut self, shape: CadShape) -> CadShapeId {
        let id = self.next_id;
        self.next_id += 1;
        self.shapes.insert(id, Entry { shape, name: None });
        id
    }

    /// Add a named shape and return its stable ID.
    pub fn add_named(&mut self, shape: CadShape, name: impl Into<String>) -> CadShapeId {
        let id = self.next_id;
        self.next_id += 1;
        self.shapes.insert(id, Entry { shape, name: Some(name.into()) });
        id
    }

    /// Return a reference to the shape, or `None` if the ID is unknown.
    pub fn get(&self, id: CadShapeId) -> Option<&CadShape> {
        self.shapes.get(&id).map(|e| &e.shape)
    }

    /// Return a mutable reference to the shape, or `None` if the ID is unknown.
    pub fn get_mut(&mut self, id: CadShapeId) -> Option<&mut CadShape> {
        self.shapes.get_mut(&id).map(|e| &mut e.shape)
    }

    /// Remove a shape from the model and return it.
    pub fn remove(&mut self, id: CadShapeId) -> Option<CadShape> {
        self.shapes.remove(&id).map(|e| e.shape)
    }

    /// Iterate over all shape IDs in insertion order (not guaranteed).
    pub fn ids(&self) -> impl Iterator<Item = CadShapeId> + '_ {
        self.shapes.keys().copied()
    }

    /// Return the name associated with a shape ID, if any.
    pub fn name(&self, id: CadShapeId) -> Option<&str> {
        self.shapes.get(&id).and_then(|e| e.name.as_deref())
    }

    /// Tessellate all shapes into a fresh [`Scene`].
    ///
    /// Each shape becomes its own root node. Shape names are used as node names where available.
    pub fn tessellate(&self, options: &CadImportOptions) -> Result<Scene> {
        let mut scene = Scene::new();
        for (id, entry) in &self.shapes {
            entry
                .shape
                .tessellate_into(&mut scene, options, None, entry.name.as_deref())
                .map_err(|e| e.context(format!("Failed to tessellate shape {id}")))?;
        }
        Ok(scene)
    }

    /// Tessellate a single shape into an existing scene as a child of `parent`.
    ///
    /// Returns `None` if `id` is not in the model.
    pub fn tessellate_shape_into(
        &self,
        id: CadShapeId,
        scene: &mut Scene,
        options: &CadImportOptions,
        parent: Option<NodeId>,
    ) -> Result<Option<CadImportResult>> {
        let Some(entry) = self.shapes.get(&id) else { return Ok(None) };
        let result =
            entry.shape.tessellate_into(scene, options, parent, entry.name.as_deref())?;
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tessellates_multiple_shapes() {
        let mut model = CadModel::new();
        model.add_named(CadShape::sphere(1.0), "sphere");
        model.add_named(CadShape::cuboid(2.0, 2.0, 2.0), "box");

        let options = CadImportOptions::default();
        let scene = model.tessellate(&options).expect("tessellation failed");
        assert_eq!(scene.mesh_count(), 2, "expected one mesh per shape");
        assert_eq!(scene.node_count(), 2, "expected one node per shape");
    }

    #[test]
    fn model_get_and_remove() {
        let mut model = CadModel::new();
        let id = model.add(CadShape::cylinder(0.5, 2.0));
        assert!(model.get(id).is_some());
        model.remove(id);
        assert!(model.get(id).is_none());
    }

    #[test]
    fn tessellate_shape_into_existing_scene() {
        let mut model = CadModel::new();
        let id = model.add_named(CadShape::torus(2.0, 0.5), "torus");

        let mut scene = Scene::new();
        let options = CadImportOptions::default();
        let result = model
            .tessellate_shape_into(id, &mut scene, &options, None)
            .expect("failed");
        assert!(result.is_some());
        assert_eq!(scene.mesh_count(), 1);
    }
}
