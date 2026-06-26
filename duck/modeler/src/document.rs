use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use duck_engine_scene::{Id, NodeId, Scene};
use duck_engine_scene::cad::{CadTessellationOptions, retessellate_node, tessellate_into};
use duck_engine_scene::common::{matrix4_to_row_major_f64, Matrix4, Transform};
use opencascade::primitives::{Edge, Face, Shape, Wire};

pub type PartId = Id;

pub struct CadPart {
    pub id: PartId,
    pub name: String,
    pub shape: Shape,
}

pub struct Document {
    parts: Vec<CadPart>,
    part_to_node: HashMap<PartId, NodeId>,
    node_to_part: HashMap<NodeId, PartId>,
    scene: Arc<Mutex<Scene>>,
}

impl Document {
    pub fn new(scene: Arc<Mutex<Scene>>) -> Self {
        Self {
            parts: Vec::new(),
            part_to_node: HashMap::new(),
            node_to_part: HashMap::new(),
            scene,
        }
    }

    pub fn set_scene(&mut self, scene: Arc<Mutex<Scene>>) {
        self.scene = scene;
    }

    pub fn scene(&self) -> &Arc<Mutex<Scene>> {
        &self.scene
    }

    /// Tessellate `shape`, add the resulting node to the scene, store the CAD part,
    /// and record the mapping — all atomically. If tessellation fails, nothing is modified.
    pub fn add_part(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        options: &CadTessellationOptions,
    ) -> Result<PartId> {
        let name = name.into();
        let node = {
            let mut scene = self.scene.lock().unwrap();
            tessellate_into(&shape, &mut *scene, options, None, Some(&name))
                .context("Failed to tessellate part")?
        };
        let id = PartId::new();
        self.parts.push(CadPart { id, name, shape });
        self.part_to_node.insert(id, node);
        self.node_to_part.insert(node, id);
        Ok(id)
    }

    /// Remove a part from the CAD store, the mapping, and the scene.
    pub fn remove_part(&mut self, id: PartId) {
        self.parts.retain(|p| p.id != id);
        if let Some(node) = self.part_to_node.remove(&id) {
            self.node_to_part.remove(&node);
            self.scene.lock().unwrap().remove_node(node);
        }
    }

    pub fn get_part(&self, id: PartId) -> Option<&CadPart> {
        self.parts.iter().find(|p| p.id == id)
    }

    pub fn get_part_mut(&mut self, id: PartId) -> Option<&mut CadPart> {
        self.parts.iter_mut().find(|p| p.id == id)
    }

    /// Bake a transform into the part's CAD geometry via an affine GTransform then
    /// re-tessellate the part in place (preserving its `NodeId`) and reset the node
    /// transform to identity.
    pub fn bake_transform(
        &mut self,
        part: PartId,
        transform: Matrix4,
        options: &CadTessellationOptions,
    ) -> Result<()> {
        let node = self
            .node_for_part(part)
            .context("bake_transform: no node for part")?;

        // Transpose common's column-major f32 matrix into OCCT's row-major f64 array.
        let mat = matrix4_to_row_major_f64(&transform);

        {
            let cad_part = self
                .get_part_mut(part)
                .context("bake_transform: part not found")?;
            cad_part.shape = cad_part.shape.gtransform(mat);
        }

        let cad_part = self
            .get_part(part)
            .context("bake_transform: part not found")?;
        let mut scene = self.scene.lock().unwrap();
        retessellate_node(&cad_part.shape, &mut scene, options, node)?;
        scene.set_node_transform(node, Transform::IDENTITY);

        Ok(())
    }

    pub fn parts(&self) -> impl Iterator<Item = &CadPart> {
        self.parts.iter()
    }

    pub fn part_for_node(&self, node: NodeId) -> Option<PartId> {
        self.node_to_part.get(&node).copied()
    }

    pub fn node_for_part(&self, part: PartId) -> Option<NodeId> {
        self.part_to_node.get(&part).copied()
    }

    /// Resolve a picked face — identified by its tessellation order (`face_index`,
    /// as carried by a `SubGeometryKind::Face` selection) — back to its OCCT [`Face`] sub-shape.
    pub fn face_subshape(&self, node: NodeId, face_index: u32) -> Option<Face> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        part.shape.faces().nth(face_index as usize)
    }

    /// Resolve a picked edge — identified by its tessellation order (`edge_index`,
    /// as carried by a `SubGeometryKind::Edge` selection) — back to its OCCT [`Edge`] sub-shape.
    pub fn edge_subshape(&self, node: NodeId, edge_index: u32) -> Option<Edge> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        part.shape.edges().nth(edge_index as usize)
    }

    /// Resolve a picked edge to the [`Wire`] that contains it in the part's B-Rep.
    ///
    /// A loft profile is a wire, but the selection system reports the individual
    /// edge the user clicked; this finds the wire that edge belongs to (the first
    /// wire containing a topologically-identical edge), so one click selects a
    /// whole multi-edge profile.
    pub fn wire_for_edge(&self, node: NodeId, edge_index: u32) -> Option<Wire> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        let target = part.shape.edges().nth(edge_index as usize)?;
        part.shape.wires().find(|wire| wire.edges().any(|e| e.is_same(&target)))
    }
}
