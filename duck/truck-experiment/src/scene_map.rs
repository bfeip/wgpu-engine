use std::collections::HashMap;

use duck_engine_scene::{MaterialId, NodeId, Scene};

use crate::body::{Body, BodyId};
use crate::edge::EdgeId;
use crate::face::FaceId;
use crate::tessellation::TessellatedBody;

/// Bidirectional mapping between CAD entities and scene nodes.
///
/// When a [`Body`] is tessellated and added to the scene, each face and edge
/// becomes its own scene node. This map tracks which node corresponds to which
/// CAD entity, enabling selection feedback from scene-level picks back to
/// CAD entities.
#[derive(Debug, Default)]
pub struct CadSceneMap {
    face_to_node: HashMap<FaceId, NodeId>,
    node_to_face: HashMap<NodeId, FaceId>,
    edge_to_node: HashMap<EdgeId, NodeId>,
    node_to_edge: HashMap<NodeId, EdgeId>,
    body_to_root_node: HashMap<BodyId, NodeId>,
}

impl CadSceneMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn face_node(&self, face_id: FaceId) -> Option<NodeId> {
        self.face_to_node.get(&face_id).copied()
    }

    pub fn node_face(&self, node_id: NodeId) -> Option<FaceId> {
        self.node_to_face.get(&node_id).copied()
    }

    pub fn edge_node(&self, edge_id: EdgeId) -> Option<NodeId> {
        self.edge_to_node.get(&edge_id).copied()
    }

    pub fn node_edge(&self, node_id: NodeId) -> Option<EdgeId> {
        self.node_to_edge.get(&node_id).copied()
    }

    pub fn body_root_node(&self, body_id: BodyId) -> Option<NodeId> {
        self.body_to_root_node.get(&body_id).copied()
    }
}

/// Add a tessellated body to the scene and build a [`CadSceneMap`].
///
/// Creates a parent node for the body, then adds each face and edge mesh
/// as child instance nodes. Face meshes use the provided `face_material`,
/// and edge meshes use `line_material`.
pub fn add_body_to_scene(
    body: &Body,
    tessellated: &TessellatedBody,
    scene: &mut Scene,
    face_material: MaterialId,
    line_material: MaterialId,
) -> anyhow::Result<CadSceneMap> {
    let mut map = CadSceneMap::new();

    // Create a root node for the body
    let body_node = scene.add_default_node(None, None)?;
    map.body_to_root_node.insert(body.id(), body_node);

    // Add face meshes
    for (&face_id, mesh) in &tessellated.face_meshes {
        let mesh_id = scene.add_mesh(mesh.clone());
        let node_id = scene.add_instance_node(
            Some(body_node),
            mesh_id,
            face_material,
            None,
            duck_engine_common::Transform::IDENTITY,
        )?;
        map.face_to_node.insert(face_id, node_id);
        map.node_to_face.insert(node_id, face_id);
    }

    // Add edge meshes
    for (&edge_id, mesh) in &tessellated.edge_meshes {
        let mesh_id = scene.add_mesh(mesh.clone());
        let node_id = scene.add_instance_node(
            Some(body_node),
            mesh_id,
            line_material,
            None,
            duck_engine_common::Transform::IDENTITY,
        )?;
        map.edge_to_node.insert(edge_id, node_id);
        map.node_to_edge.insert(node_id, edge_id);
    }

    Ok(map)
}
