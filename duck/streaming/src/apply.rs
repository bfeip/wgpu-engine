use duck_engine_import_export::format::ResourceType;
use duck_engine_scene::{Id, Scene, SceneEvent};

/// Construct the `SceneEvent` that represents sending `resource_id` of `kind` to a client.
///
/// Returns `None` if the resource no longer exists in the scene (it may have been removed
/// between the priority queue being built and the resource being sent).
pub fn build_resource_event(scene: &Scene, kind: ResourceType, resource_id: Id) -> Option<SceneEvent> {
    match kind {
        ResourceType::Node => {
            let node = scene.get_node(resource_id.cast())?.clone();
            Some(SceneEvent::NodeAdded(node))
        }
        ResourceType::Instance => {
            let id = resource_id.cast();
            let inst = scene.get_instance(id)?.clone();
            Some(SceneEvent::InstanceAdded(id, inst))
        }
        ResourceType::FaceMaterial => {
            let id = resource_id.cast();
            let mat = scene.get_face_material(id)?.clone();
            Some(SceneEvent::FaceMaterialAdded(id, mat))
        }
        ResourceType::LineMaterial => {
            let id = resource_id.cast();
            let mat = scene.get_line_material(id)?.clone();
            Some(SceneEvent::LineMaterialAdded(id, mat))
        }
        ResourceType::PointMaterial => {
            let id = resource_id.cast();
            let mat = scene.get_point_material(id)?.clone();
            Some(SceneEvent::PointMaterialAdded(id, mat))
        }
        ResourceType::Mesh => {
            let id = resource_id.cast();
            let mesh = scene.get_mesh(id)?.clone();
            Some(SceneEvent::MeshAdded(id, mesh))
        }
        ResourceType::Texture => {
            let id = resource_id.cast();
            let tex = scene.get_texture(id)?.clone();
            Some(SceneEvent::TextureAdded(id, tex))
        }
        ResourceType::EnvironmentMap => {
            let id = resource_id.cast();
            let em = scene.get_environment_map(id)?.clone();
            Some(SceneEvent::EnvironmentMapAdded(id, em))
        }
        ResourceType::Metadata => None,
    }
}

/// Apply a single `SceneEvent` to a local scene replica.
///
/// Called by the client for every event received — both during initial streaming
/// and for live mutation deltas.
pub fn apply_event_to_scene(scene: &mut Scene, event: SceneEvent) {
    match event {
        SceneEvent::MeshAdded(_, mesh) => { scene.add_mesh(mesh); }
        SceneEvent::MeshRemoved(id) => { scene.remove_mesh(id); }

        SceneEvent::FaceMaterialAdded(_, mat) => { scene.add_face_material(mat); }
        SceneEvent::FaceMaterialRemoved(id) => { scene.remove_face_material(id); }
        SceneEvent::LineMaterialAdded(_, mat) => { scene.add_line_material(mat); }
        SceneEvent::LineMaterialRemoved(id) => { scene.remove_line_material(id); }
        SceneEvent::PointMaterialAdded(_, mat) => { scene.add_point_material(mat); }
        SceneEvent::PointMaterialRemoved(id) => { scene.remove_point_material(id); }

        SceneEvent::TextureAdded(_, tex) => { scene.add_texture(tex); }

        SceneEvent::InstanceAdded(_, inst) => { scene.add_instance(inst); }
        SceneEvent::InstanceRemoved(id) => { scene.remove_instance(id); }

        // `insert_node` preserves the server-assigned UUID and automatically manages root_nodes.
        SceneEvent::NodeAdded(node) => { scene.insert_node(node); }
        SceneEvent::NodeRemoved(id) => { scene.remove_node(id); }
        SceneEvent::NodeTransformSet(id, t) => { scene.set_node_transform(id, t); }
        SceneEvent::NodePayloadSet(id, p) => { scene.set_node_payload(id, p); }
        SceneEvent::NodeVisibilitySet(id, v) => { scene.set_node_visibility(id, v); }

        SceneEvent::EnvironmentMapAdded(_, em) => { scene.add_environment_map(em); }
        SceneEvent::ActiveEnvironmentMapSet(id) => { scene.set_active_environment_map(id); }

        SceneEvent::ActiveCameraSet(id) => { scene.set_active_camera(id); }
    }
}
