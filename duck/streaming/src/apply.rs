use duck_engine_import_export::format::ResourceType;
use duck_engine_scene::{Id, Scene, SceneEvent};

/// Construct the `SceneEvent` that represents sending `resource_id` of `kind` to a client.
///
/// Returns `None` if the resource no longer exists in the scene (it may have been removed
/// between the priority queue being built and the resource being sent).
pub fn build_resource_event(scene: &Scene, kind: ResourceType, resource_id: Id) -> Option<SceneEvent> {
    match kind {
        ResourceType::Node => {
            let node = scene.get_node(resource_id)?.clone();
            Some(SceneEvent::NodeAdded(node))
        }
        ResourceType::Instance => {
            let inst = scene.get_instance(resource_id)?.clone();
            Some(SceneEvent::InstanceAdded(resource_id, inst))
        }
        ResourceType::Material => {
            let mat = scene.get_material(resource_id)?.clone();
            Some(SceneEvent::MaterialAdded(resource_id, mat))
        }
        ResourceType::Mesh => {
            let mesh = scene.get_mesh(resource_id)?.clone();
            Some(SceneEvent::MeshAdded(resource_id, mesh))
        }
        ResourceType::Texture => {
            let tex = scene.get_texture(resource_id)?.clone();
            Some(SceneEvent::TextureAdded(resource_id, tex))
        }
        ResourceType::EnvironmentMap => {
            let em = scene.get_environment_map(resource_id)?.clone();
            Some(SceneEvent::EnvironmentMapAdded(resource_id, em))
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

        SceneEvent::MaterialAdded(_, mat) => { scene.add_material(mat); }
        SceneEvent::MaterialRemoved(id) => { scene.remove_material(id); }

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
