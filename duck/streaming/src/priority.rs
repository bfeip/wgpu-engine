use duck_engine_common::Point3;
use duck_engine_import_export::format::ResourceType;
use duck_engine_scene::{Id, MeshId, Scene};

use crate::codec::CameraHint;

/// A resource pending delivery to a newly connected client, with its streaming priority.
#[derive(Debug, Clone)]
pub struct PendingResource {
    pub kind: ResourceType,
    pub id: Id,
    /// Higher value = send sooner.
    pub priority: f32,
}

/// Build a list of resources to stream to a fresh client, ordered highest-priority first.
///
/// Order:
/// 1. Nodes — tree skeleton; tiny, delivers the scene hierarchy immediately
/// 2. Instances — structural bindings required before rendering
/// 3. Materials — small, needed for shading appearance
/// 4. Meshes — sorted by estimated screen coverage (most visible first)
/// 5. Textures — potentially large; sent last
/// 6. Environment maps — large HDR data; sent last
pub fn build_priority_queue(scene: &Scene, camera: Option<&CameraHint>) -> Vec<PendingResource> {
    let mut resources: Vec<PendingResource> = Vec::new();

    for node in scene.nodes() {
        resources.push(PendingResource {
            kind: ResourceType::Node,
            id: node.id,
            priority: 10_000_000.0,
        });
    }

    for instance in scene.instances() {
        resources.push(PendingResource {
            kind: ResourceType::Instance,
            id: instance.id,
            priority: 1_000_000.0,
        });
    }

    for material in scene.materials() {
        resources.push(PendingResource {
            kind: ResourceType::Material,
            id: material.id,
            priority: 900_000.0,
        });
    }

    for mesh in scene.meshes() {
        let score = mesh_screen_priority(scene, mesh.id, camera);
        resources.push(PendingResource {
            kind: ResourceType::Mesh,
            id: mesh.id,
            priority: score,
        });
    }

    for texture in scene.textures() {
        resources.push(PendingResource {
            kind: ResourceType::Texture,
            id: texture.id,
            priority: -1.0,
        });
    }

    for em in scene.environment_maps() {
        resources.push(PendingResource {
            kind: ResourceType::EnvironmentMap,
            id: em.id,
            priority: -2.0,
        });
    }

    resources.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
    resources
}

/// Estimate a mesh's visual prominence from a camera hint.
///
/// TODO: This currently uses raw mesh bounds (object space, no transform applied).
/// It should instead aggregate the world-space bounds of all instances that reference
/// this mesh, so meshes placed far from the origin are deprioritized appropriately.
fn mesh_screen_priority(scene: &Scene, mesh_id: MeshId, camera: Option<&CameraHint>) -> f32 {
    let Some(mesh) = scene.get_mesh(mesh_id) else {
        return 0.0;
    };
    let Some(bounds) = mesh.bounding() else {
        return 0.0;
    };
    let extent = bounds.max - bounds.min;
    let radius = (extent.x * extent.x + extent.y * extent.y + extent.z * extent.z).sqrt() * 0.5;

    let Some(cam) = camera else {
        return radius;
    };

    let center = bounds.min + extent * 0.5;
    let cam_pos = Point3::from(cam.position);
    let diff = center - cam_pos;
    let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

    radius * radius / dist_sq.max(1e-6)
}
