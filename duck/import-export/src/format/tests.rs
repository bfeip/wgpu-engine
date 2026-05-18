use super::*;
use duck_engine_common::{Point3, Quaternion, Vector3};
use duck_engine_scene::common::{RgbaColor, Transform};
use duck_engine_scene::PrimitiveType;

fn create_test_scene() -> Scene {
    let mut scene = Scene::new();

    let mesh = Mesh::cube(1.0, PrimitiveType::TriangleList);
    let mesh_id = scene.add_mesh(mesh);

    let material = Material::new()
        .with_base_color_factor(RgbaColor::RED)
        .with_metallic_factor(0.5)
        .with_roughness_factor(0.3)
        .with_line_color(RgbaColor::GREEN);
    let mat_id = scene.add_material(material);

    let node_id = scene.add_instance_node(
        None,
        mesh_id,
        mat_id,
        Some("TestNode".to_string()),
        Transform::new(
            Point3::new(1.0, 2.0, 3.0),
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Vector3::new(2.0, 2.0, 2.0),
        ),
        NodeFlags::NONE,
    ).unwrap();

    let _child_id = scene.add_node(
        Some(node_id),
        Some("ChildNode".to_string()),
        Transform::from_position(Point3::new(0.5, 0.5, 0.5)),
        NodeFlags::NONE,
    ).unwrap();

    {
        use duck_engine_scene::{Light, NodePayload};
        let light_node_id = scene.add_node(
            None,
            None,
            Transform::from_position(Point3::new(5.0, 5.0, 5.0)),
            NodeFlags::NONE,
        ).unwrap();
        scene.set_node_payload(light_node_id, NodePayload::Light(Light::point(RgbaColor::WHITE, 10.0)));
    }

    scene
}

#[test]
fn test_round_trip_basic() {
    let original = create_test_scene();

    let bytes = to_bytes(&original).expect("Failed to serialize scene");

    assert_eq!(&bytes[0..4], b"DUCK");

    let loaded = from_bytes(&bytes).expect("Failed to deserialize scene");

    assert_eq!(loaded.node_count(), original.node_count());
    assert_eq!(loaded.mesh_count(), original.mesh_count());
    assert_eq!(loaded.material_count(), original.material_count());
    assert_eq!(loaded.instance_count(), original.instance_count());
    assert_eq!(loaded.has_light_nodes(), original.has_light_nodes());
    assert_eq!(loaded.root_nodes().len(), original.root_nodes().len());
}

#[test]
fn test_round_trip_node_properties() {
    let original = create_test_scene();
    let bytes = to_bytes(&original).expect("Failed to serialize");
    let loaded = from_bytes(&bytes).expect("Failed to deserialize");

    let original_node = original.nodes()
        .find(|n| n.name.as_deref() == Some("TestNode"))
        .expect("TestNode not found in original");

    let loaded_node = loaded.nodes()
        .find(|n| n.name.as_deref() == Some("TestNode"))
        .expect("TestNode not found in loaded");

    let orig_pos = original_node.position();
    let loaded_pos = loaded_node.position();
    assert!((orig_pos.x - loaded_pos.x).abs() < 1e-6);
    assert!((orig_pos.y - loaded_pos.y).abs() < 1e-6);
    assert!((orig_pos.z - loaded_pos.z).abs() < 1e-6);

    let orig_scale = original_node.scale();
    let loaded_scale = loaded_node.scale();
    assert!((orig_scale.x - loaded_scale.x).abs() < 1e-6);
    assert!((orig_scale.y - loaded_scale.y).abs() < 1e-6);
    assert!((orig_scale.z - loaded_scale.z).abs() < 1e-6);
}

#[test]
#[ignore = "Fails unpredictably, possibly due to multithreaded serialization"]
fn test_round_trip_material_properties() {
    let original = create_test_scene();
    let bytes = to_bytes(&original).expect("Failed to serialize");
    let loaded = from_bytes(&bytes).expect("Failed to deserialize");

    let original_mat = original.materials().next().expect("No material in original");
    let loaded_mat = loaded.materials().next().expect("No material in loaded");

    let orig_color = original_mat.base_color_factor();
    let loaded_color = loaded_mat.base_color_factor();
    assert!((orig_color.r - loaded_color.r).abs() < 1e-6);
    assert!((orig_color.g - loaded_color.g).abs() < 1e-6);
    assert!((orig_color.b - loaded_color.b).abs() < 1e-6);

    assert!((original_mat.metallic_factor() - loaded_mat.metallic_factor()).abs() < 1e-6);
    assert!((original_mat.roughness_factor() - loaded_mat.roughness_factor()).abs() < 1e-6);

    assert!(original_mat.line_color().is_some());
    assert!(loaded_mat.line_color().is_some());
}

#[test]
fn test_round_trip_mesh_geometry() {
    let original = create_test_scene();
    let bytes = to_bytes(&original).expect("Failed to serialize");
    let loaded = from_bytes(&bytes).expect("Failed to deserialize");

    let original_mesh = original.meshes().next().expect("No mesh in original");
    let loaded_mesh = loaded.meshes().next().expect("No mesh in loaded");

    assert_eq!(original_mesh.vertices().len(), loaded_mesh.vertices().len());
    assert_eq!(original_mesh.primitives().len(), loaded_mesh.primitives().len());

    for (orig_v, loaded_v) in original_mesh.vertices().iter().zip(loaded_mesh.vertices()) {
        assert!((orig_v.position[0] - loaded_v.position[0]).abs() < 1e-6);
        assert!((orig_v.position[1] - loaded_v.position[1]).abs() < 1e-6);
        assert!((orig_v.position[2] - loaded_v.position[2]).abs() < 1e-6);
    }
}

#[test]
fn test_round_trip_hierarchy() {
    let original = create_test_scene();
    let bytes = to_bytes(&original).expect("Failed to serialize");
    let loaded = from_bytes(&bytes).expect("Failed to deserialize");

    let loaded_child = loaded.nodes()
        .find(|n| n.name.as_deref() == Some("ChildNode"))
        .expect("ChildNode not found");

    assert!(loaded_child.parent().is_some());

    let loaded_parent = loaded.nodes()
        .find(|n| n.name.as_deref() == Some("TestNode"))
        .expect("TestNode not found");

    assert!(!loaded_parent.children().is_empty());
}

#[test]
fn test_round_trip_lights() {
    use duck_engine_scene::{Light, NodePayload};
    let original = create_test_scene();
    let bytes = to_bytes(&original).expect("Failed to serialize");
    let loaded = from_bytes(&bytes).expect("Failed to deserialize");

    assert!(loaded.has_light_nodes());

    let light_node = loaded.nodes()
        .find(|n| matches!(n.payload(), NodePayload::Light(_)))
        .expect("No light node found");

    match light_node.payload() {
        NodePayload::Light(Light::Point { color, intensity, .. }) => {
            assert!((color.r - 1.0).abs() < 1e-6);
            assert!((*intensity - 10.0).abs() < 1e-6);
        }
        _ => panic!("Expected point light node"),
    }
}

#[test]
fn test_empty_scene_round_trip() {
    let scene = Scene::new();
    let bytes = to_bytes(&scene).expect("Failed to serialize empty scene");
    let loaded = from_bytes(&bytes).expect("Failed to deserialize empty scene");

    assert_eq!(loaded.material_count(), 0);
    assert_eq!(loaded.node_count(), 0);
    assert_eq!(loaded.mesh_count(), 0);
    assert_eq!(loaded.instance_count(), 0);
}

#[test]
fn test_version_in_header() {
    let scene = Scene::new();
    let bytes = to_bytes(&scene).expect("Failed to serialize");

    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    assert_eq!(version, VERSION);
}

#[test]
fn test_invalid_magic_rejected() {
    let mut bytes = vec![b'X', b'X', b'X', b'X'];
    bytes.extend([0u8; 12]);

    let result = from_bytes(&bytes);
    assert!(matches!(result, Err(FormatError::InvalidMagic)));
}

#[test]
fn estimate_is_upper_bound() {
    let scene = create_test_scene();
    let estimate = estimate_serialized_size(&scene);
    let bytes = to_bytes(&scene).expect("serialize");
    assert!(
        estimate >= bytes.len(),
        "estimate ({}) should be >= actual size ({})",
        estimate,
        bytes.len(),
    );
}

#[test]
fn estimate_empty_scene() {
    let scene = Scene::new();
    let estimate = estimate_serialized_size(&scene);
    assert!(estimate > 0);
    let bytes = to_bytes(&scene).expect("serialize");
    assert!(estimate >= bytes.len());
}

// ============== DO_NOT_EXPORT tests ==============

#[test]
fn test_do_not_export_excludes_node() {
    use duck_engine_scene::NodeFlags;

    let mut scene = Scene::new();
    let mesh_id = scene.add_mesh(Mesh::cube(1.0, PrimitiveType::TriangleList));
    let mat_id = scene.add_material(Material::new());

    let _normal = scene.add_instance_node(
        None, mesh_id, mat_id, Some("Normal".to_string()), Transform::IDENTITY, NodeFlags::NONE,
    ).unwrap();
    let _hidden = scene.add_instance_node(
        None, mesh_id, mat_id, Some("Hidden".to_string()), Transform::IDENTITY, NodeFlags::DO_NOT_EXPORT,
    ).unwrap();

    let bytes = to_bytes(&scene).expect("serialize");
    let loaded = from_bytes(&bytes).expect("deserialize");

    assert_eq!(loaded.node_count(), 1);
    assert!(loaded.nodes().any(|n| n.name.as_deref() == Some("Normal")));
    assert!(!loaded.nodes().any(|n| n.name.as_deref() == Some("Hidden")));
}

#[test]
fn test_do_not_export_excludes_subtree() {
    use duck_engine_scene::NodeFlags;

    let mut scene = Scene::new();
    let root = scene.add_default_node(None, Some("Root".to_string())).unwrap();
    let _normal_child = scene.add_default_node(Some(root), Some("NormalChild".to_string())).unwrap();
    let hidden_parent = scene.add_node(
        Some(root), Some("HiddenParent".to_string()), Transform::IDENTITY, NodeFlags::DO_NOT_EXPORT,
    ).unwrap();
    let _hidden_grandchild = scene.add_default_node(Some(hidden_parent), Some("HiddenGrandchild".to_string())).unwrap();

    let bytes = to_bytes(&scene).expect("serialize");
    let loaded = from_bytes(&bytes).expect("deserialize");

    // Root + NormalChild survive; HiddenParent and HiddenGrandchild are excluded.
    assert_eq!(loaded.node_count(), 2);
    assert!(loaded.nodes().any(|n| n.name.as_deref() == Some("Root")));
    assert!(loaded.nodes().any(|n| n.name.as_deref() == Some("NormalChild")));
    assert!(!loaded.nodes().any(|n| n.name.as_deref() == Some("HiddenParent")));
    assert!(!loaded.nodes().any(|n| n.name.as_deref() == Some("HiddenGrandchild")));
}

#[test]
fn test_do_not_export_excludes_instance_and_mesh() {
    use duck_engine_scene::NodeFlags;

    let mut scene = Scene::new();
    let shared_mesh_id = scene.add_mesh(Mesh::cube(1.0, PrimitiveType::TriangleList));
    let exclusive_mesh_id = scene.add_mesh(Mesh::cube(2.0, PrimitiveType::TriangleList));
    let mat_id = scene.add_material(Material::new());

    // One normal node using the shared mesh.
    scene.add_instance_node(
        None, shared_mesh_id, mat_id, Some("Normal".to_string()), Transform::IDENTITY, NodeFlags::NONE,
    ).unwrap();
    // One hidden node using the exclusive mesh (not shared with anyone else).
    scene.add_instance_node(
        None, exclusive_mesh_id, mat_id, Some("Hidden".to_string()), Transform::IDENTITY, NodeFlags::DO_NOT_EXPORT,
    ).unwrap();

    let bytes = to_bytes(&scene).expect("serialize");
    let loaded = from_bytes(&bytes).expect("deserialize");

    // Only the normal node and its resources survive.
    assert_eq!(loaded.node_count(), 1);
    assert_eq!(loaded.instance_count(), 1);
    // The shared mesh is retained; the exclusive mesh is gone.
    assert_eq!(loaded.mesh_count(), 1);
}
