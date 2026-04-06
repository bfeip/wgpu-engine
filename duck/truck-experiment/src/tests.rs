use truck_modeling::builder;
use wgpu_engine_scene::{Material, PrimitiveType, Scene};

use crate::{add_body_to_scene, tessellate_body, Body, TessellationOptions};

/// Create a cube solid using truck-modeling's sweep API.
fn make_truck_cube() -> truck_modeling::Solid {
    let v = builder::vertex(truck_modeling::Point3::new(0.0, 0.0, 0.0));
    let edge = builder::tsweep(&v, truck_modeling::Vector3::unit_x());
    let face = builder::tsweep(&edge, truck_modeling::Vector3::unit_y());
    let solid = builder::tsweep(&face, truck_modeling::Vector3::unit_z());
    solid
}

#[test]
fn body_from_truck_solid_has_faces_and_edges() {
    let solid = make_truck_cube();
    let body = Body::from_truck_solid(0, &solid);

    // A cube has 6 faces
    assert_eq!(body.faces().len(), 6, "cube should have 6 faces");

    // A cube has 12 edges
    assert_eq!(body.edges().len(), 12, "cube should have 12 edges");
}

#[test]
fn body_face_and_edge_ids_are_sequential() {
    let solid = make_truck_cube();
    let body = Body::from_truck_solid(0, &solid);

    let mut face_ids: Vec<_> = body.faces().keys().copied().collect();
    face_ids.sort();
    assert_eq!(face_ids, (0..6).collect::<Vec<_>>());

    let mut edge_ids: Vec<_> = body.edges().keys().copied().collect();
    edge_ids.sort();
    assert_eq!(edge_ids, (0..12).collect::<Vec<_>>());
}

#[test]
fn tessellate_cube_produces_triangle_meshes() {
    let solid = make_truck_cube();
    let body = Body::from_truck_solid(0, &solid);
    let options = TessellationOptions { tolerance: 0.1 };

    let tessellated = tessellate_body(&body, &options);

    // One mesh per face
    assert_eq!(tessellated.face_meshes.len(), 6);

    // Each face mesh should have triangle primitives with non-zero indices
    for (_face_id, mesh) in &tessellated.face_meshes {
        assert!(!mesh.vertices().is_empty(), "face mesh should have vertices");
        let tri_count = mesh.index_count(PrimitiveType::TriangleList);
        assert!(tri_count > 0, "face mesh should have triangle indices");
        assert_eq!(tri_count % 3, 0, "triangle index count should be a multiple of 3");
    }
}

#[test]
fn tessellate_cube_produces_line_meshes() {
    let solid = make_truck_cube();
    let body = Body::from_truck_solid(0, &solid);
    let options = TessellationOptions { tolerance: 0.1 };

    let tessellated = tessellate_body(&body, &options);

    // One mesh per edge
    assert_eq!(tessellated.edge_meshes.len(), 12);

    // Each edge mesh should have line primitives
    for (_edge_id, mesh) in &tessellated.edge_meshes {
        assert!(!mesh.vertices().is_empty(), "edge mesh should have vertices");
        let line_count = mesh.index_count(PrimitiveType::LineList);
        assert!(line_count > 0, "edge mesh should have line indices");
        assert_eq!(line_count % 2, 0, "line index count should be a multiple of 2");
    }
}

#[test]
fn add_body_to_scene_creates_nodes() {
    let solid = make_truck_cube();
    let body = Body::from_truck_solid(0, &solid);
    let options = TessellationOptions { tolerance: 0.1 };
    let tessellated = tessellate_body(&body, &options);

    let mut scene = Scene::new();
    let face_material = scene.add_material(Material::default());
    let line_material = scene.add_material(Material::default().with_line_color(
        wgpu_engine_common::RgbaColor::BLACK,
    ));

    let map = add_body_to_scene(&body, &tessellated, &mut scene, face_material, line_material)
        .expect("add_body_to_scene should succeed");

    // Should have a root node for the body
    let root = map.body_root_node(0);
    assert!(root.is_some(), "body should have a root node");

    // Each face should map to a node
    for face_id in body.faces().keys() {
        let node_id = map.face_node(*face_id);
        assert!(node_id.is_some(), "face {face_id} should have a node");

        // Reverse mapping should work too
        let reverse = map.node_face(node_id.unwrap());
        assert_eq!(reverse, Some(*face_id));
    }

    // Each edge should map to a node
    for edge_id in body.edges().keys() {
        let node_id = map.edge_node(*edge_id);
        assert!(node_id.is_some(), "edge {edge_id} should have a node");

        let reverse = map.node_edge(node_id.unwrap());
        assert_eq!(reverse, Some(*edge_id));
    }
}

#[test]
fn scene_nodes_are_children_of_body_root() {
    let solid = make_truck_cube();
    let body = Body::from_truck_solid(0, &solid);
    let options = TessellationOptions { tolerance: 0.1 };
    let tessellated = tessellate_body(&body, &options);

    let mut scene = Scene::new();
    let face_material = scene.add_material(Material::default());
    let line_material = scene.add_material(Material::default().with_line_color(
        wgpu_engine_common::RgbaColor::BLACK,
    ));

    let map = add_body_to_scene(&body, &tessellated, &mut scene, face_material, line_material)
        .expect("add_body_to_scene should succeed");

    let root_id = map.body_root_node(0).unwrap();
    let root_node = scene.get_node(root_id).unwrap();

    // Root should have 6 face nodes + 12 edge nodes = 18 children
    assert_eq!(root_node.children().len(), 18, "root should have 18 children (6 faces + 12 edges)");

    // All face/edge nodes should be children of the root
    for face_id in body.faces().keys() {
        let node_id = map.face_node(*face_id).unwrap();
        let node = scene.get_node(node_id).unwrap();
        assert_eq!(node.parent(), Some(root_id));
    }

    for edge_id in body.edges().keys() {
        let node_id = map.edge_node(*edge_id).unwrap();
        let node = scene.get_node(node_id).unwrap();
        assert_eq!(node.parent(), Some(root_id));
    }
}
