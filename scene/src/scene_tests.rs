use super::*;
use cgmath::{Point3, Quaternion, Vector3, Matrix4, SquareMatrix};
use crate::common::{EPSILON, RgbaColor};

// ========================================================================
// Scene Creation and Basic Operations
// ========================================================================

#[test]
fn test_scene_new() {
    let scene = Scene::new();

    assert_eq!(scene.meshes.len(), 0);
    assert_eq!(scene.instances.len(), 0);
    assert_eq!(scene.nodes.len(), 0);
    assert_eq!(scene.root_nodes.len(), 0);
    assert_eq!(scene.lights.len(), 0);
}

#[test]
fn test_add_instance() {
    let mut scene = Scene::new();

    let instance_id = scene.add_instance(10, 5);
    assert_eq!(instance_id, 0);
    assert_eq!(scene.instances.len(), 1);

    let instance = scene.instances.get(&instance_id).unwrap();
    assert_eq!(instance.mesh, 10);
    assert_eq!(instance.material, 5);
}

#[test]
fn test_add_multiple_instances() {
    let mut scene = Scene::new();

    let id1 = scene.add_instance(1, 1);
    let id2 = scene.add_instance(2, 2);
    let id3 = scene.add_instance(3, 3);

    assert_eq!(id1, 0);
    assert_eq!(id2, 1);
    assert_eq!(id3, 2);
    assert_eq!(scene.instances.len(), 3);
}

// ========================================================================
// Scene Tree Construction
// ========================================================================

#[test]
fn test_add_root_node() {
    let mut scene = Scene::new();

    let node_id = scene.add_default_node(None, None).unwrap();

    assert_eq!(node_id, 0);
    assert_eq!(scene.nodes.len(), 1);
    assert_eq!(scene.root_nodes.len(), 1);
    assert_eq!(scene.root_nodes[0], node_id);

    let node = scene.get_node(node_id).unwrap();
    assert_eq!(node.parent(), None);
    assert_eq!(node.children().len(), 0);
}

#[test]
fn test_add_child_node() {
    let mut scene = Scene::new();

    let root = scene.add_default_node(None, None).unwrap();
    let child = scene.add_default_node(Some(root), None).unwrap();

    assert_eq!(scene.nodes.len(), 2);
    assert_eq!(scene.root_nodes.len(), 1);

    // Verify parent-child relationship is bidirectional
    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.children().len(), 1);
    assert_eq!(root_node.children()[0], child);

    let child_node = scene.get_node(child).unwrap();
    assert_eq!(child_node.parent(), Some(root));
}

#[test]
fn test_add_multiple_children() {
    let mut scene = Scene::new();

    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(root), None).unwrap();
    let child3 = scene.add_default_node(Some(root), None).unwrap();

    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.children().len(), 3);
    assert!(root_node.children().contains(&child1));
    assert!(root_node.children().contains(&child2));
    assert!(root_node.children().contains(&child3));

    // Verify all children point back to root
    assert_eq!(scene.get_node(child1).unwrap().parent(), Some(root));
    assert_eq!(scene.get_node(child2).unwrap().parent(), Some(root));
    assert_eq!(scene.get_node(child3).unwrap().parent(), Some(root));
}

#[test]
fn test_deep_hierarchy() {
    let mut scene = Scene::new();

    // Create a chain: root -> child1 -> child2 -> child3
    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(child1), None).unwrap();
    let child3 = scene.add_default_node(Some(child2), None).unwrap();

    // Verify the chain is correct
    assert_eq!(scene.get_node(root).unwrap().parent(), None);
    assert_eq!(scene.get_node(child1).unwrap().parent(), Some(root));
    assert_eq!(scene.get_node(child2).unwrap().parent(), Some(child1));
    assert_eq!(scene.get_node(child3).unwrap().parent(), Some(child2));

    // Verify each node has the correct children
    assert_eq!(scene.get_node(root).unwrap().children().len(), 1);
    assert_eq!(scene.get_node(child1).unwrap().children().len(), 1);
    assert_eq!(scene.get_node(child2).unwrap().children().len(), 1);
    assert_eq!(scene.get_node(child3).unwrap().children().len(), 0);
}

#[test]
fn test_multiple_root_nodes() {
    let mut scene = Scene::new();

    let root1 = scene.add_default_node(None, None).unwrap();
    let root2 = scene.add_default_node(None, None).unwrap();
    let root3 = scene.add_default_node(None, None).unwrap();

    assert_eq!(scene.root_nodes.len(), 3);
    assert!(scene.root_nodes.contains(&root1));
    assert!(scene.root_nodes.contains(&root2));
    assert!(scene.root_nodes.contains(&root3));

    // Verify all are truly roots
    assert_eq!(scene.get_node(root1).unwrap().parent(), None);
    assert_eq!(scene.get_node(root2).unwrap().parent(), None);
    assert_eq!(scene.get_node(root3).unwrap().parent(), None);
}

#[test]
fn test_complex_tree_structure() {
    let mut scene = Scene::new();

    // Create a tree:
    //       root
    //      /    \
    //    c1      c2
    //   /  \      \
    //  gc1  gc2   gc3

    let root = scene.add_default_node(None, None).unwrap();
    let c1 = scene.add_default_node(Some(root), None).unwrap();
    let c2 = scene.add_default_node(Some(root), None).unwrap();
    let gc1 = scene.add_default_node(Some(c1), None).unwrap();
    let gc2 = scene.add_default_node(Some(c1), None).unwrap();
    let gc3 = scene.add_default_node(Some(c2), None).unwrap();

    // Verify structure
    assert_eq!(scene.root_nodes.len(), 1);
    assert_eq!(scene.nodes.len(), 6);

    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.children().len(), 2);

    let c1_node = scene.get_node(c1).unwrap();
    assert_eq!(c1_node.children().len(), 2);

    let c2_node = scene.get_node(c2).unwrap();
    assert_eq!(c2_node.children().len(), 1);

    // Verify leaf nodes have no children
    assert_eq!(scene.get_node(gc1).unwrap().children().len(), 0);
    assert_eq!(scene.get_node(gc2).unwrap().children().len(), 0);
    assert_eq!(scene.get_node(gc3).unwrap().children().len(), 0);
}

// ========================================================================
// Node with Instance
// ========================================================================

#[test]
fn test_add_instance_node() {
    let mut scene = Scene::new();

    let node_id = scene.add_instance_node(
        None,
        10,
        5,
        None,
        Point3::new(0.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    ).unwrap();

    assert_eq!(scene.nodes.len(), 1);
    assert_eq!(scene.instances.len(), 1);

    let node = scene.get_node(node_id).unwrap();
    assert!(node.instance().is_some());

    let instance_id = node.instance().unwrap();
    let instance = scene.instances.get(&instance_id).unwrap();
    assert_eq!(instance.mesh, 10);
    assert_eq!(instance.material, 5);
}

// ========================================================================
// Transform Computation and Caching
// ========================================================================

#[test]
fn test_root_node_identity_transform() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();

    let transform = scene.nodes_transform(root);
    let identity = Matrix4::identity();

    // Convert to arrays for comparison
    let t: [[f32; 4]; 4] = transform.into();
    let i: [[f32; 4]; 4] = identity.into();

    for row in 0..4 {
        for col in 0..4 {
            assert!((t[row][col] - i[row][col]).abs() < EPSILON,
                "Transform mismatch at [{row}][{col}]");
        }
    }
}

#[test]
fn test_child_transform_accumulation() {
    let mut scene = Scene::new();

    // Root at (10, 0, 0)
    let root = scene.add_node(
        None,
        None,
        Point3::new(10.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    ).unwrap();

    // Child at (5, 0, 0) relative to parent
    let child = scene.add_node(
        Some(root),
        None,
        Point3::new(5.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    ).unwrap();

    let child_transform = scene.nodes_transform(child);

    // Child should be at (15, 0, 0) in world space
    assert!((child_transform[3][0] - 15.0).abs() < EPSILON);
    assert!((child_transform[3][1] - 0.0).abs() < EPSILON);
    assert!((child_transform[3][2] - 0.0).abs() < EPSILON);
}

#[test]
fn test_transform_with_scale() {
    let mut scene = Scene::new();

    // Parent with scale 2.0
    let parent = scene.add_node(
        None,
        None,
        Point3::new(0.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(2.0, 2.0, 2.0),
    ).unwrap();

    // Child at (1, 0, 0)
    let child = scene.add_node(
        Some(parent),
        None,
        Point3::new(1.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    ).unwrap();

    let child_transform = scene.nodes_transform(child);

    // Child should be at (2, 0, 0) due to parent's scale
    assert!((child_transform[3][0] - 2.0).abs() < EPSILON);
}

#[test]
fn test_transform_caching() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();

    // First computation
    let transform1 = scene.nodes_transform(root);

    // Second computation should use cache
    let transform2 = scene.nodes_transform(root);

    // Should be identical
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(transform1[i][j], transform2[i][j]);
        }
    }

    // Verify cache is being used
    let node = scene.get_node(root).unwrap();
    assert!(!node.transform_dirty());
}

#[test]
fn test_transform_invalidation_on_change() {
    let mut scene = Scene::new();
    let node_id = scene.add_default_node(None, None).unwrap();

    // Compute transform to populate cache
    let _transform = scene.nodes_transform(node_id);

    // Modify the node
    let node = scene.get_node_mut(node_id).unwrap();
    node.set_position(Point3::new(5.0, 5.0, 5.0));

    // Cache should be dirty
    assert!(node.transform_dirty());
}

// ========================================================================
// Scene Tree Consistency
// ========================================================================

#[test]
fn test_tree_consistency_after_multiple_operations() {
    let mut scene = Scene::new();

    // Build a tree
    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(root), None).unwrap();
    let grandchild = scene.add_default_node(Some(child1), None).unwrap();

    // Verify every node can be accessed
    assert!(scene.get_node(root).is_some());
    assert!(scene.get_node(child1).is_some());
    assert!(scene.get_node(child2).is_some());
    assert!(scene.get_node(grandchild).is_some());

    // Verify every parent reference is valid
    for (_, node) in &scene.nodes {
        if let Some(parent_id) = node.parent() {
            assert!(scene.get_node(parent_id).is_some(),
                "Node references non-existent parent");
        }
    }

    // Verify every child reference is valid
    for (_, node) in &scene.nodes {
        for &child_id in node.children() {
            assert!(scene.get_node(child_id).is_some(),
                "Node references non-existent child");
        }
    }

    // Verify all root nodes are actually roots
    for &root_id in &scene.root_nodes {
        let node = scene.get_node(root_id).unwrap();
        assert!(node.parent().is_none(),
            "Root node has a parent");
    }
}

#[test]
fn test_bidirectional_consistency() {
    let mut scene = Scene::new();

    let parent = scene.add_default_node(None, None).unwrap();
    let child = scene.add_default_node(Some(parent), None).unwrap();

    // Parent should list child
    let parent_node = scene.get_node(parent).unwrap();
    assert!(parent_node.children().contains(&child));

    // Child should reference parent
    let child_node = scene.get_node(child).unwrap();
    assert_eq!(child_node.parent(), Some(parent));
}

#[test]
fn test_no_orphaned_nodes() {
    let mut scene = Scene::new();

    let root1 = scene.add_default_node(None, None).unwrap();
    let root2 = scene.add_default_node(None, None).unwrap();
    let _child1 = scene.add_default_node(Some(root1), None).unwrap();
    let _child2 = scene.add_default_node(Some(root2), None).unwrap();

    // All nodes should be reachable from roots
    let mut reachable = std::collections::HashSet::new();

    fn visit_tree(
        scene: &Scene,
        node_id: NodeId,
        reachable: &mut std::collections::HashSet<NodeId>
    ) {
        reachable.insert(node_id);
        let node = scene.get_node(node_id).unwrap();
        for &child_id in node.children() {
            visit_tree(scene, child_id, reachable);
        }
    }

    for &root_id in &scene.root_nodes {
        visit_tree(&scene, root_id, &mut reachable);
    }

    // All nodes should be reachable
    assert_eq!(reachable.len(), scene.nodes.len());
}

#[test]
fn test_large_tree_consistency() {
    let mut scene = Scene::new();

    let root = scene.add_default_node(None, None).unwrap();

    // Create 100 children
    let mut children = Vec::new();
    for _ in 0..100 {
        let child = scene.add_default_node(Some(root), None).unwrap();
        children.push(child);
    }

    // Verify all children are registered
    let root_node = scene.get_node(root).unwrap();
    assert_eq!(root_node.children().len(), 100);

    // Verify all children reference the parent
    for &child_id in &children {
        let child_node = scene.get_node(child_id).unwrap();
        assert_eq!(child_node.parent(), Some(root));
    }
}

#[test]
fn test_get_node_mut_allows_modification() {
    let mut scene = Scene::new();
    let node_id = scene.add_default_node(None, None).unwrap();

    {
        let node = scene.get_node_mut(node_id).unwrap();
        node.set_position(Point3::new(10.0, 20.0, 30.0));
    }

    let node = scene.get_node(node_id).unwrap();
    assert_eq!(node.position(), Point3::new(10.0, 20.0, 30.0));
}

// ========================================================================
// Parent Validation Tests
// ========================================================================

#[test]
fn test_add_node_with_invalid_parent_fails() {
    let mut scene = Scene::new();

    // Try to add a node with a non-existent parent
    let result = scene.add_node(
        Some(999), // Non-existent parent ID
        None,
        Point3::new(0.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    );

    assert!(result.is_err());
    assert_eq!(scene.nodes.len(), 0);
}

#[test]
fn test_add_default_node_with_invalid_parent_fails() {
    let mut scene = Scene::new();

    let result = scene.add_default_node(Some(999), None);

    assert!(result.is_err());
    assert_eq!(scene.nodes.len(), 0);
}

#[test]
fn test_add_instance_node_with_invalid_parent_fails() {
    let mut scene = Scene::new();

    let result = scene.add_instance_node(
        Some(999),
        0,
        0,
        None,
        Point3::new(0.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    );

    assert!(result.is_err());
    // No node should be created
    assert_eq!(scene.nodes.len(), 0);
    // But instance is created before the node validation - this is a side effect
    // that could be improved in the future
}

// ========================================================================
// Visibility Tests
// ========================================================================

#[test]
fn test_set_node_visibility_to_invisible() {
    let mut scene = Scene::new();
    let node = scene.add_default_node(None, None).unwrap();

    // Default is visible
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Visible);

    // Set to invisible
    scene.set_node_visibility(node, Visibility::Invisible);
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_set_node_visibility_to_visible() {
    let mut scene = Scene::new();
    let node = scene.add_default_node(None, None).unwrap();

    // Set to invisible first
    scene.set_node_visibility(node, Visibility::Invisible);
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Invisible);

    // Set back to visible
    scene.set_node_visibility(node, Visibility::Visible);
    assert_eq!(scene.get_node(node).unwrap().visibility(), Visibility::Visible);
}

#[test]
fn test_visibility_propagation_to_children() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(child1), None).unwrap();

    // All default to visible
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(child1).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(child2).unwrap().visibility(), Visibility::Visible);

    // Set root to invisible - should propagate to all descendants
    scene.set_node_visibility(root, Visibility::Invisible);
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child1).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child2).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_visibility_propagation_to_multiple_children() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(root), None).unwrap();
    let child3 = scene.add_default_node(Some(root), None).unwrap();

    // Set root invisible - all children should become invisible
    scene.set_node_visibility(root, Visibility::Invisible);
    assert_eq!(scene.get_node(child1).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child2).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(child3).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_visibility_no_propagation_when_setting_visible() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let child = scene.add_default_node(Some(root), None).unwrap();

    // Set root invisible (propagates to child)
    scene.set_node_visibility(root, Visibility::Invisible);
    assert_eq!(scene.get_node(child).unwrap().visibility(), Visibility::Invisible);

    // Set root visible - child should stay invisible
    scene.set_node_visibility(root, Visibility::Visible);
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(child).unwrap().visibility(), Visibility::Invisible);
}

#[test]
fn test_effective_visibility_all_visible() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(child1), None).unwrap();

    // All nodes visible - effective visibility should be Visible
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Visible);
    assert_eq!(scene.node_effective_visibility(child1), EffectiveVisibility::Visible);
    assert_eq!(scene.node_effective_visibility(child2), EffectiveVisibility::Visible);
}

#[test]
fn test_effective_visibility_node_invisible() {
    let mut scene = Scene::new();
    let node = scene.add_default_node(None, None).unwrap();

    scene.set_node_visibility(node, Visibility::Invisible);
    assert_eq!(scene.node_effective_visibility(node), EffectiveVisibility::Invisible);
}

#[test]
fn test_effective_visibility_mixed() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let child1 = scene.add_default_node(Some(root), None).unwrap();
    let child2 = scene.add_default_node(Some(root), None).unwrap();

    // Make child1 invisible
    scene.set_node_visibility(child1, Visibility::Invisible);

    // Root should have Mixed effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Mixed);
    // child1 is invisible
    assert_eq!(scene.node_effective_visibility(child1), EffectiveVisibility::Invisible);
    // child2 is visible (leaf node, no children)
    assert_eq!(scene.node_effective_visibility(child2), EffectiveVisibility::Visible);
}

#[test]
fn test_effective_visibility_caching() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let _child = scene.add_default_node(Some(root), None).unwrap();

    // First call computes and caches
    let effective1 = scene.node_effective_visibility(root);
    assert_eq!(effective1, EffectiveVisibility::Visible);

    // Node should have cached value
    assert!(!scene.get_node(root).unwrap().effective_visibility_dirty());

    // Second call should use cache
    let effective2 = scene.node_effective_visibility(root);
    assert_eq!(effective2, EffectiveVisibility::Visible);
}

#[test]
fn test_effective_visibility_cache_invalidation_on_child_change() {
    let mut scene = Scene::new();
    let root = scene.add_default_node(None, None).unwrap();
    let child = scene.add_default_node(Some(root), None).unwrap();

    // Compute and cache root's effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Visible);
    assert!(!scene.get_node(root).unwrap().effective_visibility_dirty());

    // Change child visibility - should invalidate root's cache
    scene.set_node_visibility(child, Visibility::Invisible);
    assert!(scene.get_node(root).unwrap().effective_visibility_dirty());

    // Now root should have Mixed effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Mixed);
}

#[test]
fn test_deep_hierarchy_visibility_propagation() {
    let mut scene = Scene::new();
    let level0 = scene.add_default_node(None, None).unwrap();
    let level1 = scene.add_default_node(Some(level0), None).unwrap();
    let level2 = scene.add_default_node(Some(level1), None).unwrap();
    let level3 = scene.add_default_node(Some(level2), None).unwrap();
    let level4 = scene.add_default_node(Some(level3), None).unwrap();

    // Set level2 invisible - should propagate to level3 and level4
    scene.set_node_visibility(level2, Visibility::Invisible);

    assert_eq!(scene.get_node(level0).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(level1).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(level2).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(level3).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(level4).unwrap().visibility(), Visibility::Invisible);

    // Effective visibility
    assert_eq!(scene.node_effective_visibility(level0), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(level1), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(level2), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(level3), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(level4), EffectiveVisibility::Invisible);
}

#[test]
fn test_visibility_with_complex_tree() {
    let mut scene = Scene::new();

    // Create tree:
    //     root
    //    /  |  \
    //   a   b   c
    //  / \     / \
    // d   e   f   g

    let root = scene.add_default_node(None, None).unwrap();
    let a = scene.add_default_node(Some(root), None).unwrap();
    let b = scene.add_default_node(Some(root), None).unwrap();
    let c = scene.add_default_node(Some(root), None).unwrap();
    let d = scene.add_default_node(Some(a), None).unwrap();
    let e = scene.add_default_node(Some(a), None).unwrap();
    let f = scene.add_default_node(Some(c), None).unwrap();
    let g = scene.add_default_node(Some(c), None).unwrap();

    // Make 'a' invisible (affects d and e)
    scene.set_node_visibility(a, Visibility::Invisible);

    // Make 'f' invisible
    scene.set_node_visibility(f, Visibility::Invisible);

    // Check explicit visibility
    assert_eq!(scene.get_node(root).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(a).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(b).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(c).unwrap().visibility(), Visibility::Visible);
    assert_eq!(scene.get_node(d).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(e).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(f).unwrap().visibility(), Visibility::Invisible);
    assert_eq!(scene.get_node(g).unwrap().visibility(), Visibility::Visible);

    // Check effective visibility
    assert_eq!(scene.node_effective_visibility(root), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(a), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(b), EffectiveVisibility::Visible);
    assert_eq!(scene.node_effective_visibility(c), EffectiveVisibility::Mixed);
    assert_eq!(scene.node_effective_visibility(d), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(e), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(f), EffectiveVisibility::Invisible);
    assert_eq!(scene.node_effective_visibility(g), EffectiveVisibility::Visible);
}

#[test]
fn test_visibility_leaf_node_always_visible_or_invisible() {
    let mut scene = Scene::new();
    let leaf = scene.add_default_node(None, None).unwrap();

    // Leaf nodes can only be Visible or Invisible, never Mixed
    assert_eq!(scene.node_effective_visibility(leaf), EffectiveVisibility::Visible);

    scene.set_node_visibility(leaf, Visibility::Invisible);
    assert_eq!(scene.node_effective_visibility(leaf), EffectiveVisibility::Invisible);
}

// ========================================================================
// Mesh Transform Methods
// ========================================================================

#[test]
fn test_mesh_translate() {
    let mut mesh = Mesh::sphere(1.0, 8, 4, PrimitiveType::LineList);
    let gen_before = mesh.generation();

    mesh.translate(Vector3::new(5.0, 0.0, 0.0));

    assert!(mesh.generation() > gen_before);
    // All vertices should have x >= 4.0 (radius 1.0 + offset 5.0)
    for v in mesh.vertices() {
        assert!(v.position[0] >= 4.0 - EPSILON);
    }
}

#[test]
fn test_mesh_translated_chaining() {
    let mesh = Mesh::sphere(1.0, 8, 4, PrimitiveType::TriangleList)
        .translated(Vector3::new(0.0, 10.0, 0.0));

    // All vertices should have y offset by 10
    for v in mesh.vertices() {
        assert!(v.position[1] >= 9.0 - EPSILON);
    }
}

#[test]
fn test_mesh_transform_identity() {
    let original = Mesh::cube(1.0, PrimitiveType::TriangleList);
    let original_positions: Vec<[f32; 3]> = original.vertices().iter().map(|v| v.position).collect();

    let mut transformed = original.clone();
    transformed.transform(&Matrix4::identity());

    for (orig, trans) in original_positions.iter().zip(transformed.vertices()) {
        assert!((orig[0] - trans.position[0]).abs() < EPSILON);
        assert!((orig[1] - trans.position[1]).abs() < EPSILON);
        assert!((orig[2] - trans.position[2]).abs() < EPSILON);
    }
}

#[test]
fn test_mesh_transform_translation() {
    let mut mesh = Mesh::cube(1.0, PrimitiveType::TriangleList);
    let offset = Vector3::new(3.0, 4.0, 5.0);
    mesh.transform(&Matrix4::from_translation(offset));

    let bounds = mesh.bounding().unwrap();
    // Cube was [-0.5, 0.5]^3, now should be [2.5, 3.5] x [3.5, 4.5] x [4.5, 5.5]
    assert!((bounds.min.x - 2.5).abs() < EPSILON);
    assert!((bounds.min.y - 3.5).abs() < EPSILON);
    assert!((bounds.min.z - 4.5).abs() < EPSILON);
}

// ========================================================================
// Mesh::cone_directed
// ========================================================================

#[test]
fn test_cone_directed_apex_position() {
    let apex = Point3::new(1.0, 2.0, 3.0);
    let direction = Vector3::new(0.0, -1.0, 0.0);
    let mesh = Mesh::cone_directed(apex, direction, 0.5, 2.0, 8, false, PrimitiveType::LineList);

    // At least one vertex should be very close to the apex
    let has_apex_vertex = mesh.vertices().iter().any(|v| {
        let dx = v.position[0] - apex.x;
        let dy = v.position[1] - apex.y;
        let dz = v.position[2] - apex.z;
        (dx * dx + dy * dy + dz * dz).sqrt() < 0.1
    });
    assert!(has_apex_vertex, "No vertex found near the apex");
}

#[test]
fn test_cone_directed_base_along_direction() {
    let apex = Point3::new(0.0, 0.0, 0.0);
    let direction = Vector3::new(0.0, 0.0, 1.0);
    let height = 3.0;
    let mesh = Mesh::cone_directed(apex, direction, 1.0, height, 16, false, PrimitiveType::TriangleList);

    let bounds = mesh.bounding().unwrap();
    // The cone extends from the apex along direction.
    // Bounding box max.z should be past the apex (into positive z).
    assert!(bounds.max.z > 0.0, "Cone should extend in +Z direction");
    // And min.z should be near the apex
    assert!(bounds.min.z < height, "Cone min.z should be less than height");
}

// ========================================================================
// Point Light Annotation
// ========================================================================

#[test]
fn test_point_light_annotation_to_mesh_data() {
    let ann = annotation::PointLightAnnotation {
        meta: annotation::AnnotationMeta { id: 0, name: None, visible: true, node_id: None },
        light_index: 0,
        radius: 0.5,
        segments: 8,
        reified_generation: None,
    };

    let data = ann.to_mesh_data(Point3::new(1.0, 2.0, 3.0), RgbaColor::YELLOW);
    assert!(data.mesh.has_primitive_type(PrimitiveType::LineList));
    assert!(data.mesh.vertices().len() > 0);
    // Wireframe sphere: 3+ rings of 8 segments each
    assert!(data.mesh.index_count(PrimitiveType::LineList) > 0);
}

#[test]
fn test_point_light_annotation_reification() {
    let mut scene = Scene::new();
    scene.lights.push(Light::point(
        Vector3::new(1.0, 2.0, 3.0),
        RgbaColor::WHITE,
        1.0,
    ));
    scene.light_generation += 1;

    let id = scene.annotations.add_point_light(0, 0.5, 8);
    scene.reify_annotations();

    let annotation = scene.annotations.get(id).unwrap();
    assert!(annotation.is_reified());
    assert!(annotation.node_id().is_some());
}

// ========================================================================
// Spot Light Annotation
// ========================================================================

#[test]
fn test_spot_light_annotation_to_mesh_data() {
    let ann = annotation::SpotLightAnnotation {
        meta: annotation::AnnotationMeta { id: 0, name: None, visible: true, node_id: None },
        light_index: 0,
        length: 5.0,
        segments: 8,
        reified_generation: None,
    };

    let meshes = ann.to_mesh_data(
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        RgbaColor::RED,
        0.3,  // inner
        0.6,  // outer
    );
    // Should produce 2 meshes: outer and inner cone
    assert_eq!(meshes.len(), 2);
    for data in &meshes {
        assert!(data.mesh.has_primitive_type(PrimitiveType::LineList));
    }
}

#[test]
fn test_spot_light_annotation_reification() {
    let mut scene = Scene::new();
    scene.lights.push(Light::spot(
        Vector3::new(0.0, 5.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        RgbaColor::WHITE,
        1.0,
        0.3,
        0.6,
    ));
    scene.light_generation += 1;

    let id = scene.annotations.add_spot_light(0, 5.0, 8);
    scene.reify_annotations();

    let annotation = scene.annotations.get(id).unwrap();
    assert!(annotation.is_reified());
}

// ========================================================================
// Normals Annotation
// ========================================================================

#[test]
fn test_normals_annotation_to_mesh_data() {
    let ann = annotation::NormalsAnnotation {
        meta: annotation::AnnotationMeta { id: 0, name: None, visible: true, node_id: None },
        target_node_id: 0,
        color: RgbaColor::CYAN,
        length: 0.1,
        reified_generation: None,
    };

    let vertices = vec![
        (Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
        (Point3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
    ];

    let meshes = ann.to_mesh_data(&vertices);
    assert_eq!(meshes.len(), 1);
    // 2 normals = 4 vertices, 4 indices
    assert_eq!(meshes[0].mesh.vertices().len(), 4);
    assert_eq!(meshes[0].mesh.index_count(PrimitiveType::LineList), 4);
}

#[test]
fn test_normals_annotation_empty_vertices() {
    let ann = annotation::NormalsAnnotation {
        meta: annotation::AnnotationMeta { id: 0, name: None, visible: true, node_id: None },
        target_node_id: 0,
        color: RgbaColor::CYAN,
        length: 0.1,
        reified_generation: None,
    };

    let meshes = ann.to_mesh_data(&[]);
    assert!(meshes.is_empty());
}

#[test]
fn test_normals_annotation_line_direction() {
    let ann = annotation::NormalsAnnotation {
        meta: annotation::AnnotationMeta { id: 0, name: None, visible: true, node_id: None },
        target_node_id: 0,
        color: RgbaColor::CYAN,
        length: 0.5,
        reified_generation: None,
    };

    let vertices = vec![
        (Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
    ];

    let meshes = ann.to_mesh_data(&vertices);
    let verts = meshes[0].mesh.vertices();
    // First vertex: position at origin
    assert!((verts[0].position[0]).abs() < EPSILON);
    assert!((verts[0].position[1]).abs() < EPSILON);
    // Second vertex: position at (0, 0.5, 0) — origin + normal * length
    assert!((verts[1].position[0]).abs() < EPSILON);
    assert!((verts[1].position[1] - 0.5).abs() < EPSILON);
    assert!((verts[1].position[2]).abs() < EPSILON);
}

#[test]
fn test_normals_annotation_reification() {
    let mut scene = Scene::new();

    // Create a simple triangle mesh
    let mesh = Mesh::from_raw(
        vec![
            Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] },
            Vertex { position: [1.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] },
            Vertex { position: [0.5, 1.0, 0.0], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] },
        ],
        vec![MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: vec![0, 1, 2],
        }],
    );
    let mesh_id = scene.add_mesh(mesh);
    let mat_id = scene.add_material(Material::new());
    let node_id = scene.add_instance_node(
        None, mesh_id, mat_id, None,
        Point3::new(0.0, 0.0, 0.0),
        Quaternion::new(1.0, 0.0, 0.0, 0.0),
        Vector3::new(1.0, 1.0, 1.0),
    ).unwrap();

    let ann_id = scene.annotations.add_normals(node_id, RgbaColor::CYAN, 0.1);
    scene.reify_annotations();

    let annotation = scene.annotations.get(ann_id).unwrap();
    assert!(annotation.is_reified());
}

// ========================================================================
// Staleness Detection and Re-reification
// ========================================================================

#[test]
fn test_point_light_staleness_detection() {
    let mut scene = Scene::new();
    scene.lights.push(Light::point(
        Vector3::new(0.0, 0.0, 0.0),
        RgbaColor::WHITE,
        1.0,
    ));
    scene.light_generation += 1;

    let id = scene.annotations.add_point_light(0, 0.5, 8);
    let reified_count = scene.reify_annotations();
    assert_eq!(reified_count, 1);

    // Re-reify with no changes — should not re-reify
    let reified_count = scene.reify_annotations();
    assert_eq!(reified_count, 0);

    // Bump generation to simulate light change
    scene.light_generation += 1;

    // Should detect staleness and re-reify
    let reified_count = scene.reify_annotations();
    assert_eq!(reified_count, 1);

    let annotation = scene.annotations.get(id).unwrap();
    assert!(annotation.is_reified());
}

#[test]
fn test_wrong_light_type_returns_none() {
    let mut scene = Scene::new();
    // Add a directional light, but try to annotate it as a point light
    scene.lights.push(Light::directional(
        Vector3::new(0.0, -1.0, 0.0),
        RgbaColor::WHITE,
        1.0,
    ));
    scene.light_generation += 1;

    let id = scene.annotations.add_point_light(0, 0.5, 8);
    let result = scene.reify_annotation(id);
    assert!(result.is_none());
}

#[test]
fn test_light_index_out_of_bounds_returns_none() {
    let mut scene = Scene::new();
    // No lights added, index 0 is out of bounds
    let id = scene.annotations.add_point_light(0, 0.5, 8);
    let result = scene.reify_annotation(id);
    assert!(result.is_none());
}
