use std::collections::HashMap;

use cgmath::{Matrix3, Matrix4, SquareMatrix};

use cgmath::Point3;

use crate::scene::{
    AlphaMode, InstanceId, MaterialId, MeshId, NodeId, PrimitiveType, Scene,
    Visibility,
};
use crate::scene::common;

/// Represents an instance with its computed world transform.
#[derive(Clone)]
pub(crate) struct InstanceTransform {
    pub node_id: NodeId,
    pub instance_id: InstanceId,
    pub world_transform: Matrix4<f32>,
    pub normal_matrix: Matrix3<f32>,
}

impl InstanceTransform {
    /// Creates a new InstanceTransform with the given world transform.
    /// The normal matrix is computed from the world transform.
    pub fn new(node_id: NodeId, instance_id: InstanceId, world_transform: Matrix4<f32>) -> Self {
        let normal_matrix = common::compute_normal_matrix(&world_transform);
        Self {
            node_id,
            instance_id,
            world_transform,
            normal_matrix,
        }
    }
}

/// Represents a batch of instances that share the same mesh, material, and primitive type.
///
/// Batching allows us to minimize draw calls and state changes by grouping
/// instances that can be rendered together.
pub(crate) struct DrawBatch {
    pub mesh_id: MeshId,
    pub material_id: MaterialId,
    pub primitive_type: PrimitiveType,
    pub instances: Vec<InstanceTransform>,
}

impl DrawBatch {
    pub fn new(mesh_id: MeshId, material_id: MaterialId, primitive_type: PrimitiveType) -> Self {
        Self {
            mesh_id,
            material_id,
            primitive_type,
            instances: Vec::new(),
        }
    }

    pub fn add_instance(&mut self, instance_transform: InstanceTransform) {
        self.instances.push(instance_transform);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// Recursively collects instance transforms by walking the scene tree.
///
/// Propagates world transforms top-down, caching them on nodes, and collects
/// instances with their computed world transforms. Skips invisible subtrees.
fn collect_transforms_recursive(
    scene: &Scene,
    node_id: NodeId,
    parent_transform: Matrix4<f32>,
    parent_changed: bool,
    results: &mut Vec<InstanceTransform>,
) {
    let Some(node) = scene.get_node(node_id) else { return };

    let needs_recompute = parent_changed || node.transform_dirty();
    let world_transform = if needs_recompute {
        let wt = parent_transform * node.compute_local_transform();
        node.set_cached_world_transform(wt);
        wt
    } else {
        node.cached_world_transform().unwrap()
    };

    if node.visibility() == Visibility::Invisible {
        return;
    }

    if let Some(instance_id) = node.instance() {
        results.push(InstanceTransform::new(node.id, instance_id, world_transform));
    }

    for &child_id in node.children() {
        collect_transforms_recursive(scene, child_id, world_transform, needs_recompute, results);
    }
}

/// Walks the entire scene tree and collects all instances with their world transforms.
pub(crate) fn collect_instance_transforms(scene: &Scene) -> Vec<InstanceTransform> {
    let mut results = Vec::new();
    for &root_id in scene.root_nodes() {
        collect_transforms_recursive(scene, root_id, Matrix4::identity(), false, &mut results);
    }
    results
}

/// Collects all instances grouped into batches by mesh, material, and primitive type.
///
/// This walks the scene tree, computes world transforms, and groups
/// instances that share the same mesh, material, and primitive type into batches.
/// Each mesh can have multiple primitive types (triangles, lines, points), so
/// a single instance may generate multiple batches.
/// Batches are sorted to minimize state changes during rendering:
/// 1. By material ID (to minimize bind group changes)
/// 2. By primitive type (to minimize pipeline changes)
/// 3. By mesh ID (for GPU cache locality)
pub(crate) fn collect_draw_batches(scene: &Scene) -> Vec<DrawBatch> {
    let instance_transforms = collect_instance_transforms(scene);
    let mut batch_map: HashMap<(MeshId, MaterialId, PrimitiveType), DrawBatch> = HashMap::new();

    for inst_transform in instance_transforms {
        let Some(instance) = scene.get_instance(inst_transform.instance_id) else {
            continue;
        };
        let Some(mesh) = scene.get_mesh(instance.mesh()) else {
            continue;
        };

        // Create a separate batch for each primitive type the mesh supports
        for primitive_type in [
            PrimitiveType::TriangleList,
            PrimitiveType::LineList,
            PrimitiveType::PointList,
        ] {
            if !mesh.has_primitive_type(primitive_type) {
                continue;
            }

            let key = (instance.mesh(), instance.material(), primitive_type);
            batch_map
                .entry(key)
                .or_insert_with(|| DrawBatch::new(instance.mesh(), instance.material(), primitive_type))
                .add_instance(inst_transform.clone());
        }
    }

    // Convert to Vec and sort for optimal rendering
    let mut batches: Vec<DrawBatch> = batch_map.into_values().collect();
    batches.sort_by_key(|b| (b.material_id, b.primitive_type as u8, b.mesh_id));
    batches
}

/// Reorders batches so opaque/mask batches come first (preserving existing sort),
/// followed by transparent (Blend) batches sorted back-to-front by distance from camera.
pub(crate) fn sort_batches_for_transparency(
    batches: &mut Vec<DrawBatch>,
    scene: &Scene,
    camera_position: Point3<f32>,
) {
    // Stable partition: opaque/mask first, then blend
    batches.sort_by(|a, b| {
        let a_transparent = is_transparent_batch(a, scene);
        let b_transparent = is_transparent_batch(b, scene);
        match (a_transparent, b_transparent) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            (true, true) => {
                // Both transparent: sort back-to-front (farthest first)
                let dist_a = batch_centroid_distance_sq(a, camera_position);
                let dist_b = batch_centroid_distance_sq(b, camera_position);
                dist_b
                    .partial_cmp(&dist_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            (false, false) => std::cmp::Ordering::Equal,
        }
    });
}

fn is_transparent_batch(batch: &DrawBatch, scene: &Scene) -> bool {
    scene
        .get_material(batch.material_id)
        .map(|m| m.alpha_mode() == AlphaMode::Blend)
        .unwrap_or(false)
}

fn batch_centroid_distance_sq(batch: &DrawBatch, camera_position: Point3<f32>) -> f32 {
    if batch.instances.is_empty() {
        return 0.0;
    }
    let count = batch.instances.len() as f32;
    let (sx, sy, sz) = batch.instances.iter().fold((0.0f32, 0.0f32, 0.0f32), |(x, y, z), i| {
        (
            x + i.world_transform[3][0],
            y + i.world_transform[3][1],
            z + i.world_transform[3][2],
        )
    });
    let cx = sx / count - camera_position.x;
    let cy = sy / count - camera_position.y;
    let cz = sz / count - camera_position.z;
    cx * cx + cy * cy + cz * cz
}

/// Partitions batches by a predicate on instances.
///
/// Takes a list of batches and splits each batch's instances based on the predicate.
/// Returns two sets of batches: those where the predicate returned `true` and those
/// where it returned `false`.
///
/// Batches are preserved with their mesh/material/primitive type, but may be split
/// if instances within a batch have different predicate results.
///
/// Empty batches (after partitioning) are not included in the output.
pub(crate) fn partition_batches<F>(batches: &[DrawBatch], predicate: F) -> (Vec<DrawBatch>, Vec<DrawBatch>)
where
    F: Fn(&InstanceTransform) -> bool,
{
    type BatchKey = (MeshId, MaterialId, PrimitiveType);

    let mut matched: HashMap<BatchKey, DrawBatch> = HashMap::new();
    let mut unmatched: HashMap<BatchKey, DrawBatch> = HashMap::new();

    for batch in batches {
        let key = (batch.mesh_id, batch.material_id, batch.primitive_type);

        for instance in &batch.instances {
            let target = if predicate(instance) {
                matched.entry(key).or_insert_with(|| {
                    DrawBatch::new(batch.mesh_id, batch.material_id, batch.primitive_type)
                })
            } else {
                unmatched.entry(key).or_insert_with(|| {
                    DrawBatch::new(batch.mesh_id, batch.material_id, batch.primitive_type)
                })
            };
            target.add_instance(instance.clone());
        }
    }

    (matched.into_values().collect(), unmatched.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Deg, Matrix4, Quaternion, Rotation3, SquareMatrix, Vector3};
    use crate::scene::common::EPSILON;

    // ========================================================================
    // InstanceTransform Tests
    // ========================================================================

    #[test]
    fn test_instance_transform_creation() {
        let transform = Matrix4::from_scale(2.0);
        let instance_transform = InstanceTransform::new(1, 42, transform);

        assert_eq!(instance_transform.node_id, 1);
        assert_eq!(instance_transform.instance_id, 42);
        assert_eq!(instance_transform.world_transform, transform);
    }

    #[test]
    fn test_instance_transform_identity() {
        let identity = Matrix4::identity();
        let instance_transform = InstanceTransform::new(1, 1, identity);

        // Verify identity transform
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(instance_transform.world_transform[i][j], 1.0);
                } else {
                    assert_eq!(instance_transform.world_transform[i][j], 0.0);
                }
            }
        }

        // Verify identity normal matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(instance_transform.normal_matrix[i][j], 1.0);
                } else {
                    assert_eq!(instance_transform.normal_matrix[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_instance_transform_normal_matrix_computed() {
        let transform = Matrix4::from_scale(2.0);
        let instance_transform = InstanceTransform::new(1, 1, transform);

        // Normal matrix should be inverse-transpose
        // For uniform scale of 2.0, normal matrix should be 0.5
        assert!((instance_transform.normal_matrix[0][0] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[1][1] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[2][2] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_translation() {
        let transform = Matrix4::from_translation(Vector3::new(5.0, 10.0, 15.0));
        let instance_transform = InstanceTransform::new(1, 1, transform);

        // Translation should be in the transform
        assert_eq!(instance_transform.world_transform[3][0], 5.0);
        assert_eq!(instance_transform.world_transform[3][1], 10.0);
        assert_eq!(instance_transform.world_transform[3][2], 15.0);

        // Normal matrix should remain identity (translation doesn't affect normals)
        assert_eq!(instance_transform.normal_matrix[0][0], 1.0);
        assert_eq!(instance_transform.normal_matrix[1][1], 1.0);
        assert_eq!(instance_transform.normal_matrix[2][2], 1.0);
    }

    #[test]
    fn test_instance_transform_rotation() {
        let rotation = Quaternion::from_angle_z(Deg(90.0));
        let transform = Matrix4::from(rotation);
        let instance_transform = InstanceTransform::new(1, 1, transform);

        // Normal matrix should match rotation (orthogonal matrices)
        // For 90 degree Z rotation: (1,0,0) -> (0,1,0)
        let normal = instance_transform.normal_matrix;

        // Check that applying normal matrix to (1,0,0) gives approximately (0,1,0)
        let x = normal[0][0] * 1.0 + normal[1][0] * 0.0 + normal[2][0] * 0.0;
        let y = normal[0][1] * 1.0 + normal[1][1] * 0.0 + normal[2][1] * 0.0;

        assert!(x.abs() < EPSILON);
        assert!((y - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_non_uniform_scale() {
        let transform = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let instance_transform = InstanceTransform::new(1, 1, transform);

        // Normal matrix should handle non-uniform scale correctly
        // Inverse of diagonal matrix (2,3,4) is (0.5, 0.333..., 0.25)
        assert!((instance_transform.normal_matrix[0][0] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[1][1] - 1.0/3.0).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[2][2] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_different_instances_different_ids() {
        let transform = Matrix4::identity();
        let instance1 = InstanceTransform::new(1, 1, transform);
        let instance2 = InstanceTransform::new(2, 2, transform);
        let instance3 = InstanceTransform::new(3, 3, transform);

        assert_ne!(instance1.instance_id, instance2.instance_id);
        assert_ne!(instance1.instance_id, instance3.instance_id);
        assert_ne!(instance2.instance_id, instance3.instance_id);
    }

    #[test]
    fn test_instance_transform_with_complex_transform() {
        // Combine translation, rotation, and scale
        let translation = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0));
        let rotation = Matrix4::from(Quaternion::from_angle_y(Deg(45.0)));
        let scale = Matrix4::from_scale(2.0);
        let transform = translation * rotation * scale;

        let instance_transform = InstanceTransform::new(1, 42, transform);

        assert_eq!(instance_transform.node_id, 1);
        assert_eq!(instance_transform.instance_id, 42);
        assert_eq!(instance_transform.world_transform, transform);

        // Verify normal matrix was computed (not identity)
        let expected_normal = common::compute_normal_matrix(&transform);

        for i in 0..3 {
            for j in 0..3 {
                assert!((instance_transform.normal_matrix[i][j] - expected_normal[i][j]).abs() < EPSILON);
            }
        }
    }

    // ========================================================================
    // DrawBatch Tests
    // ========================================================================

    #[test]
    fn test_draw_batch_new() {
        let batch = DrawBatch::new(10, 5, PrimitiveType::TriangleList);

        assert_eq!(batch.mesh_id, 10);
        assert_eq!(batch.material_id, 5);
        assert_eq!(batch.primitive_type, PrimitiveType::TriangleList);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_draw_batch_add_instance() {
        let mut batch = DrawBatch::new(10, 5, PrimitiveType::TriangleList);

        let instance_transform = InstanceTransform::new(1, 1, Matrix4::identity());
        batch.add_instance(instance_transform);

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.instances[0].instance_id, 1);
    }

    #[test]
    fn test_draw_batch_add_multiple_instances() {
        let mut batch = DrawBatch::new(10, 5, PrimitiveType::TriangleList);

        // Add 5 instances
        for i in 0..5 {
            let instance_transform = InstanceTransform::new(i, i, Matrix4::identity());
            batch.add_instance(instance_transform);
        }

        assert_eq!(batch.len(), 5);
        assert!(!batch.is_empty());

        // Verify all instances were added
        for i in 0..5_u32 {
            assert_eq!(batch.instances[i as usize].instance_id, i);
        }
    }

    #[test]
    fn test_draw_batch_mesh_material_ids() {
        let batch1 = DrawBatch::new(10, 5, PrimitiveType::TriangleList);
        let batch2 = DrawBatch::new(20, 7, PrimitiveType::LineList);

        assert_eq!(batch1.mesh_id, 10);
        assert_eq!(batch1.material_id, 5);
        assert_eq!(batch1.primitive_type, PrimitiveType::TriangleList);

        assert_eq!(batch2.mesh_id, 20);
        assert_eq!(batch2.material_id, 7);
        assert_eq!(batch2.primitive_type, PrimitiveType::LineList);
    }

    #[test]
    fn test_draw_batch_instance_count() {
        let mut batch = DrawBatch::new(1, 1, PrimitiveType::TriangleList);

        assert_eq!(batch.len(), 0);

        batch.add_instance(InstanceTransform::new(1, 1, Matrix4::identity()));
        assert_eq!(batch.len(), 1);

        batch.add_instance(InstanceTransform::new(2, 2, Matrix4::identity()));
        assert_eq!(batch.len(), 2);

        batch.add_instance(InstanceTransform::new(3, 3, Matrix4::identity()));
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_draw_batch_instances_with_different_transforms() {
        let mut batch = DrawBatch::new(10, 5, PrimitiveType::TriangleList);

        let transform1 = Matrix4::from_scale(1.0);
        let transform2 = Matrix4::from_scale(2.0);
        let transform3 = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0));

        batch.add_instance(InstanceTransform::new(1, 1, transform1));
        batch.add_instance(InstanceTransform::new(2, 2, transform2));
        batch.add_instance(InstanceTransform::new(3, 3, transform3));

        assert_eq!(batch.len(), 3);

        // Verify transforms are preserved
        assert_eq!(batch.instances[0].world_transform, transform1);
        assert_eq!(batch.instances[1].world_transform, transform2);
        assert_eq!(batch.instances[2].world_transform, transform3);
    }

    #[test]
    fn test_draw_batch_large_number_of_instances() {
        let mut batch = DrawBatch::new(1, 1, PrimitiveType::TriangleList);

        // Add 1000 instances
        for i in 0..1000_u32 {
            batch.add_instance(InstanceTransform::new(i, i, Matrix4::identity()));
        }

        assert_eq!(batch.len(), 1000);
        assert!(!batch.is_empty());
    }

    // ========================================================================
    // partition_batches Tests
    // ========================================================================

    #[test]
    fn test_partition_batches_empty() {
        let batches: Vec<DrawBatch> = vec![];
        let (matched, unmatched) = partition_batches(&batches, |_| true);
        assert!(matched.is_empty());
        assert!(unmatched.is_empty());
    }

    #[test]
    fn test_partition_batches_all_match() {
        let mut batch = DrawBatch::new(1, 1, PrimitiveType::TriangleList);
        batch.add_instance(InstanceTransform::new(1, 1, Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(2, 2, Matrix4::identity()));

        let (matched, unmatched) = partition_batches(&[batch], |_| true);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].instances.len(), 2);
        assert!(unmatched.is_empty());
    }

    #[test]
    fn test_partition_batches_none_match() {
        let mut batch = DrawBatch::new(1, 1, PrimitiveType::TriangleList);
        batch.add_instance(InstanceTransform::new(1, 1, Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(2, 2, Matrix4::identity()));

        let (matched, unmatched) = partition_batches(&[batch], |_| false);
        assert!(matched.is_empty());
        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].instances.len(), 2);
    }

    #[test]
    fn test_partition_batches_split() {
        let mut batch = DrawBatch::new(1, 1, PrimitiveType::TriangleList);
        batch.add_instance(InstanceTransform::new(1, 1, Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(2, 2, Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(3, 3, Matrix4::identity()));

        // Partition by node_id: even nodes in one group, odd in another
        let (matched, unmatched) = partition_batches(&[batch], |inst| inst.node_id % 2 == 0);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].instances.len(), 1); // node 2 only
        assert_eq!(matched[0].instances[0].node_id, 2);

        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].instances.len(), 2); // nodes 1 and 3
    }
}
