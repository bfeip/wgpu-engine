use super::{tree::InstanceTransform, MeshId, PrimitiveType};
use super::material::MaterialId;
use std::collections::HashMap;

/// Represents a batch of instances that share the same mesh, material, and primitive type.
///
/// Batching allows us to minimize draw calls and state changes by grouping
/// instances that can be rendered together.
pub struct DrawBatch {
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
///
/// # Example
/// ```ignore
/// let (selected, non_selected) = partition_batches(&batches, |inst| {
///     selection.is_node_selected(inst.node_id)
/// });
/// ```
pub fn partition_batches<F>(batches: &[DrawBatch], predicate: F) -> (Vec<DrawBatch>, Vec<DrawBatch>)
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
    use cgmath::{Matrix4, SquareMatrix};

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
        let transform3 = Matrix4::from_translation(cgmath::Vector3::new(5.0, 0.0, 0.0));

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
