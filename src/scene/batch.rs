use super::{tree::InstanceTransform, MeshId, PrimitiveType};
use crate::material::MaterialId;

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

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }
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

        let instance_transform = InstanceTransform::new(1, Matrix4::identity());
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
            let instance_transform = InstanceTransform::new(i, Matrix4::identity());
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

        batch.add_instance(InstanceTransform::new(1, Matrix4::identity()));
        assert_eq!(batch.len(), 1);

        batch.add_instance(InstanceTransform::new(2, Matrix4::identity()));
        assert_eq!(batch.len(), 2);

        batch.add_instance(InstanceTransform::new(3, Matrix4::identity()));
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_draw_batch_instances_with_different_transforms() {
        let mut batch = DrawBatch::new(10, 5, PrimitiveType::TriangleList);

        let transform1 = Matrix4::from_scale(1.0);
        let transform2 = Matrix4::from_scale(2.0);
        let transform3 = Matrix4::from_translation(cgmath::Vector3::new(5.0, 0.0, 0.0));

        batch.add_instance(InstanceTransform::new(1, transform1));
        batch.add_instance(InstanceTransform::new(2, transform2));
        batch.add_instance(InstanceTransform::new(3, transform3));

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
            batch.add_instance(InstanceTransform::new(i, Matrix4::identity()));
        }

        assert_eq!(batch.len(), 1000);
        assert!(!batch.is_empty());
    }
}
