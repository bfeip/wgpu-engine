use super::material::MaterialId;
use super::mesh::MeshId;

/// Unique identifier for a mesh instance
pub type InstanceId = u32;

/// An instance references a mesh and material to be rendered.
#[derive(Clone)]
pub struct Instance {
    pub id: InstanceId,
    pub mesh: MeshId,
    pub material: MaterialId,
}

impl Instance {
    /// Creates a new instance referencing the given mesh and material.
    pub fn new(id: InstanceId, mesh: MeshId, material: MaterialId) -> Self {
        Self {
            id,
            mesh,
            material,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_new() {
        let instance = Instance::new(42, 10, 5);

        assert_eq!(instance.id, 42);
        assert_eq!(instance.mesh, 10);
        assert_eq!(instance.material, 5);
    }

    #[test]
    fn test_instance_id_unique() {
        let instance1 = Instance::new(1, 10, 5);
        let instance2 = Instance::new(2, 10, 5);
        let instance3 = Instance::new(3, 10, 5);

        assert_ne!(instance1.id, instance2.id);
        assert_ne!(instance1.id, instance3.id);
        assert_ne!(instance2.id, instance3.id);
    }

    #[test]
    fn test_instance_mesh_reference() {
        let instance = Instance::new(1, 42, 5);
        assert_eq!(instance.mesh, 42);

        let instance2 = Instance::new(2, 99, 5);
        assert_eq!(instance2.mesh, 99);
    }

    #[test]
    fn test_instance_material_reference() {
        let instance = Instance::new(1, 10, 7);
        assert_eq!(instance.material, 7);

        let instance2 = Instance::new(2, 10, 13);
        assert_eq!(instance2.material, 13);
    }

    #[test]
    fn test_instance_same_mesh_different_material() {
        let instance1 = Instance::new(1, 10, 5);
        let instance2 = Instance::new(2, 10, 7);

        assert_eq!(instance1.mesh, instance2.mesh);
        assert_ne!(instance1.material, instance2.material);
    }

    #[test]
    fn test_instance_same_material_different_mesh() {
        let instance1 = Instance::new(1, 10, 5);
        let instance2 = Instance::new(2, 15, 5);

        assert_ne!(instance1.mesh, instance2.mesh);
        assert_eq!(instance1.material, instance2.material);
    }
}