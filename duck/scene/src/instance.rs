use super::material::MaterialId;
use super::mesh::MeshId;

/// Unique identifier for a mesh instance
pub type InstanceId = crate::Id;

/// An instance referencing a mesh and material.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Instance {
    pub id: InstanceId,
    mesh: MeshId,
    material: MaterialId,
}

impl Instance {
    /// Creates a new instance referencing the given mesh and material.
    pub fn new(mesh: MeshId, material: MaterialId) -> Self {
        Self {
            id: crate::Id::new(),
            mesh,
            material,
        }
    }

    /// Returns the mesh ID referenced by this instance.
    pub fn mesh(&self) -> MeshId {
        self.mesh
    }

    /// Returns the material ID referenced by this instance.
    pub fn material(&self) -> MaterialId {
        self.material
    }

    /// Sets the mesh ID without validating it exists in the scene.
    pub fn set_mesh_unchecked(&mut self, mesh: MeshId) {
        self.mesh = mesh;
    }

    /// Sets the material ID without validating it exists in the scene.
    pub fn set_material_unchecked(&mut self, material: MaterialId) {
        self.material = material;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_instance_new() {
        let mesh_id = crate::Id::new();
        let mat_id = crate::Id::new();
        let instance = Instance::new(mesh_id, mat_id);

        assert_eq!(instance.mesh(), mesh_id);
        assert_eq!(instance.material(), mat_id);
    }

    #[test]
    fn test_instance_id_unique() {
        let instance1 = Instance::new(crate::Id::new(), crate::Id::new());
        let instance2 = Instance::new(crate::Id::new(), crate::Id::new());
        let instance3 = Instance::new(crate::Id::new(), crate::Id::new());

        assert_ne!(instance1.id, instance2.id);
        assert_ne!(instance1.id, instance3.id);
        assert_ne!(instance2.id, instance3.id);
    }

    #[test]
    fn test_instance_mesh_reference() {
        let mesh_id = crate::Id::new();
        let instance = Instance::new(mesh_id, crate::Id::new());
        assert_eq!(instance.mesh(), mesh_id);
    }

    #[test]
    fn test_instance_material_reference() {
        let mat_id = crate::Id::new();
        let instance = Instance::new(crate::Id::new(), mat_id);
        assert_eq!(instance.material(), mat_id);
    }

    #[test]
    fn test_instance_same_mesh_different_material() {
        let mesh_id = crate::Id::new();
        let instance1 = Instance::new(mesh_id, crate::Id::new());
        let instance2 = Instance::new(mesh_id, crate::Id::new());

        assert_eq!(instance1.mesh, instance2.mesh);
        assert_ne!(instance1.material, instance2.material);
    }

    #[test]
    fn test_instance_same_material_different_mesh() {
        let mat_id = crate::Id::new();
        let instance1 = Instance::new(crate::Id::new(), mat_id);
        let instance2 = Instance::new(crate::Id::new(), mat_id);

        assert_ne!(instance1.mesh, instance2.mesh);
        assert_eq!(instance1.material, instance2.material);
    }
}