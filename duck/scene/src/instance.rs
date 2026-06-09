use super::material::{FaceMaterialId, LineMaterialId, PointMaterialId};
use super::mesh::MeshId;

/// Unique identifier for a mesh instance
pub type InstanceId = crate::Id<Instance>;

/// An instance referencing a mesh and up to one material per primitive kind.
///
/// A mesh may contain triangle, line, and/or point primitives; each kind is
/// shaded by its own optional material slot. A slot left `None` means that
/// primitive kind is not drawn for this instance.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Instance {
    pub id: InstanceId,
    mesh: MeshId,
    face_material: Option<FaceMaterialId>,
    line_material: Option<LineMaterialId>,
    point_material: Option<PointMaterialId>,
}

impl Instance {
    /// Creates a new instance referencing the given mesh, with no materials assigned.
    ///
    /// Assign materials with [`Instance::with_face_material`] and friends.
    pub fn new(mesh: MeshId) -> Self {
        Self {
            id: InstanceId::new(),
            mesh,
            face_material: None,
            line_material: None,
            point_material: None,
        }
    }

    /// Sets the face material (chainable).
    pub fn with_face_material(mut self, material: FaceMaterialId) -> Self {
        self.face_material = Some(material);
        self
    }

    /// Sets the line material (chainable).
    pub fn with_line_material(mut self, material: LineMaterialId) -> Self {
        self.line_material = Some(material);
        self
    }

    /// Sets the point material (chainable).
    pub fn with_point_material(mut self, material: PointMaterialId) -> Self {
        self.point_material = Some(material);
        self
    }

    /// Returns the mesh ID referenced by this instance.
    pub fn mesh(&self) -> MeshId {
        self.mesh
    }

    /// Returns the face material ID, if assigned.
    pub fn face_material(&self) -> Option<FaceMaterialId> {
        self.face_material
    }

    /// Returns the line material ID, if assigned.
    pub fn line_material(&self) -> Option<LineMaterialId> {
        self.line_material
    }

    /// Returns the point material ID, if assigned.
    pub fn point_material(&self) -> Option<PointMaterialId> {
        self.point_material
    }

    /// Sets the mesh ID without validating it exists in the scene.
    pub fn set_mesh_unchecked(&mut self, mesh: MeshId) {
        self.mesh = mesh;
    }

    /// Sets the face material ID (or clears it) without validating it exists in the scene.
    pub fn set_face_material_unchecked(&mut self, material: Option<FaceMaterialId>) {
        self.face_material = material;
    }

    /// Sets the line material ID (or clears it) without validating it exists in the scene.
    pub fn set_line_material_unchecked(&mut self, material: Option<LineMaterialId>) {
        self.line_material = material;
    }

    /// Sets the point material ID (or clears it) without validating it exists in the scene.
    pub fn set_point_material_unchecked(&mut self, material: Option<PointMaterialId>) {
        self.point_material = material;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_new() {
        let mesh_id = MeshId::new();
        let mat_id = FaceMaterialId::new();
        let instance = Instance::new(mesh_id).with_face_material(mat_id);

        assert_eq!(instance.mesh(), mesh_id);
        assert_eq!(instance.face_material(), Some(mat_id));
    }

    #[test]
    fn test_instance_id_unique() {
        let instance1 = Instance::new(MeshId::new());
        let instance2 = Instance::new(MeshId::new());
        let instance3 = Instance::new(MeshId::new());

        assert_ne!(instance1.id, instance2.id);
        assert_ne!(instance1.id, instance3.id);
        assert_ne!(instance2.id, instance3.id);
    }

    #[test]
    fn test_instance_mesh_reference() {
        let mesh_id = MeshId::new();
        let instance = Instance::new(mesh_id);
        assert_eq!(instance.mesh(), mesh_id);
    }

    #[test]
    fn test_instance_material_slots() {
        let face = FaceMaterialId::new();
        let line = LineMaterialId::new();
        let instance = Instance::new(MeshId::new())
            .with_face_material(face)
            .with_line_material(line);
        assert_eq!(instance.face_material(), Some(face));
        assert_eq!(instance.line_material(), Some(line));
        assert_eq!(instance.point_material(), None);
    }

    #[test]
    fn test_instance_same_mesh_different_material() {
        let mesh_id = MeshId::new();
        let instance1 = Instance::new(mesh_id).with_face_material(FaceMaterialId::new());
        let instance2 = Instance::new(mesh_id).with_face_material(FaceMaterialId::new());

        assert_eq!(instance1.mesh, instance2.mesh);
        assert_ne!(instance1.face_material, instance2.face_material);
    }

    #[test]
    fn test_instance_same_material_different_mesh() {
        let mat_id = FaceMaterialId::new();
        let instance1 = Instance::new(MeshId::new()).with_face_material(mat_id);
        let instance2 = Instance::new(MeshId::new()).with_face_material(mat_id);

        assert_ne!(instance1.mesh, instance2.mesh);
        assert_eq!(instance1.face_material, instance2.face_material);
    }
}
