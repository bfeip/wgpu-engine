mod mesh;
mod instance;

pub use mesh::{Mesh, MeshId, Vertex};
pub use instance::{Instance, InstanceId, InstanceRaw};

use crate::{
    common::RgbaColor,
    light::Light,
    material::MaterialId,
    scene::mesh::MeshDescriptor
};
use std::collections::HashMap;

pub struct Scene {
    pub meshes: HashMap<MeshId, Mesh>,
    pub instances: HashMap<InstanceId, Instance>,
    pub lights: Vec<Light>,

    next_mesh_id: MeshId,
    next_instance_id: InstanceId
}

impl Scene {
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            instances: HashMap::new(),
            lights: Vec::new(),

            next_mesh_id: 0,
            next_instance_id: 0
        }
    }

    pub fn add_mesh<P: AsRef<std::path::Path>>(
        &mut self,
        device: &wgpu::Device,
        desc: MeshDescriptor<P>,
        label: Option<&str>
    ) -> anyhow::Result<MeshId> {
        let id = self.next_mesh_id;
        self.next_mesh_id += 1;

        let mesh = Mesh::new(id, device, desc, label)?;
        self.meshes.insert(id, mesh);
        Ok(id)
    }

    pub fn add_instance(
        &mut self,
        mesh: MeshId,
        material: MaterialId,
        position: Option<cgmath::Point3<f32>>,
        rotation: Option<cgmath::Quaternion<f32>>
    ) -> InstanceId {
        let id = self.next_instance_id;
        self.next_instance_id += 1;

        let instance = Instance::new(id, mesh, material, position, rotation);
        self.instances.insert(id, instance);
        id
    }

    pub fn demo(device: &wgpu::Device, material_id: MaterialId) -> Self {
        use cgmath::Rotation3;

        let mut scene = Self::new();
        scene.lights = vec![
            Light::new(
                cgmath::Vector3 { x: 3., y: 3., z: 3. },
                RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
            )
        ];

        // Load a sample OBJ
        let monkey_bytes = include_bytes!("monkey.obj");
        let monkey_mesh_desc: MeshDescriptor<&str> = MeshDescriptor::Obj(mesh::ObjMesh::Bytes(monkey_bytes));
        let monkey_mesh = scene.add_mesh(device, monkey_mesh_desc, Some("monkey_mesh")).unwrap();
        scene.add_instance(
            monkey_mesh,
            0,
            Some(cgmath::Point3 { x: 0., y: 0., z: 0. }),
            None
        );
        scene.add_instance(
            monkey_mesh,
            material_id,
            Some(cgmath::Point3 { x: 2., y: 0., z: 0. }),
            Some(cgmath::Quaternion::from_angle_z(cgmath::Rad(3.14_f32)))
        );

        scene
    }
}