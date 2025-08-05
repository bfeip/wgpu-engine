use crate::{common::RgbaColor, geometry::{Instance, Mesh}, light::Light};

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub lights: Vec<Light>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            lights: Vec::new()
        }
    }

    pub fn demo(device: &wgpu::Device) -> Self {
        use cgmath::Rotation3;

        let lights = vec![
            Light::new(
                cgmath::Vector3 { x: 3., y: 3., z: 3. },
                RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
            )
        ];

        // Load a sample OBJ
        let monkey_bytes = include_bytes!("monkey.obj");
        let mut monkey_mesh = Mesh::from_obj_bytes(&device, monkey_bytes).unwrap();
        monkey_mesh.add_instance(Instance::with_position(
            &monkey_mesh,
            cgmath::Vector3 { x: 0., y: 0., z: 0. }
        ));
        let mut second_instance = Instance::with_position(
            &monkey_mesh,
            cgmath::Vector3 { x: 2., y: 0., z: 0. }
        );
        second_instance.rotation = cgmath::Quaternion::from_angle_z(cgmath::Rad(3.14_f32));
        monkey_mesh.add_instance(second_instance);
        let meshes = vec![monkey_mesh];

        Self {
            meshes,
            lights
        }
    }
}