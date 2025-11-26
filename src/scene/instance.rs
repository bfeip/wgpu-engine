use std::mem::size_of;

use crate::material::MaterialId;
use super::mesh::MeshId;

pub type InstanceId = u32;

/// An instance references a mesh and material to be rendered.
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


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub transform: [[f32; 4]; 4],
    pub normal_mat: [[f32; 3]; 3]
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use crate::drawstate::VertexShaderLocations as VSL;

        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: VSL::InstanceTransformRow0 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*1]>() as wgpu::BufferAddress,
                    shader_location: VSL::InstanceTransformRow1 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*2]>() as wgpu::BufferAddress,
                    shader_location: VSL::InstanceTransformRow2 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*3]>() as wgpu::BufferAddress,
                    shader_location: VSL::InstanceTransformRow3 as u32,
                    format: wgpu::VertexFormat::Float32x4
                },

                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4*4]>() as wgpu::BufferAddress,
                    shader_location: VSL::InstanceNormalRow0 as u32,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; (4*4) + (3*1)]>() as wgpu::BufferAddress,
                    shader_location: VSL::InstanceNormalRow1 as u32,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; (4*4) + (3*2)]>() as wgpu::BufferAddress,
                    shader_location: VSL::InstanceNormalRow2 as u32,
                    format: wgpu::VertexFormat::Float32x3
                },
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Matrix4, Matrix3, Vector3, SquareMatrix};
    use crate::common;

    // ========================================================================
    // Instance Tests
    // ========================================================================

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

        // Each instance should have a unique ID
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

    // ========================================================================
    // InstanceRaw Tests
    // ========================================================================

    #[test]
    fn test_instance_raw_zeroable() {
        // Test that we can create a zeroed InstanceRaw
        let raw: InstanceRaw = bytemuck::Zeroable::zeroed();

        // All fields should be zero
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(raw.transform[i][j], 0.0);
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(raw.normal_mat[i][j], 0.0);
            }
        }
    }

    #[test]
    fn test_instance_raw_size() {
        // InstanceRaw should be exactly 100 bytes:
        // - 4x4 f32 matrix = 64 bytes
        // - 3x3 f32 matrix = 36 bytes
        // Total = 100 bytes
        assert_eq!(size_of::<InstanceRaw>(), 100);
    }

    #[test]
    fn test_instance_raw_transform_layout() {
        // Transform should be 64 bytes (4x4 matrix of f32)
        assert_eq!(size_of::<[[f32; 4]; 4]>(), 64);

        let raw = InstanceRaw {
            transform: [[1.0; 4]; 4],
            normal_mat: [[0.0; 3]; 3],
        };

        // Verify we can access transform elements
        assert_eq!(raw.transform[0][0], 1.0);
        assert_eq!(raw.transform[3][3], 1.0);
    }

    #[test]
    fn test_instance_raw_normal_matrix_layout() {
        // Normal matrix should be 36 bytes (3x3 matrix of f32)
        assert_eq!(size_of::<[[f32; 3]; 3]>(), 36);

        let raw = InstanceRaw {
            transform: [[0.0; 4]; 4],
            normal_mat: [[2.0; 3]; 3],
        };

        // Verify we can access normal matrix elements
        assert_eq!(raw.normal_mat[0][0], 2.0);
        assert_eq!(raw.normal_mat[2][2], 2.0);
    }

    #[test]
    fn test_instance_raw_from_transform() {
        let transform = Matrix4::from_scale(2.0);
        let normal_matrix = common::compute_normal_matrix(&transform);

        // Convert to InstanceRaw
        let raw = InstanceRaw {
            transform: transform.into(),
            normal_mat: normal_matrix.into(),
        };

        // Verify transform conversion
        assert_eq!(raw.transform[0][0], 2.0);
        assert_eq!(raw.transform[1][1], 2.0);
        assert_eq!(raw.transform[2][2], 2.0);
        assert_eq!(raw.transform[3][3], 1.0);

        // Verify normal matrix conversion
        assert_eq!(raw.normal_mat[0][0], 0.5); // inverse scale
        assert_eq!(raw.normal_mat[1][1], 0.5);
        assert_eq!(raw.normal_mat[2][2], 0.5);
    }

    #[test]
    fn test_instance_raw_identity_transform() {
        let identity = Matrix4::identity();
        let normal_matrix = common::compute_normal_matrix(&identity);

        let raw = InstanceRaw {
            transform: identity.into(),
            normal_mat: normal_matrix.into(),
        };

        // Identity transform
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(raw.transform[i][j], 1.0);
                } else {
                    assert_eq!(raw.transform[i][j], 0.0);
                }
            }
        }

        // Identity normal matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(raw.normal_mat[i][j], 1.0);
                } else {
                    assert_eq!(raw.normal_mat[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_instance_raw_translation() {
        let transform = Matrix4::from_translation(Vector3::new(5.0, 10.0, 15.0));
        let raw = InstanceRaw {
            transform: transform.into(),
            normal_mat: Matrix3::identity().into(),
        };

        // Check translation components (last column, but cgmath is column-major)
        assert_eq!(raw.transform[3][0], 5.0);
        assert_eq!(raw.transform[3][1], 10.0);
        assert_eq!(raw.transform[3][2], 15.0);
    }

    #[test]
    fn test_instance_raw_repr_c() {
        // Verify that InstanceRaw has the correct #[repr(C)] layout
        // by checking field offsets
        use std::mem::offset_of;

        assert_eq!(offset_of!(InstanceRaw, transform), 0);
        assert_eq!(offset_of!(InstanceRaw, normal_mat), 64);
    }

    #[test]
    fn test_instance_raw_vertex_buffer_layout() {
        let layout = InstanceRaw::desc();

        // Verify basic layout properties
        assert_eq!(layout.array_stride, 100);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Instance);
        assert_eq!(layout.attributes.len(), 7); // 4 for transform + 3 for normal

        // Verify transform attributes
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[1].offset, 16);
        assert_eq!(layout.attributes[2].offset, 32);
        assert_eq!(layout.attributes[3].offset, 48);

        // Verify normal matrix attributes
        assert_eq!(layout.attributes[4].offset, 64);
        assert_eq!(layout.attributes[5].offset, 76);
        assert_eq!(layout.attributes[6].offset, 88);

        // Verify formats
        for i in 0..4 {
            assert_eq!(layout.attributes[i].format, wgpu::VertexFormat::Float32x4);
        }
        for i in 4..7 {
            assert_eq!(layout.attributes[i].format, wgpu::VertexFormat::Float32x3);
        }
    }
}