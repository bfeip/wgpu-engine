//! Gizmo geometry builders for transform handles (translate, rotate, scale).
//!
//! Each builder produces a set of [`GizmoHandle`]s — one per axis — containing
//! a mesh and material ready to be added to the scene. All geometry is built at
//! the origin; positioning at the selection pivot is done via node transforms.

use cgmath::{Deg, Matrix4, Vector3};

use crate::common::{Axis, RgbaColor};
use crate::material::{AlphaMode, Material, MaterialFlags};
use crate::mesh::{Mesh, PrimitiveType};

/// Which type of gizmo to display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoType {
    Translate,
    Rotate,
    Scale,
}

/// A single gizmo handle (one axis of a gizmo).
pub struct GizmoHandle {
    pub mesh: Mesh,
    pub material: Material,
    pub axis: Axis,
}

/// Brighter color for hover/active highlighting.
pub fn highlight_color(axis: Axis) -> RgbaColor {
    match axis {
        Axis::X => RgbaColor { r: 1.0, g: 0.5, b: 0.3, a: 1.0 },
        Axis::Y => RgbaColor { r: 0.5, g: 1.0, b: 0.3, a: 1.0 },
        Axis::Z => RgbaColor { r: 0.3, g: 0.5, b: 1.0, a: 1.0 },
    }
}

const GIZMO_FLAGS: MaterialFlags = MaterialFlags::DO_NOT_LIGHT
    .union(MaterialFlags::DOUBLE_SIDED)
    .union(MaterialFlags::ALWAYS_ON_TOP);

const SEGMENTS: u32 = 16;

/// Create a gizmo material for the given axis color.
fn gizmo_material(color: RgbaColor) -> Material {
    Material::new()
        .with_base_color_factor(color)
        .with_flags(GIZMO_FLAGS)
        .with_alpha_mode(AlphaMode::Opaque)
}

// ─── Translation Gizmo ────────────────────────────────────────────────

/// Build translation gizmo handles: cylinder shaft + cone arrowhead per axis.
pub fn build_translate_handles(size: f32) -> Vec<GizmoHandle> {
    let shaft_radius = size * 0.025;
    let shaft_height = size * 0.8;
    let cone_radius = size * 0.08;
    let cone_height = size * 0.2;

    Axis::ALL
        .into_iter()
        .map(|axis| {
            let rotation = axis.rotation_from_y();

            let shaft = Mesh::cylinder(
                shaft_radius,
                shaft_height,
                SEGMENTS,
                false,
                PrimitiveType::TriangleList,
            )
            .transformed(
                &(rotation
                    * Matrix4::from_translation(Vector3::new(0.0, shaft_height / 2.0, 0.0))),
            );

            let cone_base_y = shaft_height + cone_height / 2.0;
            let cone = Mesh::cone(
                cone_radius,
                cone_height,
                SEGMENTS,
                true,
                PrimitiveType::TriangleList,
            )
            .transformed(
                &(rotation * Matrix4::from_translation(Vector3::new(0.0, cone_base_y, 0.0))),
            );

            let mesh = shaft.merged(&cone);
            let material = gizmo_material(axis.color());

            GizmoHandle {
                mesh,
                material,
                axis,
            }
        })
        .collect()
}

// ─── Scale Gizmo ──────────────────────────────────────────────────────

/// Build scale gizmo handles: cylinder shaft + cube tip per axis.
pub fn build_scale_handles(size: f32) -> Vec<GizmoHandle> {
    let shaft_radius = size * 0.025;
    let shaft_height = size * 0.8;
    let cube_size = size * 0.1;

    Axis::ALL
        .into_iter()
        .map(|axis| {
            let rotation = axis.rotation_from_y();

            let shaft = Mesh::cylinder(
                shaft_radius,
                shaft_height,
                SEGMENTS,
                false,
                PrimitiveType::TriangleList,
            )
            .transformed(
                &(rotation
                    * Matrix4::from_translation(Vector3::new(0.0, shaft_height / 2.0, 0.0))),
            );

            let cube_y = shaft_height + cube_size / 2.0;
            let cube = Mesh::cube(cube_size, PrimitiveType::TriangleList)
                .transformed(
                    &(rotation * Matrix4::from_translation(Vector3::new(0.0, cube_y, 0.0))),
                );

            let mesh = shaft.merged(&cube);
            let material = gizmo_material(axis.color());

            GizmoHandle {
                mesh,
                material,
                axis,
            }
        })
        .collect()
}

// ─── Rotation Gizmo ──────────────────────────────────────────────────

/// Build rotation gizmo handles: one torus ring per axis.
pub fn build_rotate_handles(size: f32) -> Vec<GizmoHandle> {
    let major_radius = size;
    let minor_radius = size * 0.02;

    Axis::ALL
        .into_iter()
        .map(|axis| {
            // Torus is built in the XZ plane (Y-axis rotation ring).
            // Rotate to orient for the other axes.
            let rotation = match axis {
                Axis::Y => Matrix4::from_scale(1.0),
                // XZ plane torus rotated 90 around Z → lies in YZ plane (X-axis ring)
                Axis::X => Matrix4::from_angle_z(Deg(90.0)),
                // XZ plane torus rotated 90 around X → lies in XY plane (Z-axis ring)
                Axis::Z => Matrix4::from_angle_x(Deg(90.0)),
            };

            let mesh = Mesh::torus(
                major_radius,
                minor_radius,
                32,
                SEGMENTS,
                PrimitiveType::TriangleList,
            )
            .transformed(&rotation);
            let material = gizmo_material(axis.color());

            GizmoHandle {
                mesh,
                material,
                axis,
            }
        })
        .collect()
}

/// Build gizmo handles for a given type.
pub fn build_handles(gizmo_type: GizmoType, size: f32) -> Vec<GizmoHandle> {
    match gizmo_type {
        GizmoType::Translate => build_translate_handles(size),
        GizmoType::Rotate => build_rotate_handles(size),
        GizmoType::Scale => build_scale_handles(size),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_handles_have_three_axes() {
        let handles = build_translate_handles(1.0);
        assert_eq!(handles.len(), 3);
        assert_eq!(handles[0].axis, Axis::X);
        assert_eq!(handles[1].axis, Axis::Y);
        assert_eq!(handles[2].axis, Axis::Z);
    }

    #[test]
    fn translate_handles_have_triangle_geometry() {
        for handle in build_translate_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn scale_handles_have_three_axes() {
        let handles = build_scale_handles(1.0);
        assert_eq!(handles.len(), 3);
    }

    #[test]
    fn scale_handles_have_triangle_geometry() {
        for handle in build_scale_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn rotate_handles_have_three_axes() {
        let handles = build_rotate_handles(1.0);
        assert_eq!(handles.len(), 3);
    }

    #[test]
    fn rotate_handles_have_triangle_geometry() {
        for handle in build_rotate_handles(1.0) {
            assert!(!handle.mesh.vertices().is_empty());
            assert!(handle.mesh.has_primitive_type(PrimitiveType::TriangleList));
        }
    }

    #[test]
    fn build_handles_dispatches_correctly() {
        assert_eq!(build_handles(GizmoType::Translate, 1.0).len(), 3);
        assert_eq!(build_handles(GizmoType::Rotate, 1.0).len(), 3);
        assert_eq!(build_handles(GizmoType::Scale, 1.0).len(), 3);
    }

    #[test]
    fn gizmo_materials_have_correct_flags() {
        for handle in build_translate_handles(1.0) {
            let flags = handle.material.flags();
            assert!(flags.contains(MaterialFlags::DO_NOT_LIGHT));
            assert!(flags.contains(MaterialFlags::DOUBLE_SIDED));
            assert!(flags.contains(MaterialFlags::ALWAYS_ON_TOP));
        }
    }
}
