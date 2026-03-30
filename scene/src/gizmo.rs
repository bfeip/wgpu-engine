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

    /// Fire a ray at the center of each gizmo handle's AABB and verify
    /// that both the AABB and mesh triangle intersection succeed.
    #[test]
    fn translate_handles_pickable_by_ray() {
        use crate::common::Ray;
        use crate::geom_query::intersect_ray;
        use cgmath::Vector3;

        for handle in build_translate_handles(1.0) {
            let aabb = handle.mesh.bounding().expect("handle should have bounds");
            let center = aabb.center();

            // Try rays from all 3 principal directions toward the center
            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut any_mesh_hit = false;
            for dir in &directions {
                let origin = center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                let aabb_hit = aabb.intersects_ray(&ray);

                if aabb_hit.is_some() {
                    let mesh_hits = intersect_ray(&handle.mesh, &ray);
                    if !mesh_hits.is_empty() {
                        any_mesh_hit = true;
                    }
                }
            }

            assert!(
                any_mesh_hit,
                "Axis {:?}: ray through AABB center hit the AABB but missed all triangles. \
                 vertices={}, triangles={}, aabb_min={:?}, aabb_max={:?}",
                handle.axis,
                handle.mesh.vertices().len(),
                handle.mesh.triangle_indices().len() / 3,
                aabb.min,
                aabb.max,
            );
        }
    }

    #[test]
    fn scale_handles_pickable_by_ray() {
        use crate::common::Ray;
        use crate::geom_query::intersect_ray;
        use cgmath::Vector3;

        for handle in build_scale_handles(1.0) {
            let aabb = handle.mesh.bounding().expect("handle should have bounds");
            let center = aabb.center();

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut any_mesh_hit = false;
            for dir in &directions {
                let origin = center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                if aabb.intersects_ray(&ray).is_some() {
                    let mesh_hits = intersect_ray(&handle.mesh, &ray);
                    if !mesh_hits.is_empty() {
                        any_mesh_hit = true;
                    }
                }
            }

            assert!(
                any_mesh_hit,
                "Axis {:?}: ray through AABB center missed all triangles. \
                 vertices={}, triangles={}",
                handle.axis,
                handle.mesh.vertices().len(),
                handle.mesh.triangle_indices().len() / 3,
            );
        }
    }

    #[test]
    fn rotate_handles_pickable_by_ray() {
        use crate::common::Ray;
        use crate::geom_query::intersect_ray;
        use cgmath::Vector3;

        for handle in build_rotate_handles(1.0) {
            let aabb = handle.mesh.bounding().expect("handle should have bounds");
            let center = aabb.center();

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut any_mesh_hit = false;
            for dir in &directions {
                let origin = center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                if aabb.intersects_ray(&ray).is_some() {
                    let mesh_hits = intersect_ray(&handle.mesh, &ray);
                    if !mesh_hits.is_empty() {
                        any_mesh_hit = true;
                    }
                }
            }

            assert!(
                any_mesh_hit,
                "Axis {:?}: ray through AABB center missed all triangles. \
                 vertices={}, triangles={}",
                handle.axis,
                handle.mesh.vertices().len(),
                handle.mesh.triangle_indices().len() / 3,
            );
        }
    }

    /// Test picking gizmo handles through the full scene pipeline,
    /// simulating how GizmoState::show() sets up the handles.
    #[test]
    fn translate_handles_pickable_through_scene() {
        use crate::common::Ray;
        use crate::geom_query::pick_all_from_ray;
        use crate::Scene;
        use cgmath::{Point3, Vector3};

        let pivot = Point3::new(5.0, 3.0, 0.0);

        let mut scene = Scene::new();

        // Add a model cube so the scene isn't empty (like the real app)
        let cube = crate::Mesh::cube(2.0, crate::PrimitiveType::TriangleList);
        let cube_mesh_id = scene.add_mesh(cube);
        let cube_mat_id = scene.add_material(crate::Material::new());
        scene
            .add_instance_node(
                None,
                cube_mesh_id,
                cube_mat_id,
                Some("Cube".to_string()),
                crate::common::Transform::IDENTITY,
            )
            .unwrap();

        // Simulate sync_gizmo: compute scene.bounding() BEFORE showing gizmo
        // (this is what the real code does — it reads model_radius first)
        let _ = scene.bounding();

        let annotation_root = scene.ensure_annotation_root();
        let pivot_transform = crate::common::Transform::from_position(pivot);

        let handles = build_translate_handles(1.0);
        let mut node_ids = Vec::new();

        for handle in &handles {
            let mesh_id = scene.add_mesh(handle.mesh.clone());
            let material_id = scene.add_material(handle.material.clone());
            let node_id = scene
                .add_instance_node(
                    Some(annotation_root),
                    mesh_id,
                    material_id,
                    None,
                    pivot_transform,
                )
                .expect("Failed to add gizmo node");
            node_ids.push(node_id);
        }

        // For each handle, fire a ray at its world-space AABB center
        for (i, handle) in handles.iter().enumerate() {
            let local_aabb = handle.mesh.bounding().expect("handle should have bounds");
            let local_center = local_aabb.center();
            let world_center = Point3::new(
                local_center.x + pivot.x,
                local_center.y + pivot.y,
                local_center.z + pivot.z,
            );

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut found = false;
            for dir in &directions {
                let origin = world_center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                let results = pick_all_from_ray(&ray, &scene);
                for result in &results {
                    if result.node_id == node_ids[i] {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            assert!(
                found,
                "Axis {:?}: full scene pipeline failed to pick gizmo handle at pivot {:?}",
                handle.axis, pivot,
            );
        }
    }

    /// Test that picking works after gizmo position is updated
    /// (simulates update_position path from sync_gizmo).
    #[test]
    fn translate_handles_pickable_after_position_update() {
        use crate::common::Ray;
        use crate::geom_query::pick_all_from_ray;
        use crate::Scene;
        use cgmath::{Point3, Vector3};

        let initial_pivot = Point3::new(0.0, 0.0, 0.0);
        let new_pivot = Point3::new(10.0, 5.0, 3.0);

        let mut scene = Scene::new();
        let annotation_root = scene.ensure_annotation_root();
        let pivot_transform = crate::common::Transform::from_position(initial_pivot);

        let handles = build_translate_handles(1.0);
        let mut node_ids = Vec::new();

        for handle in &handles {
            let mesh_id = scene.add_mesh(handle.mesh.clone());
            let material_id = scene.add_material(handle.material.clone());
            let node_id = scene
                .add_instance_node(
                    Some(annotation_root),
                    mesh_id,
                    material_id,
                    None,
                    pivot_transform,
                )
                .expect("Failed to add gizmo node");
            node_ids.push(node_id);
        }

        // Cache bounds (simulates a frame between show and pick)
        let _ = scene.bounding();

        // Move gizmo to new position (simulates update_position)
        for &node_id in &node_ids {
            scene.get_node_mut(node_id).unwrap().set_position(new_pivot);
        }

        // Try to pick at the NEW position
        for (i, handle) in handles.iter().enumerate() {
            let local_aabb = handle.mesh.bounding().expect("handle should have bounds");
            let local_center = local_aabb.center();
            let world_center = Point3::new(
                local_center.x + new_pivot.x,
                local_center.y + new_pivot.y,
                local_center.z + new_pivot.z,
            );

            let directions = [
                Vector3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, -1.0, 0.0),
                Vector3::new(-1.0, 0.0, 0.0),
            ];

            let mut found = false;
            for dir in &directions {
                let origin = world_center - dir * 10.0;
                let ray = Ray::new(origin, *dir);

                let results = pick_all_from_ray(&ray, &scene);
                for result in &results {
                    if result.node_id == node_ids[i] {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            assert!(
                found,
                "Axis {:?}: picking failed after position update to {:?}",
                handle.axis, new_pivot,
            );
        }
    }
}
