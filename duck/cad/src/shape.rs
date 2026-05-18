use std::path::Path;

use anyhow::{Context, Result};
use duck_engine_scene::{
    common::Transform, Instance, Material, NodeFlags, NodeId, NodePayload, Scene,
};
use glam::DVec3;
use opencascade::angle::Angle;
use opencascade::primitives::{Face, Shape};

use crate::tessellate::tessellate_occ_shape;
use crate::{CadImportOptions, CadImportResult, CadWire};

/// An opaque handle to a B-Rep solid. Use the primitive constructors or operations to create
/// shapes, then call [`CadShape::tessellate_into`] to render them as scene geometry.
pub struct CadShape {
    pub(crate) inner: Shape,
}

impl CadShape {
    // --- Primitives ---

    /// A sphere of the given radius, centred at the origin.
    pub fn sphere(radius: f64) -> Self {
        Self { inner: Shape::sphere(radius).build() }
    }

    /// A box centred at the origin with the given dimensions.
    pub fn cuboid(width: f64, depth: f64, height: f64) -> Self {
        Self { inner: Shape::box_centered(width, depth, height) }
    }

    /// A cylinder with base at the origin, extending along +Z.
    pub fn cylinder(radius: f64, height: f64) -> Self {
        Self { inner: Shape::cylinder_radius_height(radius, height) }
    }

    /// A cone with base at the origin, extending along +Z.
    ///
    /// Set `top_radius` to `0.0` for a sharp apex.
    pub fn cone(bottom_radius: f64, top_radius: f64, height: f64) -> Self {
        Self {
            inner: Shape::cone()
                .bottom_radius(bottom_radius)
                .top_radius(top_radius)
                .height(height)
                .build(),
        }
    }

    /// A torus centred at the origin in the XY plane.
    ///
    /// `major_radius` is the distance from the torus centre to the tube centre;
    /// `minor_radius` is the tube radius.
    pub fn torus(major_radius: f64, minor_radius: f64) -> Self {
        Self {
            inner: Shape::torus().radius_1(major_radius).radius_2(minor_radius).build(),
        }
    }

    // --- Wire-based construction ---

    /// Extrude a closed wire profile into a solid along `direction`.
    ///
    /// The wire is converted to a planar face before extrusion.
    pub fn extrude(wire: &CadWire, direction: [f64; 3]) -> Self {
        let face = Face::from_wire(&wire.inner);
        let dir = DVec3::from(direction);
        Self { inner: face.extrude(dir).into() }
    }

    /// Revolve a closed wire profile around an axis by `angle_deg` degrees.
    ///
    /// Pass `360.0` for a full revolution. The wire is converted to a planar face
    /// before revolution.
    pub fn revolve(
        wire: &CadWire,
        axis_origin: [f64; 3],
        axis: [f64; 3],
        angle_deg: f64,
    ) -> Self {
        let face = Face::from_wire(&wire.inner);
        let origin = DVec3::from(axis_origin);
        let dir = DVec3::from(axis);
        let angle = if (angle_deg - 360.0).abs() < 1e-9 {
            None
        } else {
            Some(Angle::Degrees(angle_deg))
        };
        Self { inner: face.revolve(origin, dir, angle).into() }
    }

    // --- Boolean operations ---

    /// Union of two shapes.
    pub fn union(&self, other: &Self) -> Self {
        Self { inner: self.inner.union(&other.inner).shape }
    }

    /// Subtract `other` from `self`.
    pub fn subtract(&self, other: &Self) -> Self {
        Self { inner: self.inner.subtract(&other.inner).shape }
    }

    /// Intersection of two shapes.
    pub fn intersect(&self, other: &Self) -> Self {
        Self { inner: self.inner.intersect(&other.inner).shape }
    }

    // --- Modifications ---

    /// Fillet all edges with the given radius.
    pub fn fillet(&self, radius: f64) -> Self {
        Self { inner: self.inner.fillet(radius) }
    }

    /// Chamfer all edges with the given distance.
    pub fn chamfer(&self, distance: f64) -> Self {
        Self { inner: self.inner.chamfer(distance) }
    }

    // --- Transform ---

    /// Translate the shape in-place by `offset`.
    pub fn translate(&mut self, offset: [f64; 3]) {
        self.inner.set_global_translation(DVec3::from(offset));
    }

    // --- Export ---

    /// Write the shape to a STEP file.
    pub fn write_step(&self, path: impl AsRef<Path>) -> Result<()> {
        self.inner.write_step_to_file(path).context("Failed to write STEP file")
    }

    /// Write the shape to a BRep text file.
    pub fn write_brep(&self, path: impl AsRef<Path>) -> Result<()> {
        self.inner.write_brep_text(path).context("Failed to write BRep file")
    }

    // --- Tessellation ---

    /// Tessellate the shape into scene geometry, attaching it as a child of `parent`.
    ///
    /// Returns a [`CadImportResult`] with the created root node. The node's mesh uses
    /// `options.face_color` and `options.edge_color`.
    pub fn tessellate_into(
        &self,
        scene: &mut Scene,
        options: &CadImportOptions,
        parent: Option<NodeId>,
        name: Option<&str>,
    ) -> Result<CadImportResult> {
        let mesh =
            tessellate_occ_shape(&self.inner, options.tessellation_tolerance, options.scale_factor, options.include_edges)?;
        let mat = scene.add_material(
            Material::new()
                .with_base_color_factor(options.face_color)
                .with_line_color(options.edge_color),
        );
        let mesh_id = scene.add_mesh(mesh);
        let instance_id = scene.add_instance(Instance::new(mesh_id, mat));

        let node_name = name.map(|s| s.to_string());
        let root = scene
            .add_node(parent, node_name, Transform::IDENTITY, NodeFlags::NONE)
            .context("Failed to add shape node")?;
        scene.set_node_payload(root, NodePayload::Instance(instance_id));

        Ok(CadImportResult { root, pmi_root: None, views: vec![] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_options() -> CadImportOptions {
        CadImportOptions::default()
    }

    #[test]
    fn sphere_tessellates_to_nonempty_mesh() {
        let shape = CadShape::sphere(1.0);
        let mut scene = duck_engine_scene::Scene::new();
        let result = shape
            .tessellate_into(&mut scene, &default_options(), None, Some("sphere"))
            .expect("tessellation failed");
        assert!(scene.mesh_count() > 0);
        assert!(scene.node_count() > 0);
        let _ = result.root;
    }

    #[test]
    fn cuboid_tessellates_to_nonempty_mesh() {
        let shape = CadShape::cuboid(2.0, 2.0, 2.0);
        let mut scene = duck_engine_scene::Scene::new();
        shape
            .tessellate_into(&mut scene, &default_options(), None, Some("box"))
            .expect("tessellation failed");
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn union_of_two_cuboids_tessellates() {
        let a = CadShape::cuboid(2.0, 2.0, 2.0);
        let b = CadShape::sphere(1.5);
        let combined = a.union(&b);
        let mut scene = duck_engine_scene::Scene::new();
        combined
            .tessellate_into(&mut scene, &default_options(), None, None)
            .expect("union tessellation failed");
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn cylinder_tessellates() {
        let shape = CadShape::cylinder(0.5, 2.0);
        let mut scene = duck_engine_scene::Scene::new();
        shape.tessellate_into(&mut scene, &default_options(), None, None).unwrap();
        assert!(scene.mesh_count() > 0);
    }

    #[test]
    fn torus_tessellates() {
        let shape = CadShape::torus(2.0, 0.5);
        let mut scene = duck_engine_scene::Scene::new();
        shape.tessellate_into(&mut scene, &default_options(), None, None).unwrap();
        assert!(scene.mesh_count() > 0);
    }
}
