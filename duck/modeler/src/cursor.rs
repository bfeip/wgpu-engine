//! A 3D cursor: a small marker placed at a world point and kept at a
//! constant on-screen size. Owned by the modeler and driven each frame from the
//! active tool's reported target.
//!
//! The marker is a 3D line "asterisk" (orientation-independent), rendered through
//! the same `LineList` path as the grid and CAD edges. wgpu point primitives are
//! 1px and too small to be useful; when screen-facing splats land, only
//! [`build_marker_mesh`] needs to change.

use duck_engine_viewer::common::{InnerSpace, Point3, Quaternion, RgbaColor, Transform, Vector3};
use duck_engine_viewer::scene::{
    Material, Mesh, MeshPrimitive, NodeFlags, NodeId, PositionedCamera, PrimitiveType, Scene,
    Vertex, Visibility,
};

/// Half-extent of the marker, in screen pixels (the asterisk arms are this long).
const MARKER_HALF_PIXELS: f32 = 8.0;

/// Marker color (amber), chosen to read against both the grid and parts.
const MARKER_COLOR: RgbaColor = RgbaColor { r: 1.0, g: 0.78, b: 0.12, a: 1.0 };

/// A marker placed at a world point and kept at a constant pixel size. The node
/// is created lazily on first show; redundant updates are skipped so it is cheap
/// to call every frame.
#[derive(Default)]
pub struct Cursor3d {
    node: Option<NodeId>,
    /// Last (position, uniform scale) actually written, to skip redundant writes.
    shown: Option<(Point3, f32)>,
}

impl Cursor3d {
    /// Places the cursor at `target` (rescaled to a constant pixel size), or
    /// hides it when `target` is `None`.
    pub fn update(
        &mut self,
        target: Option<Point3>,
        camera: &PositionedCamera,
        viewport: (u32, u32),
        scene: &mut Scene,
    ) {
        let Some(position) = target else {
            if self.shown.take().is_some() {
                if let Some(node) = self.node {
                    scene.set_node_visibility(node, Visibility::Invisible);
                }
            }
            return;
        };

        // Constant on-screen size: scale a unit marker by the world size of one
        // pixel at the marker's depth.
        let depth = (position - camera.eye).dot(camera.forward()).max(camera.znear);
        let scale = (camera.world_size_per_pixel(depth, viewport.1) * MARKER_HALF_PIXELS)
            .max(f32::EPSILON);

        if self.shown == Some((position, scale)) {
            return;
        }

        let node = self.ensure_node(scene);
        scene.set_node_transform(
            node,
            Transform {
                position,
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(scale, scale, scale),
            },
        );
        scene.set_node_visibility(node, Visibility::Visible);
        self.shown = Some((position, scale));
    }

    /// Returns the marker node, creating its mesh/material/node on first use.
    fn ensure_node(&mut self, scene: &mut Scene) -> NodeId {
        if let Some(node) = self.node {
            return node;
        }
        let mesh = scene.add_mesh(build_marker_mesh());
        let material = scene.add_material(Material::new().with_line_color(MARKER_COLOR));
        let node = scene
            .add_instance_node(
                None,
                mesh,
                material,
                Some("3D cursor".to_owned()),
                Transform::IDENTITY,
                // Inert: not selectable (keeps it out of geometry snapping), not
                // exported, and excluded from scene bounds.
                NodeFlags::inert(),
            )
            .expect("Failed to create 3D cursor node");
        self.node = Some(node);
        node
    }
}

/// A unit 3D asterisk: three `LineList` segments along ±X, ±Y, ±Z. Scaled
/// uniformly by the caller to a constant pixel size.
fn build_marker_mesh() -> Mesh {
    let vertex = |x: f32, y: f32, z: f32| Vertex {
        position: [x, y, z],
        tex_coords: [0.0; 3],
        normal: [0.0; 3],
    };
    let vertices = vec![
        vertex(-1.0, 0.0, 0.0),
        vertex(1.0, 0.0, 0.0),
        vertex(0.0, -1.0, 0.0),
        vertex(0.0, 1.0, 0.0),
        vertex(0.0, 0.0, -1.0),
        vertex(0.0, 0.0, 1.0),
    ];
    Mesh::from_raw(
        vertices,
        vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1, 2, 3, 4, 5],
        }],
    )
}
