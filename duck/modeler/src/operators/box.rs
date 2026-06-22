use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{
    matrix4_to_row_major_f64, InnerSpace, Matrix3, Plane, Point3, Quaternion, Ray, Vector3,
};
use duck_engine_scene::{NodeId, Visibility};
use duck_engine_scene::cad::tessellate_into;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use glam::dvec3;
use opencascade::primitives::{Face, Shape, Wire};

use crate::document::Document;
use crate::tool::{ModelingTool, ToolInfo};
use super::ConstructionOptions;

/// A dimension at or below this is degenerate: the preview is hidden and the pick
/// can't be committed.
const EPSILON: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BoxAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    /// Center placed; the cursor drives the footprint. Preview is a flat rectangle face.
    Base { center: Point3, preview_node: NodeId },
    /// Footprint fixed; the cursor drives the height. Preview is the 3D box.
    Height { center: Point3, width: f32, depth: f32, preview_node: NodeId },
}

pub struct BoxOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    bindings: InputMap<BoxAction>,
    cursor_target: Option<Point3>,
}

impl BoxOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                BoxAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                BoxAction::Cancel,
            );
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            bindings,
            cursor_target: None,
        }
    }

    /// Rotation laying a local-XY face (normal +Z) flat on `plane`.
    fn face_rotation(plane: &Plane) -> Quaternion {
        let (u, _v) = plane.basis();
        Quaternion::from(Matrix3::from_cols(u, plane.normal.cross(u), plane.normal))
    }

    /// Rotation orienting the local reference box onto `plane`.
    fn box_rotation(plane: &Plane) -> Quaternion {
        let (u, v) = plane.basis();
        Quaternion::from(Matrix3::from_cols(u, plane.normal, v))
    }

    fn footprint_transform(center: Point3, width: f32, depth: f32, plane: &Plane) -> Transform {
        Transform {
            position: center,
            rotation: Self::face_rotation(plane),
            scale: Vector3::new(width, depth, 1.0),
        }
    }

    fn box_transform(
        center: Point3,
        width: f32,
        depth: f32,
        height: f32,
        plane: &Plane,
    ) -> Transform {
        // Keep every scale component non-negative. A negative scale would make the
        // baked GTransform a reflection, flipping the box's face normals inward.
        let (base, height) = if height >= 0.0 {
            (center, height)
        } else {
            (center + plane.normal * height, -height)
        };
        Transform {
            position: base,
            rotation: Self::box_rotation(plane),
            scale: Vector3::new(width, height, depth),
        }
    }

    /// In-plane extents from the center→corner vector, as full (width, depth).
    fn footprint_dims(center: Point3, corner: Point3, plane: &Plane) -> (f32, f32) {
        let (u, v) = plane.basis();
        let d = corner - center;
        let width = 2.0 * d.dot(u).abs();
        let depth = 2.0 * d.dot(v).abs();
        (width, depth)
    }

    /// A footprint is valid once both in-plane dimensions are non-degenerate.
    fn footprint_valid(width: f32, depth: f32) -> bool {
        width > EPSILON && depth > EPSILON
    }

    /// A box is valid once it has a non-degenerate footprint and a non-zero height
    fn box_valid(width: f32, depth: f32, height: f32) -> bool {
        Self::footprint_valid(width, depth) && height.abs() > EPSILON
    }

    /// Signed height from projecting the cursor pick ray onto the plane normal through `center`.
    fn height_from_cursor(center: Point3, plane: &Plane, position: (f32, f32), ctx: &mut EventContext) -> f32 {
        let camera = ctx.camera();
        let ray: Ray = camera.ray_from_screen_point(position.0, position.1, ctx.size.0, ctx.size.1);
        ray.closest_param_on_axis(center, plane.normal).unwrap_or(0.0)
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let camera = ctx.camera();
        let Some(center) = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[], &camera, ctx, &[])
            .map(|s| s.position)
        else {
            return false;
        };

        let Ok(preview_shape): Result<Shape, _> =
            Face::from_wire(&Wire::rect(1.0, 1.0).unwrap()).map(Into::into)
        else {
            return false;
        };

        let preview_node = {
            let coptions = self.construction_options.borrow();
            let mut scene = ctx.scene.lock().unwrap();
            // A single unit face, scaled each move; preview detail is irrelevant for a flat quad.
            let Ok(node) = tessellate_into(
                &preview_shape,
                &mut *scene,
                &coptions.geometry_options,
                None,
                Some("box plane preview"),
            ) else {
                return false;
            };
            // Hidden until the cursor defines a non-degenerate footprint.
            scene.set_node_visibility(node, Visibility::Invisible);
            node
        };
        self.phase = Phase::Base { center, preview_node };
        true
    }

    fn on_place_corner(
        &mut self,
        center: Point3,
        face_node: NodeId,
        position: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let camera = ctx.camera();
        let plane = self.construction_options.borrow().construction_plane;
        // Exclude the preview so the footprint can snap through it.
        let Some(corner) = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[face_node], &camera, ctx, &[])
            .map(|s| s.position)
        else {
            return false;
        };
        let (width, depth) = Self::footprint_dims(center, corner, &plane);
        // A degenerate footprint can't be committed; stay in the footprint stage.
        if !Self::footprint_valid(width, depth) {
            return false;
        }

        // Swap the flat footprint preview for the 3D box preview.
        let preview_shape = Shape::box_from_corners(dvec3(-0.5, 0.0, -0.5), dvec3(0.5, 1.0, 0.5));
        let preview_node = {
            let coptions = self.construction_options.borrow();
            let mut scene = ctx.scene.lock().unwrap();
            scene.cleanup_node(face_node);
            let Ok(node) = tessellate_into(
                &preview_shape,
                &mut *scene,
                &coptions.geometry_options,
                None,
                Some("box preview"),
            ) else {
                return false;
            };
            // Hidden until the cursor defines a non-zero height.
            scene.set_node_visibility(node, Visibility::Invisible);
            node
        };
        self.phase = Phase::Height { center, width, depth, preview_node };
        true
    }

    fn on_place_height(
        &mut self,
        center: Point3,
        width: f32,
        depth: f32,
        preview_node: NodeId,
        position: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let plane = self.construction_options.borrow().construction_plane;
        let height = Self::height_from_cursor(center, &plane, position, ctx);
        // A zero-height (degenerate) box can't be committed; stay in the height stage.
        if !Self::box_valid(width, depth, height) {
            return false;
        }

        // Bake the box transform into a world-space shape via GTransform. The reference
        // box matches the preview.
        let world_shape = {
            let mat = matrix4_to_row_major_f64(
                &Self::box_transform(center, width, depth, height, &plane).to_matrix(),
            );
            let reference = Shape::box_from_corners(dvec3(-0.5, 0.0, -0.5), dvec3(0.5, 1.0, 0.5));
            reference.gtransform(mat)
        };

        // Discard the preview, then commit the world-space shape as a registered part.
        ctx.scene.lock().unwrap().cleanup_node(preview_node);

        let committed = {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            doc.add_part("Box".to_owned(), world_shape, &coptions.geometry_options)
                .is_ok()
        };

        self.phase = Phase::Idle;
        committed
    }

    pub fn cancel(&mut self) {
        let preview = match self.phase {
            Phase::Base { preview_node, .. } | Phase::Height { preview_node, .. } => Some(preview_node),
            Phase::Idle => None,
        };
        if let Some(preview_node) = preview {
            let scene_arc = self.document.lock().unwrap().scene().clone();
            scene_arc.lock().unwrap().cleanup_node(preview_node);
            self.phase = Phase::Idle;
        }
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);
        let plane = self.construction_options.borrow().construction_plane;
        // While defining, exclude our own preview so snapping doesn't lock onto it.
        let exclude: Vec<NodeId> = match self.phase {
            Phase::Base { preview_node, .. } | Phase::Height { preview_node, .. } => vec![preview_node],
            Phase::Idle => Vec::new(),
        };

        let camera = ctx.camera();
        let snap = self
            .construction_options
            .borrow()
            .resolve_snap(cursor, &exclude, &camera, ctx, &[]);

        match self.phase {
            Phase::Idle => {
                self.cursor_target = snap.map(|s| s.position);
            }
            Phase::Base { center, preview_node } => {
                self.cursor_target = snap.map(|s| s.position);
                let dims = snap.map(|s| Self::footprint_dims(center, s.position, &plane));
                let mut scene = ctx.scene.lock().unwrap();
                match dims {
                    Some((width, depth)) if Self::footprint_valid(width, depth) => {
                        scene.set_node_visibility(preview_node, Visibility::Visible);
                        scene.set_node_transform(
                            preview_node,
                            Self::footprint_transform(center, width, depth, &plane),
                        );
                    }
                    // No snap, or a degenerate footprint: nothing to draw.
                    _ => scene.set_node_visibility(preview_node, Visibility::Invisible),
                }
            }
            Phase::Height { center, width, depth, preview_node } => {
                let height = Self::height_from_cursor(center, &plane, cursor, ctx);
                self.cursor_target = Some(center + plane.normal * height);
                let mut scene = ctx.scene.lock().unwrap();
                if Self::box_valid(width, depth, height) {
                    scene.set_node_visibility(preview_node, Visibility::Visible);
                    scene.set_node_transform(
                        preview_node,
                        Self::box_transform(center, width, depth, height, &plane),
                    );
                } else {
                    // Degenerate height: nothing to draw.
                    scene.set_node_visibility(preview_node, Visibility::Invisible);
                }
            }
        }
    }
}

impl ModelingTool for BoxOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: "box",
            icon_uri: "bytes://box.svg",
            icon: include_bytes!("../../../../assets/svg/cube-svgrepo-com.svg"),
        }
    }

    fn deactivate(&mut self) {
        self.cancel();
        // The modeler hides the cursor for the (now inactive) tool, but clear our
        // target so a stale point can't flash if we're reactivated before a move.
        self.cursor_target = None;
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor_target
    }
}

impl Operator for BoxOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        BoxAction::Place => match self.phase {
                            Phase::Idle => self.on_place_center(*position, ctx),
                            Phase::Base { center, preview_node } => {
                                self.on_place_corner(center, preview_node, *position, ctx)
                            }
                            Phase::Height { center, width, depth, preview_node } => {
                                self.on_place_height(center, width, depth, preview_node, *position, ctx)
                            }
                        },
                        BoxAction::Cancel => {
                            let was_defining = !matches!(self.phase, Phase::Idle);
                            if was_defining {
                                self.cancel();
                            }
                            was_defining
                        }
                    };
                }
                handled
            }
            DeviceEvent::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Box"
    }
}
