use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::{MetricSpace, Point3, Vector3};
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::tessellate_into;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    event::{DeviceEvent, Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use glam::{dvec3, DVec3};
use log::warn;
use opencascade::primitives::{Edge, Shape, Wire};

use crate::document::Document;
use crate::tool::{ModelingTool, ToolInfo};
use super::ConstructionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CircleAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    Defining { center: Point3, preview_node: NodeId },
}

pub struct CircleOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    bindings: InputMap<CircleAction>,
    cursor_target: Option<Point3>,
}

fn to_dvec3(p: Point3) -> DVec3 {
    dvec3(p.x as f64, p.y as f64, p.z as f64)
}

fn vec_to_dvec3(v: Vector3) -> DVec3 {
    dvec3(v.x as f64, v.y as f64, v.z as f64)
}

/// Builds a filled planar disk bounded by a circle of `radius` centered at
/// `center`, lying in the plane with the given `normal`. Falls back to the bare
/// ring (wire) if the face can't be built. Returns `None` on construction error.
fn circle_shape(center: Point3, normal: Vector3, radius: f64) -> Option<Shape> {
    let edge = Edge::circle(to_dvec3(center), vec_to_dvec3(normal), radius)
        .map_err(|e| warn!("Failed to build circle edge: {e}"))
        .ok()?;
    let wire = Wire::from_edges(&[edge])
        .map_err(|e| warn!("Failed to build circle wire: {e}"))
        .ok()?;
    // Filled disk (Region): fall back to the bare ring if the face can't be
    // built. `to_face` consumes the wire, so capture the ring shape first.
    let ring = Shape::from(&wire);
    match wire.to_face() {
        Ok(face) => Some(Shape::from(&face)),
        Err(_) => Some(ring),
    }
}

impl CircleOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                CircleAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                CircleAction::Cancel,
            );
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            bindings,
            cursor_target: None,
        }
    }

    /// Tessellates a preview disk into the scene and returns its node. Uses the
    /// coarser preview tolerance since the preview is rebuilt on every move.
    fn make_preview(&self, center: Point3, radius: f64, ctx: &mut EventContext) -> Option<NodeId> {
        let coptions = self.construction_options.borrow();
        let shape = circle_shape(center, coptions.construction_plane.normal, radius)?;
        let preview = coptions.preview_options();
        let mut scene = ctx.scene.lock().unwrap();
        tessellate_into(&shape, &mut *scene, &preview, None, Some("circle")).ok()
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
        let Some(preview_node) = self.make_preview(center, 0.01, ctx) else {
            return false;
        };
        self.phase = Phase::Defining { center, preview_node };
        true
    }

    fn on_place_outer(
        &mut self,
        center: Point3,
        preview_node: NodeId,
        position: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let camera = ctx.camera();
        // Exclude the preview so the radius can snap through a corner, not to the
        // preview's own geometry.
        let radius = self
            .construction_options
            .borrow()
            .resolve_snap(position, &[preview_node], &camera, ctx, &[])
            .map(|s| center.distance(s.position).max(0.01) as f64)
            .unwrap_or(0.01);

        let shape = {
            let coptions = self.construction_options.borrow();
            circle_shape(center, coptions.construction_plane.normal, radius)
        };

        // Discard the preview node, then commit the world-space shape as a part.
        ctx.scene.lock().unwrap().remove_node(preview_node);

        let committed = if let Some(shape) = shape {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            doc.add_part("Circle".to_owned(), shape, &coptions.geometry_options)
                .is_ok()
        } else {
            warn!("Failed to build circle shape.");
            false
        };

        self.phase = Phase::Idle;
        committed
    }

    pub fn cancel(&mut self) {
        if let Phase::Defining { preview_node, .. } = self.phase {
            let scene_arc = self.document.lock().unwrap().scene().clone();
            scene_arc.lock().unwrap().remove_node(preview_node);
            self.phase = Phase::Idle;
        }
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);
        // While defining, exclude our own preview so the radius doesn't snap to it.
        let exclude: Vec<NodeId> = match self.phase {
            Phase::Defining { preview_node, .. } => vec![preview_node],
            Phase::Idle => Vec::new(),
        };

        let camera = ctx.camera();
        let snap = self
            .construction_options
            .borrow()
            .resolve_snap(cursor, &exclude, &camera, ctx, &[]);

        // Record where the modeler should draw the 3D cursor.
        self.cursor_target = snap.map(|s| s.position);

        // Rebuild the preview disk from the snapped radius while defining. A flat
        // disk's orientation depends on the plane normal, so we re-tessellate
        // rather than scaling a unit mesh.
        if let Phase::Defining { center, preview_node } = self.phase {
            if let Some(snap) = snap {
                let radius = center.distance(snap.position).max(0.01) as f64;
                if let Some(new_node) = self.make_preview(center, radius, ctx) {
                    ctx.scene.lock().unwrap().remove_node(preview_node);
                    self.phase = Phase::Defining { center, preview_node: new_node };
                }
            }
        }
    }
}

impl ModelingTool for CircleOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: "circle",
            icon_uri: "bytes://circle.svg",
            icon: include_bytes!("../../../../assets/svg/circle-svgrepo-com.svg"),
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

impl Operator for CircleOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        CircleAction::Place => {
                            if let Phase::Defining { center, preview_node } = self.phase {
                                self.on_place_outer(center, preview_node, *position, ctx)
                            } else {
                                self.on_place_center(*position, ctx)
                            }
                        }
                        CircleAction::Cancel => {
                            let was_defining = matches!(self.phase, Phase::Defining { .. });
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
        "Circle"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_shape_from_positive_radius() {
        let center = Point3::new(0.0, 0.0, 0.0);
        let normal = Vector3::new(0.0, 1.0, 0.0);
        assert!(circle_shape(center, normal, 1.0).is_some());
    }

    #[test]
    fn circle_shape_off_origin() {
        let center = Point3::new(3.0, 0.0, -2.0);
        let normal = Vector3::new(0.0, 1.0, 0.0);
        assert!(circle_shape(center, normal, 2.5).is_some());
    }
}
