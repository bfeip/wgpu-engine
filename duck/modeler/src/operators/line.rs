use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::Point3;
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::tessellate_into;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    event::{Event, EventContext},
    input::{ElementState, Key, Modifiers, MouseButton, NamedKey},
    operator::Operator,
    scene::PositionedCamera,
};
use glam::{dvec3, DVec3};
use log::warn;
use opencascade::primitives::{Edge, Shape, Wire};

use crate::document::Document;
use crate::snap::{Snap, SnapKind, SnapProvider, WireStartSnap};
use crate::tool::ModelingTool;
use super::ConstructionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LineAction {
    /// Place a point (or close the wire when hovering the start point).
    AddPoint,
    /// Finalize the current open polyline.
    Finish,
}

enum Phase {
    Idle,
    /// Building a polyline: `points` are the placed vertices, `preview_node` is the
    /// transient preview geometry (rebuilt each cursor move), and `closing` records
    /// whether the cursor is currently snapped onto the start point.
    Building {
        points: Vec<Point3>,
        preview_node: Option<NodeId>,
        closing: bool,
    },
}

pub struct LineOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    bindings: InputMap<LineAction>,
    /// Where the modeler's 3D cursor should sit (the latest snap point), or `None`
    /// to hide it. Read by the modeler via [`ModelingTool::cursor_target`].
    cursor_target: Option<Point3>,
    /// Set when a line has been committed so the modeler cedes back to selection.
    finished: bool,
}

fn to_dvec3(p: &Point3) -> DVec3 {
    dvec3(p.x as f64, p.y as f64, p.z as f64)
}

/// Builds an open polyline wire shape from `points` (one segment per consecutive
/// pair). Returns `None` if there are fewer than two points.
fn open_wire_shape(points: &[Point3]) -> Option<Shape> {
    if points.len() < 2 {
        return None;
    }
    let edges: Vec<Edge> = points
        .windows(2)
        .map(|w| Edge::segment(to_dvec3(&w[0]), to_dvec3(&w[1])))
        .collect::<Result<_, _>>()
        .map_err(|e| warn!("Failed to build polyline edge: {e}"))
        .ok()?;
    let wire = Wire::from_edges(&edges)
        .map_err(|e| warn!("Failed to build polyline wire: {e}"))
        .ok()?;
    Some(Shape::from(&wire))
}

/// Builds a closed wire from `points`. Returns `None` with fewer than three points.
/// `Wire::from_ordered_points` closes the loop automatically.
fn closed_wire(points: &[Point3]) -> Option<Wire> {
    if points.len() < 3 {
        return None;
    }
    Wire::from_ordered_points(points.iter().map(to_dvec3))
        .map_err(|e| warn!("Failed to build closed wire: {e}"))
        .ok()
}

/// Builds a closed wire shape (the loop, with no fill) from `points`. Used as a
/// fallback when the points are not co-planar and a face cannot be built.
fn closed_wire_shape(points: &[Point3]) -> Option<Shape> {
    Some(Shape::from(&closed_wire(points)?))
}

/// Builds a closed planar face shape from `points`. Returns `None` with fewer than
/// three points, or when the points are not co-planar (no face can be built).
fn closed_face_shape(points: &[Point3]) -> Option<Shape> {
    let wire = closed_wire(points)?;
    let face = wire.to_face().map_err(|e| warn!("Failed to build face from wire: {e}")).ok()?;
    Some(Shape::from(&face))
}

impl LineOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                LineAction::AddPoint,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                LineAction::Finish,
            );
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            bindings,
            cursor_target: None,
            finished: false,
        }
    }

    /// Resolves a snapped point. While building a face (≥3 points), offers the
    /// wire's start point as a [`SnapKind::WireStart`] candidate so the snap
    /// engine can close the path; callers read `snap.kind == WireStart` to learn
    /// whether the cursor is closing the wire.
    fn snapped_point(
        &self,
        cursor: (f32, f32),
        exclude: &[NodeId],
        camera: &PositionedCamera,
        ctx: &EventContext,
    ) -> Option<Snap> {
        // A face needs at least three vertices to enclose an area.
        let wire_start = match &self.phase {
            Phase::Building { points, .. } if points.len() >= 3 => {
                Some(WireStartSnap { start: points[0] })
            }
            _ => None,
        };
        let extra: Vec<&dyn SnapProvider> =
            wire_start.iter().map(|p| p as &dyn SnapProvider).collect();

        self.construction_options
            .borrow()
            .resolve_snap(cursor, exclude, camera, ctx, &extra)
    }

    /// Removes the transient preview node, if any, using the supplied scene lock.
    fn remove_preview(&mut self, ctx: &mut EventContext) {
        if let Phase::Building { preview_node, .. } = &mut self.phase {
            if let Some(node) = preview_node.take() {
                ctx.scene.lock().unwrap().remove_node(node);
            }
        }
    }

    /// Rebuilds the preview geometry. `cursor_point` is the live (snapped) cursor
    /// position to draw a rubber-band segment to, or `None` to show only the placed
    /// points. When `closing` is set, shows a filled face preview instead.
    fn rebuild_preview(&mut self, cursor_point: Option<Point3>, closing: bool, ctx: &mut EventContext) {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return,
        };
        self.remove_preview(ctx);

        let shape = if closing {
            // Non-coplanar points can't form a face; fall back to the closed wire.
            closed_face_shape(&points).or_else(|| closed_wire_shape(&points))
        } else {
            let mut all = points;
            if let Some(c) = cursor_point {
                all.push(c);
            }
            open_wire_shape(&all)
        };

        let new_node = shape.and_then(|s| {
            let coptions = self.construction_options.borrow();
            let mut scene = ctx.scene.lock().unwrap();
            tessellate_into(&s, &mut *scene, &coptions.geometry_preview_options, None, Some("line")).ok()
        });

        if let Phase::Building { preview_node, closing: c, .. } = &mut self.phase {
            *preview_node = new_node;
            *c = closing;
        }
    }

    /// Adds a point to the wire or starts building a wire if there were no previous points.
    /// Returns true if a point was successfully added.
    fn on_add_point(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let exclude: Vec<NodeId> = match &self.phase {
            Phase::Building { preview_node: Some(n), .. } => vec![*n],
            _ => Vec::new(),
        };
        let camera = ctx.camera();
        let Some(snap) = self.snapped_point(position, &exclude, &camera, ctx) else {
            return false;
        };
        let point = snap.position;
        let closing = snap.kind == SnapKind::WireStart;

        match &mut self.phase {
            Phase::Idle => {
                self.phase = Phase::Building { points: vec![point], preview_node: None, closing: false };
            }
            Phase::Building { .. } => {
                if closing {
                    self.commit_closed(ctx);
                } else {
                    if let Phase::Building { points, .. } = &mut self.phase {
                        points.push(point);
                    }
                    // Show the placed polyline; the next cursor move adds the rubber band.
                    self.rebuild_preview(None, false, ctx);
                }
            }
        }

        true
    }

    /// Commits the placed points as a closed planar region and ends the tool.
    fn commit_closed(&mut self, ctx: &mut EventContext) {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return,
        };
        self.remove_preview(ctx);

        if let Some(shape) = closed_face_shape(&points).or_else(|| closed_wire_shape(&points)) {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            if doc
                .add_part(
                    "Region".to_owned(),
                    shape,
                    &coptions.geometry_preview_options,
                )
                .is_ok()
            {
                self.finished = true;
            }
        }
        else {
            warn!("Failed to build closed face shape for polyline.")
        }
        self.phase = Phase::Idle;
    }

    /// Commits the placed points as an open polyline wire and ends the tool. Does
    /// nothing if fewer than one segment has been defined.
    fn finish(&mut self, ctx: &mut EventContext) -> bool {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return false,
        };
        self.remove_preview(ctx);

        let mut committed = false;
        if let Some(shape) = open_wire_shape(&points) {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            committed = doc
                .add_part(
                    "Line".to_owned(),
                    shape,
                    &coptions.geometry_preview_options,
                )
                .is_ok();
        }
        self.phase = Phase::Idle;
        if committed {
            self.finished = true;
        }
        committed
    }

    /// Discards the in-progress line, removing its preview. Self-sources the scene
    /// so it can run without an [`EventContext`] (e.g. from `deactivate`).
    pub fn cancel(&mut self) {
        if let Phase::Building { preview_node: Some(node), .. } = &self.phase {
            let scene_arc = self.document.lock().unwrap().scene().clone();
            scene_arc.lock().unwrap().remove_node(*node);
        }
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);
        let exclude: Vec<NodeId> = match &self.phase {
            Phase::Building { preview_node: Some(n), .. } => vec![*n],
            _ => Vec::new(),
        };
        let camera = ctx.camera();
        let snapped = self.snapped_point(cursor, &exclude, &camera, ctx);

        // Record where the 3D cursor should sit: a real snap, not the free
        // construction-plane fallback (which sits under the cursor).
        self.cursor_target = snapped
            .filter(|s| s.kind != SnapKind::ConstructionPlane)
            .map(|s| s.position);

        if matches!(self.phase, Phase::Building { .. }) {
            if let Some(snap) = snapped {
                let closing = snap.kind == SnapKind::WireStart;
                self.rebuild_preview(Some(snap.position), closing, ctx);
            }
        }
    }
}

impl ModelingTool for LineOperator {
    fn deactivate(&mut self) {
        self.cancel();
        self.cursor_target = None;
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor_target
    }
}

impl Operator for LineOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        match event {
            Event::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        LineAction::AddPoint => self.on_add_point(*position, ctx),
                        LineAction::Finish => self.finish(ctx),
                    };
                }
                handled
            }
            Event::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            Event::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => self.finish(ctx),
                    Key::Named(NamedKey::Escape) => {
                        let was_building = matches!(self.phase, Phase::Building { .. });
                        if was_building {
                            self.cancel();
                        }
                        was_building
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Line"
    }
}
