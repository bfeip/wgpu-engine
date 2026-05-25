use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::event::{Event, EventContext};
use crate::geom_query::{RayHit, RayPickQuery, RayPickResult, pick_all_from_ray};
use crate::input::{Modifiers, MouseButton};
use crate::operator::Operator;
use crate::selection::SelectionItem;
use duck_engine_common::InnerSpace;

/// Semantic actions for the selection operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelectionAction {
    /// Cast a ray and update the selection.
    Select,
}

/// Operator for selecting objects in the scene via mouse click.
pub struct SelectionOperator {
    pub bindings: InputMap<SelectionAction>,
}

impl SelectionOperator {
    /// Creates a new selection operator.
    pub fn new() -> Self {
        let bindings = InputMap::new().bind(
            InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
            SelectionAction::Select,
        );
        Self { bindings }
    }

    /// Performs selection at the given position and prints results to console.
    fn perform_selection(cursor_x: f32, cursor_y: f32, ctx: &mut EventContext) {
        let camera = ctx.camera();

        let ray = camera.ray_from_screen_point(cursor_x, cursor_y, ctx.size.0, ctx.size.1);

        let camera_distance = (camera.eye - camera.target).magnitude();

        // Line tolerance: 6 pixels in world space, calibrated to the camera target depth.
        // Approximation: uses a single depth reference, so the effective pixel budget
        // varies with geometry depth (near objects get more pixels, far objects fewer).
        let line_tolerance = camera.world_size_per_pixel(camera_distance, ctx.size.1) * 6.0;

        let results = pick_all_from_ray(&RayPickQuery::all(ray, line_tolerance), ctx.scene);

        if let Some(closest_hit) = results.first() {
            let item = resolve_hit_to_selection(closest_hit, ctx.scene);
            ctx.selection.set(item);
        } else {
            ctx.selection.clear();
        }
    }
}

impl Operator for SelectionOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::MouseClick { button, position, .. } = event else { return false };
        if self.bindings.actions_for_click(*button, ctx.modifiers).contains(&SelectionAction::Select) {
            Self::perform_selection(position.0, position.1, ctx);
            true
        } else {
            false
        }
    }

    fn name(&self) -> &str {
        "Selection"
    }
}

/// Resolves a ray pick result to the most specific [`SelectionItem`] available.
///
/// - Triangle hit with topology → `SelectionItem::Face`
/// - Segment hit with topology  → `SelectionItem::Edge`
/// - Anything else              → `SelectionItem::Node`
fn resolve_hit_to_selection(hit: &RayPickResult, scene: &crate::scene::Scene) -> SelectionItem {
    let mesh = scene
        .get_instance(hit.instance_id)
        .and_then(|inst| scene.get_mesh(inst.mesh()));

    match hit.hit {
        RayHit::Triangle { triangle_index, .. } => {
            let face_index = mesh.and_then(|m| m.face_for_triangle(triangle_index as u32));
            match face_index {
                Some(fi) => SelectionItem::Face { node_id: hit.node_id, face_index: fi },
                None => SelectionItem::Node(hit.node_id),
            }
        }
        RayHit::Segment { segment_index, .. } => {
            let edge_index = mesh.and_then(|m| m.edge_for_segment(segment_index as u32));
            match edge_index {
                Some(ei) => SelectionItem::Edge { node_id: hit.node_id, edge_index: ei },
                None => SelectionItem::Node(hit.node_id),
            }
        }
    }
}
