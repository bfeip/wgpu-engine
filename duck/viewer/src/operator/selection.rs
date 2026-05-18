use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::geom_query::{RayHit, RayPickQuery, RayPickResult, pick_all_from_ray};
use crate::operator::{Operator, OperatorId};
use crate::selection::SelectionItem;
use cgmath::{InnerSpace};

/// Operator for selecting objects in the scene via mouse click.
///
/// - Left mouse button click: Cast a ray and print the selected object to console
pub struct SelectionOperator {
    id: OperatorId,
    callback_ids: Vec<CallbackId>,
}

impl SelectionOperator {
    /// Creates a new selection operator with the given ID.
    pub fn new(id: OperatorId) -> Self {
        Self {
            id,
            callback_ids: Vec::new(),
        }
    }

    /// Performs selection at the given position and prints results to console.
    fn perform_selection(cursor_x: f32, cursor_y: f32, ctx: &mut EventContext) {
        let camera = ctx.camera();

        // Create ray from screen point
        let ray = camera.ray_from_screen_point(
            cursor_x,
            cursor_y,
            ctx.size.0,
            ctx.size.1,
        );

        // Calculate camera distance for miss visualization and line tolerance
        let camera_distance = (camera.eye - camera.target).magnitude();

        // Line tolerance: 6 pixels in world space, calibrated to the camera target depth.
        // Approximation: uses a single depth reference, so the effective pixel budget
        // varies with geometry depth (near objects get more pixels, far objects fewer).
        let line_tolerance = camera.world_size_per_pixel(camera_distance, ctx.size.1) * 6.0;

        // Perform picking
        let results = pick_all_from_ray(&RayPickQuery::all(ray, line_tolerance), ctx.scene);

        // Update selection based on pick results
        if let Some(closest_hit) = results.first() {
            let item = resolve_hit_to_selection(closest_hit, ctx.scene);
            ctx.selection.set(item);
        } else {
            ctx.selection.clear();
        }
    }
}

impl Operator for SelectionOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        // Register MouseClick handler for selection
        let mouse_click_callback = dispatcher.register(EventKind::MouseClick, move |event, ctx| {
            let Event::MouseClick { button, position, .. } = event else {
                return false;
            };

            use crate::input::MouseButton;

            // Only handle left-click events
            if matches!(button, MouseButton::Left) {
                SelectionOperator::perform_selection(position.0, position.1, ctx);
                true // Stop event propagation (we handled the click)
            } else {
                false
            }
        });

        self.callback_ids = vec![mouse_click_callback];
    }

    fn deactivate(&mut self, dispatcher: &mut EventDispatcher) {
        for id in &self.callback_ids {
            dispatcher.unregister(*id);
        }
        self.callback_ids.clear();
    }

    fn id(&self) -> OperatorId {
        self.id
    }

    fn name(&self) -> &str {
        "Selection"
    }

    fn callback_ids(&self) -> &[CallbackId] {
        &self.callback_ids
    }

    fn is_active(&self) -> bool {
        !self.callback_ids.is_empty()
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