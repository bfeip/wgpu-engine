use crate::common::RgbaColor;
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::geom_query::{RayHit, RayPickQuery, RayPickResult, pick_all_from_ray};
use crate::operator::{Operator, OperatorId};
use crate::selection::SelectionItem;
use cgmath::{InnerSpace, Point3};

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
        // Create ray from screen point
        let ray = ctx.camera.ray_from_screen_point(
            cursor_x,
            cursor_y,
            ctx.size.0,
            ctx.size.1,
        );

        // Calculate camera distance for miss visualization and line tolerance
        let camera_distance = (ctx.camera.eye - ctx.camera.target).magnitude();

        // Line tolerance: 6 pixels in world space, calibrated to the camera target depth.
        // Approximation: uses a single depth reference, so the effective pixel budget
        // varies with geometry depth (near objects get more pixels, far objects fewer).
        let line_tolerance = ctx.camera.world_size_per_pixel(camera_distance, ctx.size.1) * 6.0;

        // Perform picking
        let results = pick_all_from_ray(&RayPickQuery::all(ray, line_tolerance), ctx.scene);

        // Convert ray origin Vector3 to Point3
        let ray_origin = Point3::new(ray.origin.x, ray.origin.y, ray.origin.z);

        // Draw debug annotation lines if enabled
        if ctx.selection.config().debug_annotations {
            draw_debug_annotations(&ray, ray_origin, camera_distance, &results, ctx);
        }

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

/// Draws debug pick-ray annotations into the scene.
///
/// - Green ray to the closest geometry hit
/// - Yellow ray to a bounding-box hit (geometry miss)
/// - Red ray into the void on a complete miss
fn draw_debug_annotations(
    ray: &crate::common::Ray,
    ray_origin: Point3<f32>,
    camera_distance: f32,
    results: &[RayPickResult],
    ctx: &mut EventContext,
) {
    if results.is_empty() {
        let annotation_root = ctx.scene.annotations.root_node();
        let mut closest_bbox_hit: Option<f32> = None;

        for &root_id in ctx.scene.root_nodes() {
            // Skip the annotation root node to avoid hitting our debug rays
            if annotation_root == Some(root_id) {
                continue;
            }

            if let Some(bounds) = ctx.scene.nodes_bounding(root_id) {
                // Test if ray hits the bounding box
                if let Some(t) = bounds.intersects_ray(ray) {
                    closest_bbox_hit = Some(match closest_bbox_hit {
                        Some(existing) => existing.min(t),
                        None => t,
                    });
                }
            }
        }

        // Draw debug ray based on what was hit
        if let Some(bbox_t) = closest_bbox_hit {
            // Yellow ray: hit bounding box but not geometry
            let hit_point = ray_origin + ray.direction * bbox_t;
            ctx.scene.annotations.add_line(
                ray_origin,
                hit_point,
                RgbaColor { r: 1.0, g: 1.0, b: 0.0, a: 1.0 }, // Yellow
            );
        } else {
            // Red ray: complete miss
            let end_point = ray_origin + ray.direction * camera_distance;
            ctx.scene.annotations.add_line(
                ray_origin,
                end_point,
                RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, // Red
            );
        }
    } else if let Some(closest_hit) = results.first() {
        // Draw green ray to the closest hit point
        let hit_point = Point3::new(
            closest_hit.hit_point.x,
            closest_hit.hit_point.y,
            closest_hit.hit_point.z,
        );
        ctx.scene.annotations.add_line(
            ray_origin,
            hit_point,
            RgbaColor { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }, // Green
        );
    }
}