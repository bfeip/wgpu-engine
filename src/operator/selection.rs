use crate::common::{Ray, RgbaColor};
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::geom_query::pick_all_from_ray;
use crate::operator::{Operator, OperatorId};
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

        // Debug output
        println!("\n=== Selection Debug ===");
        println!("Screen: {}x{}", ctx.state.size.width, ctx.state.size.height);
        println!("Cursor: ({:.1}, {:.1})", cursor_x, cursor_y);
        println!("Camera: eye={:?}, target={:?}", ctx.state.camera.eye, ctx.state.camera.target);

        // Create ray from screen point
        let ray = Ray::from_screen_point(
            cursor_x,
            cursor_y,
            ctx.state.size.width,
            ctx.state.size.height,
            &ctx.state.camera,
        );

        println!("Ray: origin={:?}, direction={:?}", ray.origin, ray.direction);

        // Calculate camera distance for miss visualization
        let camera_distance = (ctx.state.camera.eye - ctx.state.camera.target).magnitude();

        // Perform picking
        let results = pick_all_from_ray(&ray, ctx.scene);

        // Convert ray origin Vector3 to Point3
        let ray_origin = Point3::new(ray.origin.x, ray.origin.y, ray.origin.z);

        // Print results to console
        if results.is_empty() {
            println!("Selection: No objects hit");

            // Debug: Check scene bounds (excluding annotations)
            println!("\nScene info:");
            println!("  Root nodes: {}", ctx.scene.root_nodes().len());

            let annotation_root = ctx.annotation_manager.root_node();
            let mut closest_bbox_hit: Option<f32> = None;

            for &root_id in ctx.scene.root_nodes() {
                // Skip the annotation root node to avoid hitting our debug rays
                if root_id == annotation_root {
                    continue;
                }

                if let Some(bounds) = ctx.scene.nodes_bounding(root_id) {
                    println!("  Root node {} bounds: min={:?}, max={:?}", root_id, bounds.min, bounds.max);

                    // Test if ray hits the bounding box
                    if let Some(t) = bounds.intersects_ray(&ray) {
                        println!("    Ray HITS bounds at t={:.3}", t);
                        closest_bbox_hit = Some(match closest_bbox_hit {
                            Some(existing) => existing.min(t),
                            None => t,
                        });
                    } else {
                        println!("    Ray MISSES bounds");
                    }
                }
            }

            // Draw debug ray based on what was hit
            if let Some(bbox_t) = closest_bbox_hit {
                // Yellow ray: hit bounding box but not geometry
                let hit_point = ray_origin + ray.direction * bbox_t;
                let _ = ctx.annotation_manager.add_line(
                    ctx.scene,
                    &ctx.state.device,
                    &mut ctx.state.material_manager,
                    ray_origin,
                    hit_point,
                    RgbaColor { r: 1.0, g: 1.0, b: 0.0, a: 1.0 }, // Yellow
                );
                println!("Drew YELLOW debug ray (bbox hit, no geometry)");
            } else {
                // Red ray: complete miss
                let end_point = ray_origin + ray.direction * camera_distance;
                let _ = ctx.annotation_manager.add_line(
                    ctx.scene,
                    &ctx.state.device,
                    &mut ctx.state.material_manager,
                    ray_origin,
                    end_point,
                    RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, // Red
                );
                println!("Drew RED debug ray (complete miss)");
            }
        } else {
            println!("Selection: Found {} hit(s)", results.len());
            for (i, result) in results.iter().enumerate() {
                println!(
                    "  [{}] Node ID: {}, Instance ID: {}, Distance: {}, Hit point: ({}, {}, {})",
                    i,
                    result.node_id,
                    result.instance_id,
                    result.distance,
                    result.hit_point.x,
                    result.hit_point.y,
                    result.hit_point.z
                );
            }

            // Draw green ray to the closest hit point
            if let Some(closest_hit) = results.first() {
                let hit_point = Point3::new(
                    closest_hit.hit_point.x,
                    closest_hit.hit_point.y,
                    closest_hit.hit_point.z,
                );
                let _ = ctx.annotation_manager.add_line(
                    ctx.scene,
                    &ctx.state.device,
                    &mut ctx.state.material_manager,
                    ray_origin,
                    hit_point,
                    RgbaColor { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }, // Green
                );
                println!("Drew GREEN debug ray (geometry hit)");
            }
        }
        println!("======================\n");
    }
}

impl Operator for SelectionOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        // Register MouseClick handler for selection
        let mouse_click_callback = dispatcher.register(EventKind::MouseClick, move |event, ctx| {
            let Event::MouseClick { button, position, .. } = event else {
                return false;
            };

            use winit::event::MouseButton;

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
