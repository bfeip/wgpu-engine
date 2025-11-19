use crate::common::Ray;
use crate::event::{CallbackId, Event, EventContext, EventDispatcher, EventKind};
use crate::geom_query::pick_all_from_ray;
use crate::operator::{Operator, OperatorId};

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

        // Perform picking
        let results = pick_all_from_ray(&ray, ctx.scene);

        // Print results to console
        if results.is_empty() {
            println!("Selection: No objects hit");

            // Debug: Check scene bounds
            println!("\nScene info:");
            println!("  Root nodes: {}", ctx.scene.root_nodes().len());
            for &root_id in ctx.scene.root_nodes() {
                if let Some(bounds) = ctx.scene.nodes_bounding(root_id) {
                    println!("  Root node {} bounds: min={:?}, max={:?}", root_id, bounds.min, bounds.max);

                    // Test if ray hits the bounding box
                    if let Some(t) = bounds.intersects_ray(&ray) {
                        println!("    Ray HITS bounds at t={:.3}", t);
                    } else {
                        println!("    Ray MISSES bounds");
                    }
                }
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
