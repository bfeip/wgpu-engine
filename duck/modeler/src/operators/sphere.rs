use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::rc::Rc;

use duck_engine_common::{MetricSpace, Point3, Quaternion, Vector3};
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::tessellate_into;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{Event, EventContext},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use glam::dvec3;
use opencascade::primitives::Shape;

use crate::document::Document;
use crate::tool::ModelingTool;
use super::ConstructionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SphereAction {
    Place,
    Cancel,
}

enum Phase {
    Idle,
    Defining { center: Point3, preview_node: NodeId },
}

pub struct SphereOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    bindings: InputMap<SphereAction>,
}

impl SphereOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                SphereAction::Place,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                SphereAction::Cancel,
            );
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            bindings,
        }
    }

    fn preview_transform(center: Point3, radius: f32) -> Transform {
        Transform {
            position: center,
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale: Vector3::new(radius, radius, radius),
        }
    }

    fn on_place_center(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let coptions = self.construction_options.borrow();
        let ray = ctx.camera().ray_from_screen_point(
            position.0, position.1, ctx.size.0, ctx.size.1,
        );
        let Some((_, center)) = ray.intersect_plane(&coptions.construction_plane) else {
            return false;
        };
        let preview_shape = Shape::sphere(1.0).build();
        let preview_node = {
            let mut scene = ctx.scene.lock().unwrap();
            let Ok(node) = tessellate_into(
                &preview_shape,
                &mut *scene,
                &coptions.geometry_preview_options,
                None,
                Some("sphere"),
            ) else {
                return false;
            };
            scene.set_node_transform(node, Self::preview_transform(center, 0.01));
            node
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
        let coptions = self.construction_options.borrow();
        let ray = ctx.camera().ray_from_screen_point(
            position.0, position.1, ctx.size.0, ctx.size.1,
        );
        let radius = ray
            .intersect_plane(&coptions.construction_plane)
            .map(|(_, hit)| center.distance(hit).max(0.01))
            .unwrap_or(0.01);

        let world_shape = Shape::sphere(radius as f64)
            .at(dvec3(center.x as f64, center.y as f64, center.z as f64))
            .build();

        // Discard the preview node, then commit the world-space shape as a registered part.
        ctx.scene.lock().unwrap().remove_node(preview_node);

        let mut doc = self.document.lock().unwrap();
        if doc.add_part(
            "Sphere".to_owned(),
            world_shape,
            coptions.geometry_preview_options.face_color,
            &coptions.geometry_preview_options,
        ).is_err() {
            self.phase = Phase::Idle;
            return false;
        }
        self.phase = Phase::Idle;
        true
    }

    pub fn cancel(&mut self) {
        if let Phase::Defining { preview_node, .. } = self.phase {
            self.document.lock().unwrap().scene().lock().unwrap().remove_node(preview_node);
            self.phase = Phase::Idle;
        }
    }

    fn on_cursor_moved(&self, position: (f64, f64), ctx: &mut EventContext) {
        let Phase::Defining { center, preview_node } = self.phase else { return };
        let coptions = self.construction_options.borrow();
        let ray = ctx.camera().ray_from_screen_point(
            position.0 as f32, position.1 as f32, ctx.size.0, ctx.size.1,
        );
        if let Some((_, hit)) = ray.intersect_plane(&coptions.construction_plane) {
            let radius = center.distance(hit).max(0.01);
            ctx.scene.lock().unwrap().set_node_transform(preview_node, Self::preview_transform(center, radius));
        }
    }
}

impl ModelingTool for SphereOperator {
    fn deactivate(&mut self) {
        self.cancel();
    }
}

impl Operator for SphereOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        match event {
            Event::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        SphereAction::Place => {
                            if let Phase::Defining { center, preview_node } = self.phase {
                                self.on_place_outer(center, preview_node, *position, ctx)
                            } else {
                                self.on_place_center(*position, ctx)
                            }
                        }
                        SphereAction::Cancel => {
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
            Event::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Sphere"
    }
}
