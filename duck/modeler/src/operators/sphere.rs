use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use duck_engine_common::{MetricSpace, Point3, Quaternion, Vector3};
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::tessellate_into;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    common::Transform,
    event::{CallbackId, Event, EventContext, EventDispatcher, EventKind},
    input::{Modifiers, MouseButton},
    operator::Operator,
};
use opencascade::primitives::Shape;

use crate::{document::{CadDocument, PartId}};
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

struct Inner {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Rc<RefCell<CadDocument>>,
    node_map: Rc<RefCell<HashMap<PartId, NodeId>>>,
}

impl Inner {
    fn sphere_transform(center: Point3, radius: f32) -> Transform {
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
        let shape = Shape::sphere(1.0).build();
        let Ok(preview_node) = tessellate_into(
            &shape,
            ctx.scene,
            &coptions.geometry_preview_options,
            None,
            Some("sphere"),
        ) else {
            return false;
        };
        ctx.scene.set_node_transform(preview_node, Self::sphere_transform(center, 0.01));
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
        let transform = Self::sphere_transform(center, radius);
        ctx.scene.set_node_transform(preview_node, transform);
        // Commit to the CAD document; reuse the preview scene node.
        let shape = Shape::sphere(1.0).build();
        let part_id = self.document.borrow_mut().add_part(
            "Sphere".to_owned(),
            shape,
            transform,
            coptions.geometry_preview_options.face_color,
        );
        self.node_map.borrow_mut().insert(part_id, preview_node);
        self.phase = Phase::Idle;
        true
    }

    fn on_cancel(&mut self, preview_node: NodeId, ctx: &mut EventContext) -> bool {
        ctx.scene.remove_node(preview_node);
        self.phase = Phase::Idle;
        true
    }

    fn on_cursor_moved(&self, position: (f64, f64), ctx: &mut EventContext) {
        let Phase::Defining { center, preview_node } = self.phase else { return };
        let coptions = self.construction_options.borrow();
        let ray = ctx.camera().ray_from_screen_point(
            position.0 as f32, position.1 as f32, ctx.size.0, ctx.size.1,
        );
        if let Some((_, hit)) = ray.intersect_plane(&coptions.construction_plane) {
            let radius = center.distance(hit).max(0.01);
            ctx.scene.set_node_transform(preview_node, Self::sphere_transform(center, radius));
        }
    }
}

pub struct SphereOperator {
    inner: Rc<RefCell<Inner>>,
    callback_ids: Vec<CallbackId>,
    bindings: Rc<InputMap<SphereAction>>,
}

impl SphereOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Rc<RefCell<CadDocument>>,
        node_map: Rc<RefCell<HashMap<PartId, NodeId>>>,
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
            inner: Rc::new(RefCell::new(Inner {
                phase: Phase::Idle,
                construction_options,
                document,
                node_map,
            })),
            callback_ids: Vec::new(),
            bindings: Rc::new(bindings),
        }
    }
}

impl SphereOperator {
    fn register_click_handler(&self, dispatcher: &mut EventDispatcher) -> CallbackId {
        let inner = self.inner.clone();
        let bindings = self.bindings.clone();

        dispatcher.register(EventKind::MouseClick, move |event, ctx| {
            let Event::MouseClick { button, position, .. } = event else { return false };
            let actions = bindings.actions_for_click(*button, ctx.modifiers).to_vec();
            let mut state = inner.borrow_mut();
            let mut handled = false;

            for action in actions {
                match (&state.phase, action) {
                    (Phase::Idle, SphereAction::Place) => {
                        handled |= state.on_place_center(*position, ctx);
                    }
                    (Phase::Defining { center, preview_node }, SphereAction::Place) => {
                        let (c, n) = (*center, *preview_node);
                        handled |= state.on_place_outer(c, n, *position, ctx);
                    }
                    (Phase::Defining { preview_node, .. }, SphereAction::Cancel) => {
                        let n = *preview_node;
                        handled |= state.on_cancel(n, ctx);
                    }
                    _ => {}
                }
            }
            handled
        })
    }

    fn register_move_handler(&self, dispatcher: &mut EventDispatcher) -> CallbackId {
        let inner = self.inner.clone();

        dispatcher.register(EventKind::CursorMoved, move |event, ctx| {
            let Event::CursorMoved { position } = event else { return false };
            inner.borrow().on_cursor_moved(*position, ctx);
            false
        })
    }
}

impl Operator for SphereOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        let click_cb = self.register_click_handler(dispatcher);
        let move_cb = self.register_move_handler(dispatcher);
        self.callback_ids = vec![click_cb, move_cb];
    }

    fn deactivate(&mut self, dispatcher: &mut EventDispatcher) {
        for id in &self.callback_ids {
            dispatcher.unregister(*id);
        }
        self.callback_ids.clear();
    }

    fn name(&self) -> &str {
        "Sphere"
    }

    fn callback_ids(&self) -> &[CallbackId] {
        &self.callback_ids
    }

    fn is_active(&self) -> bool {
        !self.callback_ids.is_empty()
    }
}
