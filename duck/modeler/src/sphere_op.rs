use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use duck_engine_common::{MetricSpace, Point3, Quaternion, Vector3};
use duck_engine_common::RgbaColor;
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::{tessellate_into, CadTessellationOptions};
use duck_engine_viewer::{
    common::{Ray, Transform},
    event::{CallbackId, Event, EventDispatcher, EventKind},
    input::MouseButton,
    operator::Operator,
};
use opencascade::primitives::Shape;

use crate::document::{CadDocument, PartId};

enum Phase {
    Idle,
    Defining { center: Point3, preview_node: NodeId },
}

struct Inner {
    phase: Phase,
}

pub struct SphereOperator {
    inner: Rc<RefCell<Inner>>,
    callback_ids: Vec<CallbackId>,
    document: Rc<RefCell<CadDocument>>,
    node_map: Rc<RefCell<HashMap<PartId, NodeId>>>,
}

impl SphereOperator {
    pub fn new(
        document: Rc<RefCell<CadDocument>>,
        node_map: Rc<RefCell<HashMap<PartId, NodeId>>>,
    ) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Inner { phase: Phase::Idle })),
            callback_ids: Vec::new(),
            document,
            node_map,
        }
    }
}

fn intersect_xz(ray: &Ray) -> Option<Point3> {
    if ray.direction.y.abs() < 1e-6 {
        return None;
    }
    let t = -ray.origin.y / ray.direction.y;
    if t < 0.0 {
        return None;
    }
    Some(ray.origin + t * ray.direction)
}

fn preview_options() -> CadTessellationOptions {
    CadTessellationOptions {
        tessellation_tolerance: 0.05,
        scale_factor: 1.0,
        face_color: RgbaColor { r: 0.55, g: 0.65, b: 0.9, a: 1.0 },
        edge_color: RgbaColor { r: 0.2, g: 0.2, b: 0.5, a: 1.0 },
        include_edges: true,
    }
}

fn sphere_transform(center: Point3, radius: f32) -> Transform {
    Transform {
        position: center,
        rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
        scale: Vector3::new(radius, radius, radius),
    }
}

impl Operator for SphereOperator {
    fn activate(&mut self, dispatcher: &mut EventDispatcher) {
        let inner = self.inner.clone();
        let document = self.document.clone();
        let node_map = self.node_map.clone();

        let click_cb = dispatcher.register(EventKind::MouseClick, move |event, ctx| {
            let Event::MouseClick { button, position, .. } = event else {
                return false;
            };

            let mut state = inner.borrow_mut();

            match (&state.phase, button) {
                (Phase::Idle, MouseButton::Left) => {
                    let ray = ctx.camera().ray_from_screen_point(
                        position.0, position.1, ctx.size.0, ctx.size.1,
                    );
                    let Some(center) = intersect_xz(&ray) else { return false };

                    let shape = Shape::sphere(1.0).build();
                    let Ok(node) =
                        tessellate_into(&shape, ctx.scene, &preview_options(), None, Some("sphere"))
                    else {
                        return false;
                    };

                    ctx.scene.set_node_transform(node, sphere_transform(center, 0.01));
                    state.phase = Phase::Defining { center, preview_node: node };
                    true
                }
                (Phase::Defining { center, preview_node }, MouseButton::Left) => {
                    let (center, node) = (*center, *preview_node);
                    let ray = ctx.camera().ray_from_screen_point(
                        position.0, position.1, ctx.size.0, ctx.size.1,
                    );
                    let radius = intersect_xz(&ray)
                        .map(|hit| center.distance(hit).max(0.01))
                        .unwrap_or(0.01);

                    let transform = sphere_transform(center, radius);
                    ctx.scene.set_node_transform(node, transform);

                    // Commit to the CAD document; reuse the preview scene node.
                    let shape = Shape::sphere(1.0).build();
                    let part_id = document.borrow_mut().add_part(
                        "Sphere".to_owned(),
                        shape,
                        transform,
                        RgbaColor { r: 0.55, g: 0.65, b: 0.9, a: 1.0 },
                    );
                    node_map.borrow_mut().insert(part_id, node);

                    state.phase = Phase::Idle;
                    true
                }
                (Phase::Defining { preview_node, .. }, MouseButton::Right) => {
                    let node = *preview_node;
                    ctx.scene.remove_node(node);
                    state.phase = Phase::Idle;
                    true
                }
                _ => false,
            }
        });

        let inner = self.inner.clone();
        let move_cb = dispatcher.register(EventKind::CursorMoved, move |event, ctx| {
            let Event::CursorMoved { position } = event else { return false };

            let state = inner.borrow();
            let Phase::Defining { center, preview_node } = state.phase else { return false };

            let ray = ctx.camera().ray_from_screen_point(
                position.0 as f32, position.1 as f32, ctx.size.0, ctx.size.1,
            );
            if let Some(hit) = intersect_xz(&ray) {
                let radius = center.distance(hit).max(0.01);
                ctx.scene.set_node_transform(preview_node, sphere_transform(center, radius));
            }
            false
        });

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
