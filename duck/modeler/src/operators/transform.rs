use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_viewer::event::{AppEvent, Event, EventContext};
use duck_engine_viewer::operator::{Operator, SelectionMode, TransformOperator};

use crate::document::Document;
use crate::tool::{ModelingTool, ToolInfo};
use super::ConstructionOptions;

/// CAD-aware transform tool: translate / rotate / scale of whole parts.
///
/// This is a thin wrapper around the viewer's [`TransformOperator`], which does
/// all the interactive work (gizmos, axis constraints, mouse math, preview,
/// confirm/cancel) by mutating the *scene node* transform. That moves the
/// tessellated mesh but leaves the underlying CAD B-Rep untouched, so on every
/// confirmed transform we bake the node's final transform into the part's
/// geometry via [`Document::bake_transform`], keeping the CAD shape and the
/// rendered mesh in sync.
///
/// Controls match the viewer operator: G/R/S to grab/rotate/scale, X/Y/Z to
/// constrain to an axis, left-click or Enter to confirm, right-click or Escape
/// to cancel.
pub struct TransformTool {
    transform_op: TransformOperator,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
}

impl TransformTool {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        Self {
            transform_op: TransformOperator::new(),
            construction_options,
            document,
        }
    }

    /// Bake every just-confirmed node's transform into its CAD part.
    fn bake_committed(&mut self, nodes: &[duck_engine_viewer::scene::NodeId], ctx: &mut EventContext) {
        let options = self.construction_options.borrow().geometry_options.clone();
        let mut doc = self.document.lock().unwrap();
        for &node in nodes {
            let Some(part) = doc.part_for_node(node) else { continue };
            // Stay in common's matrix type; the OCCT array conversion happens
            // inside bake_transform.
            let delta = ctx
                .scene
                .lock()
                .unwrap()
                .get_node(node)
                .map(|n| n.transform().to_matrix());
            if let Some(delta) = delta {
                if let Err(e) = doc.bake_transform(part, delta, &options) {
                    log::error!("transform bake failed for node {node:?}: {e}");
                }
            }
        }
    }
}

impl Operator for TransformTool {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        // The inner operator emits `AppEvent::TransformCommitted` on confirm; the
        // dispatcher re-dispatches it back to us. Bake the committed nodes when we see
        // it.
        if let Event::App(AppEvent::TransformCommitted { nodes }) = event {
            self.bake_committed(nodes, ctx);
            return false;
        }
        self.transform_op.dispatch(event, ctx)
    }

    fn name(&self) -> &str {
        "TransformTool"
    }
}

impl ModelingTool for TransformTool {
    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: "transform",
            icon_uri: "bytes://move-arrows.svg",
            icon: include_bytes!("../../../../assets/svg/move-arrows-svgrepo-com.svg"),
        }
    }

    fn deactivate(&mut self) {
        // Abort any in-progress transform and remove gizmo/annotation nodes.
        let scene_arc = self.document.lock().unwrap().scene().clone();
        self.transform_op.teardown(&mut scene_arc.lock().unwrap());
    }

    /// Transform whole parts, not sub-geometry.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }
}
