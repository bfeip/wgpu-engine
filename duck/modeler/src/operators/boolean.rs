use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::NodeId;
use duck_engine_scene::Visibility;
use duck_engine_viewer::{
    event::{Event, EventContext},
    input::{ElementState, Key, NamedKey},
    operator::Operator,
    selection::{SelectionItem, SelectionManager},
};

use crate::boolean::{execute_boolean, preview_boolean, BooleanKind};
use crate::document::Document;
use crate::tool::ModelingTool;
use super::ConstructionOptions;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum BooleanPhase {
    #[default]
    Configuring,
    Done,
    Cancelled,
}

pub struct BooleanOperator {
    pub kind: BooleanKind,
    phase: BooleanPhase,

    preview_node: Option<NodeId>,
    hidden_nodes: Vec<NodeId>,

    preview_target: Option<NodeId>,
    preview_tools: Vec<NodeId>,
    last_kind: BooleanKind,

    document: Arc<Mutex<Document>>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
}

impl BooleanOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        Self {
            kind: BooleanKind::default(),
            phase: BooleanPhase::default(),
            preview_node: None,
            hidden_nodes: Vec::new(),
            preview_target: None,
            preview_tools: Vec::new(),
            last_kind: BooleanKind::default(),
            document,
            construction_options,
        }
    }

    /// Execute the boolean operation and clean up preview state.
    /// On success sets phase = Done; on failure stays in Configuring so the user can retry.
    /// Call `selection.clear()` yourself on success.
    pub fn apply(&mut self) -> anyhow::Result<()> {
        let Some(target) = self.preview_target else {
            return Ok(());
        };
        let tools = self.preview_tools.clone();

        let options = self.construction_options.borrow().geometry_preview_options.clone();
        let mut doc = self.document.lock().unwrap();

        // Remove preview node; originals stay hidden — execute_boolean will delete them.
        if let Some(node) = self.preview_node.take() {
            doc.scene().lock().unwrap().remove_node(node);
        }
        self.hidden_nodes.clear();

        execute_boolean(self.kind, target, &tools, &mut *doc, &options)?;

        self.preview_target = None;
        self.preview_tools.clear();
        self.phase = BooleanPhase::Done;
        Ok(())
    }

    /// Abort the operation, restoring the visibility of all hidden original parts.
    pub fn cancel(&mut self) {
        let doc = self.document.lock().unwrap();
        let mut scene = doc.scene().lock().unwrap();
        if let Some(node) = self.preview_node.take() {
            scene.remove_node(node);
        }
        for &node in &self.hidden_nodes {
            scene.set_node_visibility(node, Visibility::Visible);
        }
        drop(scene);
        drop(doc);
        self.hidden_nodes.clear();
        self.preview_target = None;
        self.preview_tools.clear();
        self.phase = BooleanPhase::Cancelled;
    }

    fn selection_snapshot(selection: &SelectionManager) -> (Option<NodeId>, Vec<NodeId>) {
        let primary = selection.primary();
        let target = primary.and_then(|item| match item {
            SelectionItem::Node(id) => Some(id),
            _ => None,
        });
        let tools: Vec<_> = selection.iter()
            .filter(|&&item| Some(item) != primary)
            .filter_map(|item| match item {
                SelectionItem::Node(id) => Some(*id),
                _ => None,
            })
            .collect();
        (target, tools)
    }

    fn refresh_preview(&mut self, selection: &SelectionManager) {
        let doc = self.document.lock().unwrap();

        // Tear down old preview.
        {
            let mut scene = doc.scene().lock().unwrap();
            if let Some(node) = self.preview_node.take() {
                scene.remove_node(node);
            }
            for &node in &self.hidden_nodes {
                scene.set_node_visibility(node, Visibility::Visible);
            }
        }
        self.hidden_nodes.clear();

        let (target, tools) = Self::selection_snapshot(selection);
        self.preview_target = target;
        self.preview_tools = tools.clone();
        self.last_kind = self.kind;

        let Some(target_node) = target else { return };

        let options = self.construction_options.borrow().geometry_preview_options.clone();

        match preview_boolean(self.kind, target_node, &tools, &*doc, &options) {
            Ok(preview) => {
                self.preview_node = Some(preview);
                let mut scene = doc.scene().lock().unwrap();
                scene.set_node_visibility(target_node, Visibility::Invisible);
                for &tool in &tools {
                    scene.set_node_visibility(tool, Visibility::Invisible);
                }
                self.hidden_nodes = std::iter::once(target_node).chain(tools).collect();
            }
            Err(e) => log::warn!("Boolean preview failed: {e}"),
        }
    }
}

impl ModelingTool for BooleanOperator {
    fn deactivate(&mut self) {
        self.cancel();
        self.phase = BooleanPhase::Configuring;
    }

    fn is_finished(&self) -> bool {
        matches!(self.phase, BooleanPhase::Done | BooleanPhase::Cancelled)
    }
}

impl Operator for BooleanOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        match event {
            Event::Update { .. } => {
                let (current_target, current_tools) = Self::selection_snapshot(ctx.selection);
                let selection_changed = current_target != self.preview_target
                    || current_tools != self.preview_tools;
                let kind_changed = self.kind != self.last_kind;
                if selection_changed || kind_changed {
                    self.refresh_preview(ctx.selection);
                }
                false
            }
            Event::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => {
                        if let Err(e) = self.apply() {
                            log::error!("Boolean failed: {e}");
                        } else {
                            ctx.selection.clear();
                        }
                        true
                    }
                    Key::Named(NamedKey::Escape) => {
                        self.cancel();
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Boolean"
    }
}
