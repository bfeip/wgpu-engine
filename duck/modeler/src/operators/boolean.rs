use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::NodeId;
use duck_engine_scene::Visibility;
use duck_engine_viewer::{
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, NamedKey},
    operator::{Operator, SelectionMode},
    selection::{SelectionItem, SelectionManager},
};

use crate::boolean::{execute_boolean, preview_boolean, BooleanKind};
use crate::document::Document;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};
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
    fn apply(&mut self) -> anyhow::Result<()> {
        let Some(target) = self.preview_target else {
            return Ok(());
        };
        let tools = self.preview_tools.clone();

        let options = self.construction_options.borrow().geometry_options.clone();
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

    /// Apply and, on success, clear the selection. Shared by the Enter key
    /// handler and the panel's Apply button.
    pub fn apply_and_clear(&mut self, selection: &mut SelectionManager) {
        if let Err(e) = self.apply() {
            log::error!("Boolean failed: {e}");
        } else {
            selection.clear();
        }
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

        let options = self.construction_options.borrow().preview_options();

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

    /// The boolean configuration window (operation kind, target/tool parts,
    /// Apply/Cancel). Part names are snapshotted under the document lock before
    /// rendering so the egui closure holds no locks.
    fn render_panel(&mut self, ctx: &egui::Context, panel: &mut PanelContext) {
        let mut apply_clicked = false;
        let mut cancel_clicked = false;

        let primary = panel.selection.primary();
        let target_node = primary.and_then(|item| match item {
            SelectionItem::Node(id) => Some(id),
            _ => None,
        });
        let tool_items: Vec<SelectionItem> = panel.selection.iter()
            .filter(|&&item| Some(item) != primary)
            .copied()
            .collect();

        let (target_name, tool_entries) = {
            let doc = self.document.lock().unwrap();
            let name_for = |node: NodeId| {
                doc.part_for_node(node)
                    .and_then(|p| doc.get_part(p).map(|part| part.name.clone()))
            };
            let target_name = target_node
                .and_then(name_for)
                .unwrap_or_else(|| "(none — click a part)".to_owned());
            let tool_entries: Vec<(SelectionItem, String)> = tool_items
                .into_iter()
                .map(|item| {
                    let name = match item {
                        SelectionItem::Node(id) => name_for(id),
                        _ => None,
                    }
                    .unwrap_or_else(|| "Unknown".to_owned());
                    (item, name)
                })
                .collect();
            (target_name, tool_entries)
        };

        egui::Window::new("Boolean Operation")
            .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label("Operation");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.kind, BooleanKind::Subtract, "Subtract");
                    ui.selectable_value(&mut self.kind, BooleanKind::Union, "Union");
                    ui.selectable_value(&mut self.kind, BooleanKind::Intersect, "Intersect");
                });

                ui.separator();

                ui.label("Target");
                ui.label(&target_name);

                ui.separator();

                ui.label("Tools");
                if tool_entries.is_empty() {
                    ui.label("(shift-click parts to add tools)");
                } else {
                    for (item, name) in &tool_entries {
                        ui.horizontal(|ui| {
                            ui.label(name);
                            if ui.small_button("×").clicked() {
                                panel.selection.remove(item);
                            }
                        });
                    }
                }

                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel_clicked = true;
                    }
                    if ui.button("Apply  ⏎").clicked() {
                        apply_clicked = true;
                    }
                });
            });

        // Act after the window closure so we don't call &mut self methods
        // while it still borrows self.
        if apply_clicked {
            self.apply_and_clear(panel.selection);
        } else if cancel_clicked {
            self.cancel();
        }
    }
}

impl ModelingTool for BooleanOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo {
            id: "boolean",
            icon_uri: "bytes://boolean-and.svg",
            icon: include_bytes!("../../../../assets/svg/boolean-and.svg"),
        }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.phase = BooleanPhase::Configuring;
    }

    fn is_finished(&self) -> bool {
        matches!(self.phase, BooleanPhase::Done | BooleanPhase::Cancelled)
    }

    // Boolean operates on whole parts, so drop the always-on selection
    // operator to node granularity while active.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Node
    }

    fn panel_ui(&mut self, ctx: &egui::Context, panel: &mut PanelContext) {
        self.render_panel(ctx, panel);
    }
}

impl Operator for BooleanOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::Update { .. } => {
                let (current_target, current_tools) = Self::selection_snapshot(ctx.selection);
                let selection_changed = current_target != self.preview_target
                    || current_tools != self.preview_tools;
                let kind_changed = self.kind != self.last_kind;
                if selection_changed || kind_changed {
                    self.refresh_preview(ctx.selection);
                }
                false
            }
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => {
                        self.apply_and_clear(ctx.selection);
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
