use std::sync::{Arc, Mutex};

use duck_engine_viewer::event::{Event, EventContext, EventDispatcher};
use duck_engine_viewer::operator::{Operator, SelectionMode, SelectionOperator};
use duck_engine_viewer::scene::Scene;
use duck_engine_viewer::selection::SelectionManager;

use crate::cursor::Cursor3d;
use crate::tool::{ModelingTool, PanelContext, ToolInfo};

/// The single dispatcher-registered operator for all modeling tools. Forwards
/// events to the active tool, if any. Registered once at startup, in front of
/// the selection/navigation operators; never added or removed afterwards, so
/// switching tools involves no dispatcher churn.
struct ToolHost {
    active: Option<Arc<Mutex<dyn ModelingTool>>>,
}

impl Operator for ToolHost {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        match &self.active {
            Some(tool) => tool.lock().unwrap().dispatch(event, ctx),
            None => false,
        }
    }

    fn name(&self) -> &str {
        "ToolHost"
    }
}

/// Owns the registered modeling tools and everything generic about driving
/// them: activation/deactivation, the always-on selection operator's
/// granularity, auto-return to selection when a tool finishes, the 3D cursor,
/// and the tool palette. Adding a tool to the modeler is implementing
/// [`ModelingTool`] plus one [`ToolManager::register`] call.
pub struct ToolManager {
    tools: Vec<Arc<Mutex<dyn ModelingTool>>>,
    /// Index into `tools`; `None` means plain selection mode.
    active: Option<usize>,
    host: Arc<Mutex<ToolHost>>,
    sel_op: Arc<Mutex<SelectionOperator>>,
    /// The modeler-owned 3D cursor, driven each frame from the active tool.
    cursor: Cursor3d,
}

impl ToolManager {
    pub fn new(sel_op: Arc<Mutex<SelectionOperator>>) -> Self {
        Self {
            tools: Vec::new(),
            active: None,
            host: Arc::new(Mutex::new(ToolHost { active: None })),
            sel_op,
            cursor: Cursor3d::default(),
        }
    }

    /// Registers the forwarding host with the dispatcher at highest priority.
    /// Call once, after the selection/navigation operators are registered.
    pub fn install(&self, dispatcher: &mut EventDispatcher) {
        dispatcher.push_front(Arc::clone(&self.host));
    }

    pub fn register<T: ModelingTool>(&mut self, tool: T) {
        self.tools.push(Arc::new(Mutex::new(tool)));
    }

    /// Switches the active tool; `None` returns to plain selection.
    /// Re-activating the already active tool is a no-op.
    ///
    /// Locks are taken strictly one at a time: old tool (deactivate), host
    /// (swap), new tool (activate + selection mode), selection operator.
    pub fn activate(&mut self, index: Option<usize>) {
        if index == self.active {
            return;
        }

        if let Some(old) = self.active {
            self.tools[old].lock().unwrap().deactivate();
        }

        self.host.lock().unwrap().active = index.map(|i| Arc::clone(&self.tools[i]));

        let mode = match index {
            Some(i) => {
                let mut tool = self.tools[i].lock().unwrap();
                tool.activate();
                tool.selection_mode()
            }
            None => SelectionMode::SubGeometry,
        };
        self.sel_op.lock().unwrap().mode = mode;

        self.active = index;
    }

    /// Per-frame upkeep: cede back to selection once the active tool reports
    /// finished, then drive the 3D cursor from the active tool's target.
    /// Runs every frame so the cursor marker also rescales with the camera.
    pub fn update(&mut self, scene: &Arc<Mutex<Scene>>) {
        if self.active.is_some_and(|i| self.tools[i].lock().unwrap().is_finished()) {
            self.activate(None);
        }

        let target = self
            .active
            .and_then(|i| self.tools[i].lock().unwrap().cursor_target());
        self.cursor.update(target, &mut scene.lock().unwrap());
    }

    /// The left tool palette: a Select button plus one button per registered
    /// tool. Tool clicks are applied after the panel closure returns so no
    /// tool lock is held while egui renders.
    pub fn palette_ui(&mut self, ctx: &egui::Context) {
        const CURSOR_SVG: &[u8] = include_bytes!("../../../assets/svg/cursor-svgrepo-com.svg");

        let entries: Vec<(ToolInfo, bool)> = self
            .tools
            .iter()
            .enumerate()
            .map(|(i, tool)| (tool.lock().unwrap().info(), self.active == Some(i)))
            .collect();

        let mut clicked: Option<Option<usize>> = None;

        egui::SidePanel::left("operator_palette")
            .resizable(false)
            .exact_width(56.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);

                let select_btn = ui.add(
                    egui::Button::image(
                        egui::Image::from_bytes("bytes://cursor.svg", CURSOR_SVG)
                            .fit_to_exact_size(egui::vec2(32.0, 32.0)),
                    )
                    .selected(self.active.is_none()),
                )
                .on_hover_text("select");
                if select_btn.clicked() {
                    clicked = Some(None);
                }

                for (i, (info, selected)) in entries.iter().enumerate() {
                    ui.add_space(4.0);
                    let btn = ui.add(
                        egui::Button::image(
                            egui::Image::from_bytes(info.icon_uri, info.icon)
                                .fit_to_exact_size(egui::vec2(32.0, 32.0)),
                        )
                        .selected(*selected),
                    )
                    .on_hover_text(info.id);
                    if btn.clicked() {
                        clicked = Some(Some(i));
                    }
                }
            });

        if let Some(index) = clicked {
            self.activate(index);
        }
    }

    /// Renders the active tool's panel, if it has one.
    pub fn panel_ui(&mut self, ctx: &egui::Context, selection: &mut SelectionManager) {
        let Some(i) = self.active else { return };
        let mut panel = PanelContext { selection };
        self.tools[i].lock().unwrap().panel_ui(ctx, &mut panel);
    }
}
