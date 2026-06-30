use std::sync::{Arc, Mutex, MutexGuard};

use duck_engine_viewer::event::{Event, EventContext, EventDispatcher};
use duck_engine_viewer::operator::{Operator, SelectionMode, SelectionOperator};
use duck_engine_viewer::scene::Scene;

use crate::cursor::Cursor3d;
use crate::tool::{ModelingTool, ToolInfo};

/// The single dispatcher-registered operator for all modeling tools.
/// 
/// Forwards events to the active tool, if any. Registered once at startup, in front
/// of the selection/navigation operators.
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
/// them.
/// 
/// Handles activation/deactivation, the always-on selection operator's
/// granularity, auto-return to selection when a tool finishes, and the 3D cursor.
/// Adding a tool to the modeler is implementing
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
    pub fn activate(&mut self, index: Option<usize>) {
        if index == self.active {
            return;
        }

        // Locks must be taken strictly one at a time
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
            None => SelectionMode::default(),
        };
        self.sel_op.lock().unwrap().mode = mode;

        self.active = index;
    }

    /// Per-frame update. Should be called every frame.
    pub fn update(&mut self, scene: &Arc<Mutex<Scene>>) {
        if self.active.is_some_and(|i| self.tools[i].lock().unwrap().is_finished()) {
            self.activate(None);
        }

        let target = self
            .active
            .and_then(|i| self.tools[i].lock().unwrap().cursor_target());
        self.cursor.update(target, &mut scene.lock().unwrap());
    }

    /// Palette snapshot for the `ui` module: `(info, selected)` per tool.
    /// Taken without holding any tool lock across egui rendering.
    pub fn palette_entries(&self) -> Vec<(ToolInfo, bool)> {
        self.tools
            .iter()
            .enumerate()
            .map(|(i, tool)| (tool.lock().unwrap().info(), self.active == Some(i)))
            .collect()
    }

    /// The active tool, locked for panel rendering, or `None` in selection mode.
    pub fn active_tool(&self) -> Option<MutexGuard<'_, dyn ModelingTool>> {
        self.active.map(|i| self.tools[i].lock().unwrap())
    }
}
