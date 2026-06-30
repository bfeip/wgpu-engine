//! The modeler's panel UI: a thin orchestrator over docked egui panels.

pub(crate) mod icons;
mod right_panel;
mod scene_tab;
mod tool_palette;
mod tool_panel;

use std::sync::{Arc, Mutex};

use duck_engine_viewer::selection::SelectionManager;

use crate::document::Document;
use crate::tool_manager::ToolManager;

use right_panel::RightPanel;
use tool_palette::ToolPalette;
use tool_panel::ToolPanel;

/// Owns the modeler's persistent panel state.
#[derive(Default)]
pub struct ModelerUi {
    palette: ToolPalette,
    right: RightPanel,
    tool_panel: ToolPanel,
}

impl ModelerUi {
    /// Render the panels for this frame.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        document: &Arc<Mutex<Document>>,
        selection: &mut SelectionManager,
        tools: &mut ToolManager,
    ) {
        self.palette.show(ctx, tools);
        {
            // The document lock must be released before drawing the tool panel,
            // which may also lock the document, causing a deadlock.
            // TODO: would be better to just pass the Arc instead of locking here.
            let mut document = document.lock().unwrap();
            self.right.show(ctx, &mut document, selection);
        }
        self.tool_panel.show(ctx, tools, selection);
    }
}
