//! The modeler's panel UI: a thin orchestrator over docked egui panels.

mod icons;
mod right_panel;
mod scene_tab;

use duck_engine_viewer::selection::SelectionManager;

use crate::document::Document;

use right_panel::RightPanel;

/// Owns the modeler's persistent panel state.
#[derive(Default)]
pub struct ModelerUi {
    right: RightPanel,
}

impl ModelerUi {
    /// Render the panels for this frame.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        document: &mut Document,
        selection: &mut SelectionManager,
    ) {
        self.right.show(ctx, document, selection);
    }
}
